"""Estudio de CAPACIDAD: hasta donde llega cada regla.

Idea (dos fitness separados):
  - FITNESS DE ENTRENAMIENTO: lo que el EDA minimiza, calculado SOLO sobre las
    reglas de entrenamiento (regret/binary/...). El modelo no ve las demas.
  - SEÑAL DE SATURACION: accuracy GLOBAL = % de TODAS las reglas reales que
    cumple el mejor individuo (el modelo NO la optimiza, solo la observamos).

Curriculum:
  - Arrancamos con 1 regla ALEATORIA.
  - Entrenamos. Cada generacion observamos la accuracy global del mejor individuo.
  - Cuando la accuracy global se ESTANCA o EMPEORA durante `patience` generaciones
    (no hay nuevo maximo global), añadimos otra regla ALEATORIA del pool: le damos
    mas supervision para ver si puede recuperar mas reglas reales.
  - Paramos cuando el mejor individuo recupera TODAS las reglas reales (global=100%)
    O cuando, agotado el pool, la accuracy global vuelve a estancarse (capacidad
    saturada). Tope MAX_ITER de seguridad.

Asi medimos la CURVA DE CAPACIDAD: nº de reglas reales recuperadas (n_rules_correct)
frente al nº de reglas de entrenamiento dadas, con "puntos gordos" en cada adicion.
Si tras un punto gordo la accuracy salta, esa regla aportaba; si no, no; si baja,
la nueva regla estorbaba.

Plan: bypass2 (mode='both') x 4 EDAs x 5 fitness x 10 reps = 200 corridas.
Salida: example/explore_capacity.csv (1 fila por corrida, gen).
"""
import os
import time
import numpy as np
import pandas as pd

from id_recovery import IDRecovery


NETS = {
    'bypass2': {
        'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
        'rules_csv': r'example\bypass2\reglas_generadas.csv',
        'mode': 'both',
    },
}

OPTIMIZERS = ['umda', 'emna', 'egna', 'keda']
FITNESS_TYPES = ['binary', 'margin', 'softmax', 'regret', 'entropy']
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}

PATIENCE = 10            # gens sin nuevo maximo global -> +1 regla

COMMON = {
    'u_range':            (0, 10),
    'min_max_ut':         True,
    'alpha':              0.5,
    'elite_factor':       0.0,
    'stop_mode':          'top50',   # informativo; el disparo real es por estancamiento global
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
    'incremental_rules':    True,
    'incremental_start_with': 1,        # 1ª regla aleatoria
    'incremental_trigger':  'stagnation',
    'incremental_patience': PATIENCE,   # estancamiento de accuracy GLOBAL
}

GLOBAL_FULL = 99.999     # mejor individuo recupera TODAS las reglas reales
MAX_ITER       = 200
TARGET_FITNESS = 1e-5
N_REPS         = 10
BASE_SEED      = 42

RAW_CSV = r'example\explore_capacity.csv'


def run_one(net_name, optimizer, fitness_type, rep, seed):
    cfg = NETS[net_name]
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': cfg['xdsl_path'],
        'rules_csv': cfg['rules_csv'],
        'mode':      cfg['mode'],
        'fitness_type':     fitness_type,
        'n_decision_rules': -1,           # pool = todas las reglas
        'optimizer_type':   optimizer,
        'random_seed':      seed,
    })
    exp = IDRecovery(**params)

    original_fitness = exp.fitness
    def custom_fitness(v):
        val = original_fitness(v)
        if exp.evals_this_gen == 0 and len(exp.history) > 0:
            max_acc = float(np.max(exp.history[-1]['accuracies']))
            # 1) Capacidad plena: el mejor individuo ya recupera TODAS las reales.
            if max_acc >= GLOBAL_FULL:
                raise StopIteration("Capacidad plena: best recupera todas las reglas reales")
            # 2) Pool agotado y accuracy global terminalmente estancada.
            if not exp.train_pool_remaining and exp.acc_stagnation_counter >= PATIENCE:
                raise StopIteration("Pool agotado y accuracy global estancada (capacidad saturada)")
        return val
    exp.fitness = custom_fitness

    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS, patience=float('inf'))
    except (ValueError, np.linalg.LinAlgError) as e:
        n_done = len(exp.history)
        print(f"    !! {net_name}/{optimizer}/{fitness_type} rep={rep} "
              f"{type(e).__name__} tras gen={n_done} (se conservan esas {n_done} gens)")
    if not exp.history:
        return []

    rows, total_rules = [], len(exp.all_rules)
    for h in exp.history:
        accs = h['accuracies']
        rows.append({
            'net':            net_name,
            'optimizer':      optimizer,
            'rep':            rep,
            'fitness_type':   fitness_type,
            'seed':           seed,
            'size_gen':       sg,
            'mode':           cfg['mode'],
            'total_rules':    total_rules,
            'gen':            int(h['gen']),
            'n_train_rules':  int(h['n_train_rules']),
            'rule_added_after_gen': bool(h.get('rule_added_after_gen', False)),
            'max_accuracy':   float(np.max(accs)),
            'n_rules_correct': int(round(np.max(accs) / 100.0 * total_rules)),
            'mean_accuracy':  float(np.mean(accs)),
            'pct_success_indv': float(np.sum(accs >= 99.999) / len(accs) * 100.0),
            'gen_cpu_time':   float(h.get('gen_cpu_time', float('nan'))),
        })
    return rows


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df['net'], df['optimizer'], df['fitness_type'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(n, o, f, r) for n in NETS
                            for o in OPTIMIZERS
                            for f in FITNESS_TYPES
                            for r in range(N_REPS)]
    total = len(combos)
    todo  = total - len(done)
    print(f"Plan capacidad: {total} corridas ({todo} pendientes, {len(done)} hechas)")

    t0, completed_now = time.time(), 0
    for net_name, opt, fit, rep in combos:
        if (net_name, opt, fit, rep) in done:
            continue
        seed = BASE_SEED + rep
        new_rows = run_one(net_name, opt, fit, rep, seed)
        completed_now += 1
        if new_rows:
            rows.extend(new_rows)
            last = new_rows[-1]
            rules_given = last['n_train_rules']
            nrc = last['n_rules_correct']; tot = last['total_rules']
            n_gens = last['gen']
            n_events = sum(1 for r in new_rows if r['rule_added_after_gen'])
        else:
            rules_given = nrc = tot = n_gens = n_events = 0
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {net_name}/{opt}/{fit} rep={rep}: "
              f"reglas_train={rules_given} recuperadas={nrc}/{tot} gens={n_gens} "
              f"adiciones={n_events}  ({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} corridas en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    last = df.sort_values('gen').groupby(['net','optimizer','fitness_type','rep']).tail(1)
    print("\n=== CAPACIDAD: reglas de entrenamiento necesarias para recuperar todas ===")
    print(last.groupby(['optimizer','fitness_type'])[['n_train_rules','n_rules_correct']]
              .mean().round(1).to_string())


if __name__ == '__main__':
    main()
