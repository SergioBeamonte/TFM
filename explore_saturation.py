"""Estudio: SATURACION por estancamiento (cuantas reglas se NECESITAN).

Hermano del curriculum incremental, pero con disparador distinto:

  - explore_incremental: se añade una regla cuando el modelo HA APRENDIDO
    (el stop_mode se cumple). Mide nº de GENERACIONES.
  - explore_saturation (este): se añade una regla cuando el modelo DEJA DE
    APRENDER: `incremental_patience` generaciones seguidas sin mejorar la
    accuracy real (sobre TODAS las reglas). Mide nº de REGLAS necesarias,
    independientemente de cuantas generaciones cueste.

Arrancamos con 1 regla. Cada vez que la accuracy real se estanca 5 gens,
metemos otra regla aleatoria (le damos mas señal para escapar del plateau).
La corrida se corta cuando TODAS las reglas reales se cumplen para el 50%
de los individuos de la poblacion.

El "punto gordo" (rule_added_after_gen) marca cada adicion: si tras el
punto la accuracy salta, la regla era necesaria; si no salta, el plateau
era del optimizador.

Salida: example/explore_saturation.csv (1 fila por red, eda, rep, fit, gen).
Solo bypass2 (nhlv1 descartado por coste, igual que en el incremental).
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

COMMON = {
    'u_range':            (0, 10),
    'min_max_ut':         True,
    'alpha':              0.5,
    'elite_factor':       0.0,
    'stop_mode':          'top50',   # informativo: el disparo real es por estancamiento
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
    'incremental_rules':    True,
    'incremental_start_with': 1,
    'incremental_trigger':  'stagnation',
    'incremental_patience': 5,        # 5 gens sin mejorar accuracy -> +1 regla
}

# Fraccion de individuos que deben cumplir TODAS las reglas reales para cortar.
STOP_PERFECT_RATIO = 0.50

MAX_ITER       = 500
TARGET_FITNESS = 1e-5
N_REPS         = 10
BASE_SEED      = 42

RAW_CSV = r'example\explore_saturation.csv'


def run_one(net_name, optimizer, fitness_type, rep, seed):
    cfg = NETS[net_name]
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': cfg['xdsl_path'],
        'rules_csv': cfg['rules_csv'],
        'mode':      cfg['mode'],
        'fitness_type':     fitness_type,
        'n_decision_rules': -1,       # pool = todas las reglas
        'optimizer_type':   optimizer,
        'random_seed':      seed,
    })
    exp = IDRecovery(**params)

    original_fitness = exp.fitness
    def custom_fitness(v):
        val = original_fitness(v)
        # Justo al cerrar una generacion: evals_this_gen vuelve a 0 y hay history.
        if exp.evals_this_gen == 0 and len(exp.history) > 0:
            accs = exp.history[-1]['accuracies']
            perfect_ratio = np.sum(accs >= 99.999) / len(accs)
            if perfect_ratio >= STOP_PERFECT_RATIO:
                raise StopIteration(
                    f"El {STOP_PERFECT_RATIO*100:.0f}% de los individuos cumple "
                    f"TODAS las reglas reales.")
        return val
    exp.fitness = custom_fitness

    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS, patience=float('inf'))
    except (ValueError, np.linalg.LinAlgError) as e:
        # KEDA puede reventar por covarianza singular tras converger: conservamos
        # la historia acumulada (es valida hasta ese punto).
        n_done = len(exp.history)
        print(f"    !! {net_name}/{optimizer}/{fitness_type} rep={rep} "
              f"{type(e).__name__} tras gen={n_done} (se conservan esas {n_done} gens)")
    if not exp.history:
        return []

    rows = []
    total_rules = len(exp.all_rules)
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
    print(f"Plan: {total} corridas ({todo} pendientes, {len(done)} hechas)")
    print(f"Filas en CSV previo: {len(rows)}")

    t0 = time.time()
    completed_now = 0
    for net_name, opt, fit, rep in combos:
        if (net_name, opt, fit, rep) in done:
            continue
        seed = BASE_SEED + rep
        new_rows = run_one(net_name, opt, fit, rep, seed)
        completed_now += 1
        if new_rows:
            rows.extend(new_rows)
            last = new_rows[-1]
            final_acc = last['mean_accuracy']
            rules_needed = last['n_train_rules']
            n_gens = last['gen']
            n_events = sum(1 for r in new_rows if r['rule_added_after_gen'])
            final_perfect = last['pct_success_indv']
        else:
            final_acc, rules_needed, n_gens, n_events, final_perfect = (
                float('nan'), 0, 0, 0, float('nan'))
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {net_name}/{opt}/{fit} rep={rep}: "
              f"REGLAS_NECESARIAS={rules_needed} gens={n_gens} eventos={n_events} "
              f"perfectos={final_perfect:.0f}% mean_acc={final_acc:.0f}%  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} corridas nuevas en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    last = df.sort_values('gen').groupby(['net','optimizer','fitness_type','rep']).tail(1)
    print("\n=== REGLAS NECESARIAS (media) por (optimizer, fitness) ===")
    print(last.groupby(['optimizer','fitness_type'])['n_train_rules'].mean()
              .unstack().round(1).to_string())


if __name__ == '__main__':
    main()
