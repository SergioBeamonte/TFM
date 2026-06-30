"""Acto 4.1 - ESCALADO a la red grande (nhlv1, utility_only, ~67 reglas).

Pregunta: el patron observado en bypass2 (generalizacion temprana + nº de
reglas necesarias) ¿sobrevive al aumentar el tamaño de la red?

Decision metodologica: NO repetimos el barrido completo. Los Actos 1-3 ya
fijaron la configuracion ganadora, asi que escalamos SOLO esa:
  - fitness = regret  (el mejor transversalmente)
  - min_max_ut = True, alpha=0.5, elite=0, T=1
  - optimizadores = UMDA, EMNA, KEDA  (EGNA descartado: mismo resultado, ~22x CPU)

Corremos los DOS disparadores de curriculum sobre la misma red, para comparar
directamente con bypass2:
  - 'success'    (estilo explore_incremental): añade regla al cumplir top50.
  - 'stagnation' (estilo explore_saturation): añade regla tras 5 gens sin
                  mejorar; corta cuando el 50% de individuos son perfectos.

Salida: example/explore_scale_nhlv1.csv (1 fila por trigger, eda, rep, gen).
"""
import os
import time
import numpy as np
import pandas as pd

from id_recovery import IDRecovery


NET = {
    'name': 'nhlv1',
    'xdsl_path': r'example\nhlv1\network-nhlv1.xdsl',
    'rules_csv': r'example\nhlv1\reglas_generadas.csv',
    'mode': 'utility_only',
}

OPTIMIZERS = ['umda', 'emna']                # EGNA y KEDA fuera por coste en nhlv1 (keda=400pob -> ~28h)
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'keda': 400}
TRIGGERS = ['success', 'stagnation']
FITNESS = 'regret'

COMMON = {
    'u_range':            (0, 10),
    'min_max_ut':         True,
    'alpha':              0.5,
    'elite_factor':       0.0,
    'fitness_type':       FITNESS,
    'stop_mode':          'top50',
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
    'incremental_rules':    True,
    'incremental_start_with': 1,
    'incremental_patience': 5,
}

STOP_PERFECT_RATIO = 0.50     # corte para el trigger 'stagnation'
MAX_ITER       = 100   # nhlv1 es lento (~70x bypass2); la parada por exito/estancamiento suele llegar antes
TARGET_FITNESS = 1e-5
N_REPS         = 5            # menos reps: la red es cara
BASE_SEED      = 42

RAW_CSV = r'example\explore_scale_nhlv1.csv'


def run_one(trigger, optimizer, rep, seed):
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': NET['xdsl_path'],
        'rules_csv': NET['rules_csv'],
        'mode':      NET['mode'],
        'n_decision_rules': -1,
        'optimizer_type':   optimizer,
        'random_seed':      seed,
        'incremental_trigger': trigger,
    })
    exp = IDRecovery(**params)

    # Corte final comun a ambos triggers: cuando el 50% de individuos son
    # perfectos (criterio comparable). La diferencia entre 'success' y
    # 'stagnation' esta en COMO id_recovery decide añadir cada regla.
    original_fitness = exp.fitness
    def custom_fitness(v):
        val = original_fitness(v)
        if exp.evals_this_gen == 0 and len(exp.history) > 0:
            accs = exp.history[-1]['accuracies']
            if np.sum(accs >= 99.999) / len(accs) >= STOP_PERFECT_RATIO:
                raise StopIteration("50% individuos perfectos")
        return val
    exp.fitness = custom_fitness

    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS, patience=float('inf'))
    except (ValueError, np.linalg.LinAlgError) as e:
        n_done = len(exp.history)
        print(f"    !! {trigger}/{optimizer} rep={rep} "
              f"{type(e).__name__} tras gen={n_done} (se conservan {n_done} gens)")
    if not exp.history:
        return []

    rows, total_rules = [], len(exp.all_rules)
    for h in exp.history:
        accs = h['accuracies']
        rows.append({
            'net':            NET['name'],
            'trigger':        trigger,
            'optimizer':      optimizer,
            'rep':            rep,
            'fitness_type':   FITNESS,
            'seed':           seed,
            'size_gen':       sg,
            'mode':           NET['mode'],
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
        done = set(zip(df['trigger'], df['optimizer'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(t, o, r) for t in TRIGGERS for o in OPTIMIZERS for r in range(N_REPS)]
    total = len(combos)
    todo  = total - len(done)
    print(f"Plan nhlv1: {total} corridas ({todo} pendientes, {len(done)} hechas)")

    t0, completed_now = time.time(), 0
    for trig, opt, rep in combos:
        if (trig, opt, rep) in done:
            continue
        seed = BASE_SEED + rep
        new_rows = run_one(trig, opt, rep, seed)
        completed_now += 1
        if new_rows:
            last = new_rows[-1]
            rules_needed = last['n_train_rules']
            n_gens = last['gen']
            perfect = last['pct_success_indv']
            nrc = last['n_rules_correct']; tot = last['total_rules']
            rows.extend(new_rows)
        else:
            rules_needed = n_gens = perfect = nrc = tot = 0
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {trig}/{opt} rep={rep}: "
              f"reglas={rules_needed} gens={n_gens} nrc={nrc}/{tot} "
              f"perfectos={perfect:.0f}%  ({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} corridas en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    last = df.sort_values('gen').groupby(['trigger','optimizer','rep']).tail(1)
    print("\n=== nhlv1: reglas necesarias y nrc final por (trigger, optimizer) ===")
    print(last.groupby(['trigger','optimizer'])[['n_train_rules','n_rules_correct','gen']]
              .mean().round(1).to_string())


if __name__ == '__main__':
    main()
