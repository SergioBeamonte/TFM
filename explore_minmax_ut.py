"""Mini-estudio: efecto de fijar la mejor/peor consecuencia (min_max_ut)
en utilidades.

Compara `min_max_ut=True` (anclamos los extremos a u_max/u_min y la EDA optimiza
solo el resto) vs `min_max_ut=False` (la EDA optimiza todas las entradas de la
tabla, incluyendo extremos).

Configuración:
  - Redes:        bypass2 (mode='both') y nhlv1 (mode='utility_only')
  - Optimizadores: UMDA, EMNA, KEDA (con SIZE_GEN_PER_OPTIMIZER del grid)
  - Resto:        regret, symmetric, top50, alpha=0.5, elite=0, T=1, rules=20%
  - 10 reps por celda  →  2 × 3 × 2 × 10 = 120 runs

Salidas:
  - example/explore_minmax_ut.csv  (raw, 1 fila por rep)
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
        'total_rules': 20,
    },
    'nhlv1': {
        'xdsl_path': r'example\nhlv1\network-nhlv1.xdsl',
        'rules_csv': r'example\nhlv1\reglas_generadas.csv',
        'mode': 'utility_only',
        'total_rules': 67,
    },
}

OPTIMIZERS = ['umda', 'emna', 'keda', 'egna']

# Mismos overrides de tamaño que en grid_search.py (KEDA necesita más por la
# covarianza singular en gaussian_kde).
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}

COMMON = {
    'u_range':            (0, 10),
    'alpha':              0.5,
    'elite_factor':       0.0,
    'fitness_type':       'binary',
    'stop_mode':          'top50',
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
}

N_REPS         = 10
MAX_ITER       = 60
TARGET_FITNESS = 1e-5
BASE_SEED      = 42
RULES_PCT      = 10

RAW_CSV = r'example\explore_minmax_ut.csv'


def n_rules_for(total_rules, pct):
    return max(1, round(total_rules * pct / 100))


def run_one(net_name, optimizer, min_max_ut, rep, seed):
    cfg = NETS[net_name]
    n_rules = n_rules_for(cfg['total_rules'], RULES_PCT)
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': cfg['xdsl_path'],
        'rules_csv': cfg['rules_csv'],
        'mode':      cfg['mode'],
        'n_decision_rules': n_rules,
        'min_max_ut':       min_max_ut,
        'optimizer_type':   optimizer,
        'random_seed':      seed,
    })
    exp = IDRecovery(**params)
    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {net_name}/{optimizer}/mmu={min_max_ut} rep={rep} ABORT: "
              f"{type(e).__name__}: {e}")
        return None
    if not exp.history:
        return None
    last = exp.history[-1]
    gen_cpus = [float(h.get('gen_cpu_time', float('nan'))) for h in exp.history]
    return {
        'net':            net_name,
        'optimizer':      optimizer,
        'min_max_ut':     bool(min_max_ut),
        'rep':            rep,
        'seed':           seed,
        'size_gen':       sg,
        'n_decision_rules': n_rules,
        'mode':           cfg['mode'],
        'stop_generation': last['gen'],
        'best_fitness':    float(np.min(last['fitness'])),
        'best_accuracy':   float(np.max(last['accuracies'])),
        'mean_accuracy':   float(np.mean(last['accuracies'])),
        'mse_chance':      float(np.min(last['errors_chance'])),
        'mse_utility':     float(np.min(last['errors_utility'])),
        'cpu_per_gen':     float(np.nanmean(gen_cpus)) if gen_cpus else float('nan'),
        'cpu_total':       float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df['net'], df['optimizer'], df['min_max_ut'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    print(f"Cargados {len(rows)} runs previos; pendientes...")
    combos = [(n, o, m, r)
              for n in NETS
              for o in OPTIMIZERS
              for m in (True, False)
              for r in range(N_REPS)]
    total = len(combos)
    todo = total - len(done)
    print(f"Total plan: {total} runs ({todo} pendientes)")

    t0 = time.time()
    completed_now = 0
    for net_name, opt, mmu, rep in combos:
        if (net_name, opt, mmu, rep) in done:
            continue
        seed = BASE_SEED + rep
        r = run_one(net_name, opt, mmu, rep, seed)
        completed_now += 1
        if r is not None:
            rows.append(r)
            acc = r['best_accuracy']; gen = r['stop_generation']
        else:
            acc, gen = float('nan'), 0
        elapsed = time.time() - t0
        eta = elapsed / completed_now * (todo - completed_now) if completed_now else 0
        print(f"[{completed_now}/{todo}] {net_name}/{opt}/mmu={mmu} rep={rep}: "
              f"acc={acc:.0f}% gen={gen}  ({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed_now} runs nuevos en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    print("\n=== Accuracy media por (net, optimizer, min_max_ut) ===")
    print(df.pivot_table(values='best_accuracy',
                          index=['net', 'optimizer'],
                          columns='min_max_ut', aggfunc='mean').round(1))
    print("\n=== MSE utility media por (net, optimizer, min_max_ut) ===")
    print(df.pivot_table(values='mse_utility',
                          index=['net', 'optimizer'],
                          columns='min_max_ut', aggfunc='mean').round(2))
    print("\n=== Stop gen media por (net, optimizer, min_max_ut) ===")
    print(df.pivot_table(values='stop_generation',
                          index=['net', 'optimizer'],
                          columns='min_max_ut', aggfunc='mean').round(1))


if __name__ == '__main__':
    main()
