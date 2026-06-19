"""Mini-estudio: efecto de alpha (truncación) y elite_factor (elitismo) en EMNA.

Réplica del estudio de UMDA (explore_alpha_elite.py) para EMNA, fijando el resto
a la "best config" de Grid 0 (symmetric + margin + mode=both + rules=20% + T=1).

EMNA estima una Gaussiana multivariante completa; con poblaciones truncadas muy
pequeñas la covarianza puede degenerar, así que cada run va protegido con
try/except y se marca NaN si aborta (como en explore_size_gen.py).

Salidas:
  - example/bypass2/explore_alpha_elite_emna.csv             (raw, 1 fila/rep)
  - example/bypass2/grid_search_results_EMNA_alpha_elite.csv (agregado dashboard)
"""
import os
import time

import numpy as np
import pandas as pd

from id_recovery import IDRecovery


CONFIG = {
    'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
    'rules_csv': r'example\bypass2\reglas_generadas.csv',
    'min_max_ut': True,
    'u_range': (0, 10),
    'n_decision_rules': 4,           # 20%
    'fitness_type': 'margin',        # ganador de Grid 0 en mode=both
    'stop_mode': 'top50',
    'optimizer_type': 'emna',
    'mode': 'both',
    'symmetric_sampling': True,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
}

ALPHAS = [0.25, 0.5, 0.75]
ELITES = [0.0, 0.1, 0.2]
N_REPS = 5
SIZE_GEN = 50
MAX_ITER = 60
TARGET_FITNESS = 1e-5
BASE_SEED = 42

RAW_CSV = r'example\bypass2\explore_alpha_elite_emna.csv'
AGG_CSV = r'example\bypass2\grid_search_results_EMNA_alpha_elite.csv'

# Constantes que el dashboard espera en el agregado (mismas columnas que UMDA).
CONSTANTS_FOR_AGG = {
    'mode':                 'both',
    'sampling_mode':        'symmetric',
    'fitness_type':         'margin',
    'stop_mode':            'top50',
    'n_decision_rules':     4,
    'n_decision_rules_pct': 20,
    'total_rules':          20,
    'chance_temperature':   1.0,
    'utility_temperature':  1.0,
}

AGG_COLS = [
    'mode', 'sampling_mode', 'fitness_type', 'stop_mode', 'n_decision_rules',
    'n_decision_rules_pct', 'total_rules', 'chance_temperature',
    'utility_temperature', 'alpha', 'elite_factor',
    'stop_gen_mean', 'stop_gen_std', 'stop_gen_min', 'stop_gen_max',
    'fitness_mean', 'fitness_std', 'accuracy_mean', 'accuracy_std',
    'accuracy_min', 'accuracy_max', 'mse_chance_mean', 'mse_chance_std',
    'mse_utility_mean', 'mse_utility_std', 'entropy_norm_mean',
    'entropy_norm_std', 'util_dev_mean', 'util_dev_std',
]


def run_one(alpha, elite, rep, seed):
    exp = IDRecovery(**CONFIG, alpha=alpha, elite_factor=elite, random_seed=seed)
    try:
        exp.run(g=SIZE_GEN, i=MAX_ITER, target_fitness=TARGET_FITNESS)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! alpha={alpha} elite={elite} rep={rep} aborted: "
              f"{type(e).__name__}: {e}")
        return None
    if not exp.history:
        return None
    last = exp.history[-1]
    gen_cpus = [float(h.get('gen_cpu_time', float('nan'))) for h in exp.history]
    return {
        'alpha': alpha,
        'elite_factor': elite,
        'rep': rep,
        'seed': seed,
        'stop_generation': last['gen'],
        'best_fitness': float(np.min(last['fitness'])),
        'best_accuracy': float(np.max(last['accuracies'])),
        'mean_accuracy': float(np.mean(last['accuracies'])),
        'mse_chance':  float(np.min(last['errors_chance'])),
        'mse_utility': float(np.min(last['errors_utility'])),
        'cpu_per_gen': float(np.nanmean(gen_cpus)) if gen_cpus else float('nan'),
        'cpu_total':   float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }


def aggregate_and_save(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    agg_rows = []
    for (alpha, elite), g in df.groupby(['alpha', 'elite_factor']):
        row = dict(CONSTANTS_FOR_AGG)
        row['alpha'] = alpha
        row['elite_factor'] = elite
        row['stop_gen_mean'] = g['stop_generation'].mean()
        row['stop_gen_std']  = g['stop_generation'].std()
        row['stop_gen_min']  = g['stop_generation'].min()
        row['stop_gen_max']  = g['stop_generation'].max()
        row['fitness_mean']  = g['best_fitness'].mean()
        row['fitness_std']   = g['best_fitness'].std()
        row['accuracy_mean'] = g['best_accuracy'].mean()
        row['accuracy_std']  = g['best_accuracy'].std()
        row['accuracy_min']  = g['best_accuracy'].min()
        row['accuracy_max']  = g['best_accuracy'].max()
        row['mse_chance_mean']  = g['mse_chance'].mean()
        row['mse_chance_std']   = g['mse_chance'].std()
        row['mse_utility_mean'] = g['mse_utility'].mean()
        row['mse_utility_std']  = g['mse_utility'].std()
        row['entropy_norm_mean'] = float('nan')
        row['entropy_norm_std']  = float('nan')
        row['util_dev_mean']     = float('nan')
        row['util_dev_std']      = float('nan')
        agg_rows.append(row)
    pd.DataFrame(agg_rows)[AGG_COLS].to_csv(AGG_CSV, index=False)


def main():
    rows = []
    t0 = time.time()
    total = len(ALPHAS) * len(ELITES) * N_REPS
    done = 0

    for alpha in ALPHAS:
        for elite in ELITES:
            for rep in range(N_REPS):
                seed = BASE_SEED + rep
                r = run_one(alpha, elite, rep, seed)
                done += 1
                if r is not None:
                    rows.append(r)
                    acc = r['best_accuracy']
                else:
                    acc = float('nan')
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] alpha={alpha} elite={elite} rep={rep} -> "
                      f"acc={acc:.0f}%  ({elapsed:.0f}s / ETA {eta:.0f}s)")
                pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    aggregate_and_save(rows)
    print(f"\nGuardado raw: {RAW_CSV}")
    print(f"Guardado agg: {AGG_CSV}")

    df = pd.DataFrame(rows)
    if not df.empty:
        print("\n=== best_accuracy por (alpha, elite_factor) ===")
        print(df.groupby(['alpha', 'elite_factor'])['best_accuracy']
              .agg(['mean', 'std', 'max']).round(2))
        print("\n=== stop_generation por (alpha, elite_factor) ===")
        print(df.groupby(['alpha', 'elite_factor'])['stop_generation']
              .agg(['mean', 'std']).round(2))


if __name__ == '__main__':
    main()
