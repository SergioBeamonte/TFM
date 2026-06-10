"""Mini-estudio: efecto de alpha (truncación) y elite_factor (elitismo) en UMDA.

Hiperparámetros internos del EDA que NO están en el grid principal. Se fija el
resto a la "best config" encontrada en Grid 0 (UMDA + symmetric + margin +
mode=both + rules=20% + T=1) para aislar el efecto.

Salida: example/bypass2/explore_alpha_elite.csv
"""
import os, sys, time
import pandas as pd
import numpy as np
from id_recovery import IDRecovery


CONFIG = {
    'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
    'rules_csv': r'example\bypass2\reglas_generadas.csv',
    'min_max_ut': True,
    'u_range': (0, 10),
    'n_decision_rules': 4,           # 20%
    'fitness_type': 'margin',        # ganador de Grid 0 en mode=both
    'stop_mode': 'top50',
    'optimizer_type': 'umda',
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


def main():
    rows = []
    t0 = time.time()
    total = len(ALPHAS) * len(ELITES) * N_REPS
    done = 0

    for alpha in ALPHAS:
        for elite in ELITES:
            for rep in range(N_REPS):
                seed = BASE_SEED + rep
                exp = IDRecovery(
                    **CONFIG,
                    alpha=alpha,
                    elite_factor=elite,
                    random_seed=seed,
                )
                exp.run(g=SIZE_GEN, i=MAX_ITER, target_fitness=TARGET_FITNESS)

                if exp.history:
                    last = exp.history[-1]
                    gen_times = [float(h.get('gen_time', float('nan'))) for h in exp.history]
                    rows.append({
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
                        'gen_time_mean': float(np.nanmean(gen_times)) if gen_times else float('nan'),
                        'wall_time':     float(np.nansum(gen_times)) if gen_times else float('nan'),
                    })
                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done}/{total}] alpha={alpha} elite={elite} rep={rep} -> "
                      f"acc={rows[-1]['best_accuracy']:.0f}%  "
                      f"({elapsed:.0f}s / ETA {eta:.0f}s)")

    df = pd.DataFrame(rows)
    out = r'example\bypass2\explore_alpha_elite.csv'
    df.to_csv(out, index=False)
    print(f"\nGuardado: {out}")

    print("\n=== Resumen: best_accuracy por (alpha, elite_factor) ===")
    agg = df.groupby(['alpha', 'elite_factor'])['best_accuracy'].agg(['mean', 'std', 'max'])
    print(agg.round(2))

    print("\n=== Resumen: stop_generation por (alpha, elite_factor) ===")
    agg_gen = df.groupby(['alpha', 'elite_factor'])['stop_generation'].agg(['mean', 'std'])
    print(agg_gen.round(2))


if __name__ == '__main__':
    main()
