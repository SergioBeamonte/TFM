"""Mini-estudio: efecto de las temperaturas de decodificación (chance/utility) en EMNA.

Réplica del estudio de temps de UMDA pero con fitness=margin (ganador de Grid 0 en
mode=both), fijando el resto a la best config: symmetric+non_symmetric, mode=both,
rules=20%, stop=top50. Barre Tc∈{0.5,1,2,5} × Tu∈{1,2,5}.

Reutiliza los helpers de grid_search.py para que el formato de CSV (resultados y
curvas) sea idéntico al del grid principal y lo lea el dashboard sin cambios.

Salidas:
  - example/bypass2/grid_search_results_EMNA_temps.csv
  - example/bypass2/grid_search_curves_EMNA_temps.csv
"""
import os
import time
import itertools

import numpy as np

import grid_search as gs


BASE_FOLDER = r"example\bypass2"
RESULTS_CSV = os.path.join(BASE_FOLDER, "grid_search_results_EMNA_temps.csv")
CURVES_CSV  = os.path.join(BASE_FOLDER, "grid_search_curves_EMNA_temps.csv")

BASE_CONFIG = {
    'xdsl_path': os.path.join(BASE_FOLDER, "network-bypass2.xdsl"),
    'rules_csv': os.path.join(BASE_FOLDER, "reglas_generadas.csv"),
    'min_max_ut': True,
    'u_range': (0, 10),
    'alpha': 0.5,
    'elite_factor': 0.0,
    'optimizer_type': 'emna',
}

# Grid de temperaturas (mismo esquema que UMDA: utility nunca 0.5; (1,1) es baseline)
CHANCE_TEMPERATURES  = [0.5, 1.0, 2.0, 5.0]
UTILITY_TEMPERATURES = [1.0, 2.0, 5.0]
MODES          = ['both']
SAMPLING_MODES = ['non_symmetric', 'symmetric']
FITNESS_TYPE   = 'margin'
STOP_MODE      = 'top50'
RULES_PCT      = 20

SIZE_GEN = 50
MAX_ITER = 60
TARGET_FITNESS = 1e-5
N_REPS = 5
BASE_SEED = 42


def main():
    total_rules = gs.count_total_rules(BASE_CONFIG['rules_csv'])
    n_rules = gs.compute_n_rules(total_rules, RULES_PCT)

    grid = list(itertools.product(
        MODES, SAMPLING_MODES, CHANCE_TEMPERATURES, UTILITY_TEMPERATURES))
    # Saltar baseline (1,1) que ya vive en el grid principal de EMNA.
    grid = [g for g in grid if not (g[2] == 1.0 and g[3] == 1.0)]

    gs.ensure_csv_header(RESULTS_CSV, gs.RESULTS_HEADER)
    gs.ensure_csv_header(CURVES_CSV, gs.CURVES_HEADER)
    completed = gs.get_completed_combinations(RESULTS_CSV)

    print("=" * 70)
    print("  EXPLORE TEMPS — EMNA")
    print(f"  fitness={FITNESS_TYPE} | rules={n_rules} ({RULES_PCT}%) | sg={SIZE_GEN}")
    print(f"  combos={len(grid)} x {N_REPS} reps | ya hechas={len(completed)}")
    print("=" * 70)

    t0 = time.time()
    for idx, (mode, sampling, tc, tu) in enumerate(grid, 1):
        key = (mode, sampling, f"{tc}", f"{tu}", FITNESS_TYPE, STOP_MODE, str(n_rules))
        if key in completed:
            print(f"[{idx}/{len(grid)}] SKIP (hecha) Tc={tc} Tu={tu} sampling={sampling}")
            continue

        print(f"\n[{idx}/{len(grid)}] mode={mode} sampling={sampling} "
              f"Tc={tc} Tu={tu} fitness={FITNESS_TYPE} rules={n_rules}")

        experiments, all_results = [], []
        for rep in range(N_REPS):
            seed = BASE_SEED + rep
            config = {
                **BASE_CONFIG,
                'n_decision_rules': n_rules,
                'fitness_type': FITNESS_TYPE,
                'stop_mode': STOP_MODE,
                'chance_temperature': tc,
                'utility_temperature': tu,
                'mode': mode,
                'symmetric_sampling': (sampling == 'symmetric'),
                'random_seed': seed,
            }
            exp, results = gs.run_single_experiment(config, SIZE_GEN, MAX_ITER, TARGET_FITNESS)
            experiments.append(exp)
            all_results.append(results)
            print(f"    rep {rep+1}/{N_REPS}: gen={results['stop_generation']} "
                  f"acc={results['best_accuracy']:.1f}%")

        stop_gens     = [r['stop_generation']  for r in all_results]
        best_fits     = [r['best_fitness']      for r in all_results]
        best_accs     = [r['best_accuracy']     for r in all_results]
        mse_chances   = [r['best_mse_chance']   for r in all_results]
        mse_utilities = [r['best_mse_utility']  for r in all_results]
        entropy_norms = [r['best_entropy_norm'] for r in all_results]
        util_devs     = [r['best_util_dev']     for r in all_results]

        row = {
            'mode': mode, 'sampling_mode': sampling,
            'chance_temperature': tc, 'utility_temperature': tu,
            'fitness_type': FITNESS_TYPE, 'stop_mode': STOP_MODE,
            'n_decision_rules': n_rules, 'n_decision_rules_pct': RULES_PCT,
            'total_rules': total_rules,
            'stop_gen_mean': f"{np.mean(stop_gens):.2f}",
            'stop_gen_std':  f"{np.std(stop_gens):.2f}",
            'stop_gen_min':  min(stop_gens), 'stop_gen_max': max(stop_gens),
            'fitness_mean': f"{np.mean(best_fits):.6f}",
            'fitness_std':  f"{np.std(best_fits):.6f}",
            'accuracy_mean': f"{np.mean(best_accs):.2f}",
            'accuracy_std':  f"{np.std(best_accs):.2f}",
            'accuracy_min':  f"{min(best_accs):.2f}",
            'accuracy_max':  f"{max(best_accs):.2f}",
            'mse_chance_mean': f"{np.mean(mse_chances):.6f}",
            'mse_chance_std':  f"{np.std(mse_chances):.6f}",
            'mse_utility_mean': f"{np.mean(mse_utilities):.6f}",
            'mse_utility_std':  f"{np.std(mse_utilities):.6f}",
            'entropy_norm_mean': f"{np.mean(entropy_norms):.6f}",
            'entropy_norm_std':  f"{np.std(entropy_norms):.6f}",
            'util_dev_mean': f"{np.mean(util_devs):.6f}",
            'util_dev_std':  f"{np.std(util_devs):.6f}",
        }
        gs.append_results_row(RESULTS_CSV, row)

        curve_rows = []
        for point in gs.average_histories(experiments):
            curve_rows.append({
                'mode': mode, 'sampling_mode': sampling,
                'chance_temperature': tc, 'utility_temperature': tu,
                'fitness_type': FITNESS_TYPE, 'stop_mode': STOP_MODE,
                'n_decision_rules': n_rules, 'n_decision_rules_pct': RULES_PCT,
                'generation': point['generation'],
                'mean_fitness': f"{point['mean_fitness']:.6f}",
                'mean_accuracy': f"{point['mean_accuracy']:.2f}",
                'mean_error_chance':  f"{point['mean_error_chance']:.6f}",
                'mean_error_utility': f"{point['mean_error_utility']:.6f}",
                'mean_entropy_norm':  f"{point['mean_entropy_norm']:.6f}",
                'mean_util_dev':      f"{point['mean_util_dev']:.6f}",
            })
        gs.append_curves_rows(CURVES_CSV, curve_rows)

        elapsed = time.time() - t0
        print(f"    OK ({elapsed/60:.1f} min acumulado)")

    print(f"\n=== DONE en {(time.time()-t0)/60:.1f} min ===")
    print(f"Resultados: {RESULTS_CSV}")
    print(f"Curvas:     {CURVES_CSV}")


if __name__ == '__main__':
    main()
