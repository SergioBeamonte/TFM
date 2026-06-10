"""Mini-estudio: efecto de porcentajes BAJOS de reglas en la red grande (nhlv1)
sobre utilidades (mode='utility_only') para los 4 EDAs.

Resto de parámetros = configuración "normal" del grid:
  - regret + symmetric + top50 + alpha=0.5 + elite=0 + T=1
  - 5 reps por celda
  - size_gen: 50 para UMDA/EMNA/EGNA, 400 para KEDA

Reutiliza los helpers de grid_search.py para que la salida sea idéntica en
formato al del grid principal (lo lee el dashboard sin tocar nada).

Salidas:
  - example/nhlv1/grid_search_results_<OPT>_rules.csv
  - example/nhlv1/grid_search_curves_<OPT>_rules.csv
"""
import os
import time
import itertools

import numpy as np

import grid_search as gs


BASE_FOLDER = r"example\nhlv1"
XDSL_PATH   = os.path.join(BASE_FOLDER, "network-nhlv1.xdsl")
RULES_CSV   = os.path.join(BASE_FOLDER, "reglas_generadas.csv")

BASE_CONFIG = {
    'xdsl_path': XDSL_PATH,
    'rules_csv': RULES_CSV,
    'min_max_ut': True,
    'u_range': (0, 10),
    'alpha': 0.5,
    'elite_factor': 0.0,
}

# Porcentajes BAJOS de reglas (la red tiene ~67 reglas; 5/10/15/20 → 3/7/10/13)
RULES_PERCENTAGES = [5, 10, 15, 20]
OPTIMIZERS = ['umda', 'emna', 'egna', 'keda']
FITNESS_TYPE   = 'regret'
STOP_MODE      = 'top50'
MODE           = 'utility_only'
SAMPLING       = 'non-simetric'
SIZE_GEN_PER_OPT = {'umda': 200, 'emna': 50, 'egna': 50, 'keda': 400}

CHANCE_T  = 1.0
UTILITY_T = 1.0

MAX_ITER       = 60
TARGET_FITNESS = 1e-5
N_REPS         = 5
BASE_SEED      = 42


def run_optimizer(opt):
    results_csv = os.path.join(BASE_FOLDER, f"grid_search_results_{opt.upper()}_rules.csv")
    curves_csv  = os.path.join(BASE_FOLDER, f"grid_search_curves_{opt.upper()}_rules.csv")

    total_rules = gs.count_total_rules(RULES_CSV)
    rules_vals  = [(pct, gs.compute_n_rules(total_rules, pct)) for pct in RULES_PERCENTAGES]

    gs.ensure_csv_header(results_csv, gs.RESULTS_HEADER)
    gs.ensure_csv_header(curves_csv, gs.CURVES_HEADER)
    completed = gs.get_completed_combinations(results_csv)

    sg = SIZE_GEN_PER_OPT[opt]
    print(f"\n{'='*70}\n  {opt.upper()}  | sg={sg} | rules pct={RULES_PERCENTAGES}\n{'='*70}")
    print(f"  Ya hechas: {len(completed)} / {len(rules_vals)}")

    t0 = time.time()
    for idx, (pct, n_rules) in enumerate(rules_vals, 1):
        key = (MODE, SAMPLING, f"{CHANCE_T}", f"{UTILITY_T}", FITNESS_TYPE, STOP_MODE, str(n_rules))
        if key in completed:
            print(f"[{idx}/{len(rules_vals)}] SKIP pct={pct} n_rules={n_rules}")
            continue
        print(f"\n[{idx}/{len(rules_vals)}] {opt} | pct={pct} | n_rules={n_rules}")

        experiments, all_results = [], []
        for rep in range(N_REPS):
            seed = BASE_SEED + rep
            config = {
                **BASE_CONFIG,
                'n_decision_rules': n_rules,
                'fitness_type': FITNESS_TYPE,
                'stop_mode': STOP_MODE,
                'chance_temperature': CHANCE_T,
                'utility_temperature': UTILITY_T,
                'mode': MODE,
                'symmetric_sampling': (SAMPLING == 'symmetric'),
                'optimizer_type': opt,
                'random_seed': seed,
            }
            exp, results = gs.run_single_experiment(config, sg, MAX_ITER, TARGET_FITNESS)
            experiments.append(exp); all_results.append(results)
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
            'mode': MODE, 'sampling_mode': SAMPLING,
            'chance_temperature': CHANCE_T, 'utility_temperature': UTILITY_T,
            'fitness_type': FITNESS_TYPE, 'stop_mode': STOP_MODE,
            'n_decision_rules': n_rules, 'n_decision_rules_pct': pct,
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
        gs.append_results_row(results_csv, row)

        curve_rows = []
        for point in gs.average_histories(experiments):
            curve_rows.append({
                'mode': MODE, 'sampling_mode': SAMPLING,
                'chance_temperature': CHANCE_T, 'utility_temperature': UTILITY_T,
                'fitness_type': FITNESS_TYPE, 'stop_mode': STOP_MODE,
                'n_decision_rules': n_rules, 'n_decision_rules_pct': pct,
                'generation': point['generation'],
                'mean_fitness': f"{point['mean_fitness']:.6f}",
                'mean_accuracy': f"{point['mean_accuracy']:.2f}",
                'mean_error_chance':  f"{point['mean_error_chance']:.6f}",
                'mean_error_utility': f"{point['mean_error_utility']:.6f}",
                'mean_entropy_norm':  f"{point['mean_entropy_norm']:.6f}",
                'mean_util_dev':      f"{point['mean_util_dev']:.6f}",
            })
        gs.append_curves_rows(curves_csv, curve_rows)

        print(f"    OK ({(time.time()-t0)/60:.1f}min acumulado en {opt})")


def main():
    t0 = time.time()
    for opt in OPTIMIZERS:
        run_optimizer(opt)
    print(f"\n=== TOTAL: {(time.time()-t0)/60:.1f} min ===")


if __name__ == '__main__':
    main()
