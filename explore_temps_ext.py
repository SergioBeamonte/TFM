"""Extiende el estudio de temperaturas:
  (1) añade una temperatura MÁS ALTA (10.0) a Tc y Tu;
  (2) añade el optimizador KEDA (CSV nuevos).

Reutiliza los helpers de grid_search. Como el esquema de RESULTS/CURVES creció
(se añadieron columnas cpu_*), primero MIGRA los CSV de temps existentes al
esquema nuevo (rellena las columnas que faltan con vacío) para no recorromperlos
al appendear. El skip-completed evita re-ejecutar las celdas ya hechas: solo se
corren las combinaciones con Tc=10 o Tu=10 (UMDA/EMNA) y toda la malla para KEDA.

KEDA puede abortar por covarianza singular (colapso del KDE): cada rep va
protegida con try/except; si todas fallan, se escribe una fila NaN para registrar
el hueco. KEDA usa size_gen alto (la covarianza del KDE necesita n > nº de vars).

Salidas (se appendea / se crean):
  - example/bypass2/grid_search_results_{UMDA,EMNA,KEDA}_temps.csv (+ _curves_)
"""
import os
import csv
import time
import itertools

import numpy as np

import grid_search as gs


BASE_FOLDER = r"example\bypass2"

BASE_CONFIG = {
    'xdsl_path': os.path.join(BASE_FOLDER, "network-bypass2.xdsl"),
    'rules_csv': os.path.join(BASE_FOLDER, "reglas_generadas.csv"),
    'min_max_ut': True,
    'u_range': (0, 10),
    'alpha': 0.5,
    'elite_factor': 0.0,
}

# Malla extendida: se añade 10.0 (la novedad de esta tanda).
CHANCE_TEMPERATURES  = [0.5, 1.0, 2.0, 5.0, 10.0]
UTILITY_TEMPERATURES = [1.0, 2.0, 5.0, 10.0]
MODES          = ['both']
SAMPLING_MODES = ['non_symmetric', 'symmetric']
STOP_MODE      = 'top50'
RULES_PCT      = 20

MAX_ITER = 60
TARGET_FITNESS = 1e-5
N_REPS = 5
BASE_SEED = 42

# Config por optimizador: fitness a barrer y tamaño de población.
# KEDA necesita size_gen alto (KDE: la covarianza es singular si n <= nº variables).
OPT_CONFIG = {
    'umda': {'fitness': ['regret', 'binary'],            'size_gen': 50,  'tag': 'UMDA'},
    'emna': {'fitness': ['margin', 'binary'],            'size_gen': 50,  'tag': 'EMNA'},
    'keda': {'fitness': ['regret', 'binary', 'margin'],  'size_gen': 400, 'tag': 'KEDA'},
}


def _paths(tag):
    return (os.path.join(BASE_FOLDER, f"grid_search_results_{tag}_temps.csv"),
            os.path.join(BASE_FOLDER, f"grid_search_curves_{tag}_temps.csv"))


def migrate_to_new_schema(path, header):
    """Si el CSV existe con un header más corto que el actual, lo rellena con
    columnas vacías al final para que coincida con `header` (las columnas nuevas
    van siempre al final)."""
    if not os.path.exists(path):
        return
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) == len(header):
        return
    W = len(header)
    out = [list(header)]
    for r in rows[1:]:
        out.append(r + [''] * (W - len(r)) if len(r) < W else r[:W])
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(out)
    print(f"  migrado a esquema nuevo: {os.path.basename(path)} ({len(rows[0])}->{W} cols)")


def _agg_row(mode, sampling, tc, tu, fitness, n_rules, total_rules, all_results, n_fail):
    """Construye la fila de resultados a partir de las reps que SÍ corrieron."""
    if all_results:
        g  = lambda k: [r[k] for r in all_results]
        sg = g('stop_generation'); bf = g('best_fitness'); ba = g('best_accuracy')
        mc = g('best_mse_chance');  mu = g('best_mse_utility')
        en = g('best_entropy_norm'); ud = g('best_util_dev')
        stat = dict(
            stop_gen_mean=f"{np.mean(sg):.2f}", stop_gen_std=f"{np.std(sg):.2f}",
            stop_gen_min=min(sg), stop_gen_max=max(sg),
            fitness_mean=f"{np.mean(bf):.6f}", fitness_std=f"{np.std(bf):.6f}",
            accuracy_mean=f"{np.mean(ba):.2f}", accuracy_std=f"{np.std(ba):.2f}",
            accuracy_min=f"{min(ba):.2f}", accuracy_max=f"{max(ba):.2f}",
            mse_chance_mean=f"{np.mean(mc):.6f}", mse_chance_std=f"{np.std(mc):.6f}",
            mse_utility_mean=f"{np.mean(mu):.6f}", mse_utility_std=f"{np.std(mu):.6f}",
            entropy_norm_mean=f"{np.mean(en):.6f}", entropy_norm_std=f"{np.std(en):.6f}",
            util_dev_mean=f"{np.mean(ud):.6f}", util_dev_std=f"{np.std(ud):.6f}",
        )
    else:
        stat = {k: 'nan' for k in (
            'stop_gen_mean','stop_gen_std','stop_gen_min','stop_gen_max',
            'fitness_mean','fitness_std','accuracy_mean','accuracy_std',
            'accuracy_min','accuracy_max','mse_chance_mean','mse_chance_std',
            'mse_utility_mean','mse_utility_std','entropy_norm_mean','entropy_norm_std',
            'util_dev_mean','util_dev_std')}
    base = dict(mode=mode, sampling_mode=sampling,
                chance_temperature=tc, utility_temperature=tu,
                fitness_type=fitness, stop_mode=STOP_MODE,
                n_decision_rules=n_rules, n_decision_rules_pct=RULES_PCT,
                total_rules=total_rules)
    base.update(stat)
    return base


def run_for(optimizer):
    cfg = OPT_CONFIG[optimizer]
    tag = cfg['tag']; size_gen = cfg['size_gen']
    results_csv, curves_csv = _paths(tag)

    total_rules = gs.count_total_rules(BASE_CONFIG['rules_csv'])
    n_rules = gs.compute_n_rules(total_rules, RULES_PCT)

    migrate_to_new_schema(results_csv, gs.RESULTS_HEADER)
    migrate_to_new_schema(curves_csv, gs.CURVES_HEADER)
    gs.ensure_csv_header(results_csv, gs.RESULTS_HEADER)
    gs.ensure_csv_header(curves_csv, gs.CURVES_HEADER)
    completed = gs.get_completed_combinations(results_csv)

    grid = list(itertools.product(SAMPLING_MODES, CHANCE_TEMPERATURES, UTILITY_TEMPERATURES))
    grid = [g for g in grid if not (g[1] == 1.0 and g[2] == 1.0)]  # baseline (1,1) fuera

    print("=" * 70)
    print(f"  TEMPS EXT — {tag}  (sg={size_gen}, fitness={cfg['fitness']})")
    print("=" * 70)

    t0 = time.time()
    for fitness in cfg['fitness']:
        for sampling, tc, tu in grid:
            key = ('both', sampling, f"{tc}", f"{tu}", fitness, STOP_MODE, str(n_rules))
            if key in completed:
                continue
            experiments, all_results, n_fail = [], [], 0
            for rep in range(N_REPS):
                config = {
                    **BASE_CONFIG, 'optimizer_type': optimizer,
                    'n_decision_rules': n_rules, 'fitness_type': fitness,
                    'stop_mode': STOP_MODE, 'chance_temperature': tc,
                    'utility_temperature': tu, 'mode': 'both',
                    'symmetric_sampling': (sampling == 'symmetric'),
                    'random_seed': BASE_SEED + rep,
                }
                try:
                    exp, results = gs.run_single_experiment(config, size_gen, MAX_ITER, TARGET_FITNESS)
                    experiments.append(exp); all_results.append(results)
                except Exception as e:
                    n_fail += 1
                    print(f"    [{fitness} Tc={tc} Tu={tu} {sampling}] rep{rep} FALLO: {type(e).__name__}")
            status = f"acc={np.mean([r['best_accuracy'] for r in all_results]):.0f}%" if all_results else "TODAS FALLARON"
            print(f"[{tag}/{fitness}] Tc={tc} Tu={tu} {sampling}: {status} (fallos={n_fail})")

            gs.append_results_row(results_csv,
                _agg_row('both', sampling, tc, tu, fitness, n_rules, total_rules, all_results, n_fail))

            if experiments:
                curve_rows = [{
                    'mode': 'both', 'sampling_mode': sampling,
                    'chance_temperature': tc, 'utility_temperature': tu,
                    'fitness_type': fitness, 'stop_mode': STOP_MODE,
                    'n_decision_rules': n_rules, 'n_decision_rules_pct': RULES_PCT,
                    'generation': p['generation'],
                    'mean_fitness': f"{p['mean_fitness']:.6f}",
                    'mean_accuracy': f"{p['mean_accuracy']:.2f}",
                    'mean_error_chance':  f"{p['mean_error_chance']:.6f}",
                    'mean_error_utility': f"{p['mean_error_utility']:.6f}",
                    'mean_entropy_norm':  f"{p['mean_entropy_norm']:.6f}",
                    'mean_util_dev':      f"{p['mean_util_dev']:.6f}",
                } for p in gs.average_histories(experiments)]
                gs.append_curves_rows(curves_csv, curve_rows)

    print(f"=== {tag} DONE en {(time.time()-t0)/60:.1f} min ===\n")


def main():
    for opt in ['umda', 'emna', 'keda']:
        run_for(opt)
    print("=== TODO LISTO: Tc/Tu=10 añadido (UMDA/EMNA) + KEDA temps creado ===")


if __name__ == '__main__':
    main()
