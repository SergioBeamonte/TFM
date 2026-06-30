"""Estudio de TEMPERATURA y FITNESS (criterio de parada ya fijado a top90).

Misma maquinaria que explore_stopmode.py, pero ahora el criterio de parada está
FIJO (top90, con mínimo de 50 generaciones) y se barren:
  - temperatura softmax de las CPTs (chance_temperature)
  - temperatura sigmoide de las utilidades (utility_temperature)
  - función de fitness
sobre dos niveles de % de reglas, para mostrar que la recuperación depende más
de la temperatura / el fitness que del número de reglas.

Se guarda, por generación, la curva best/mean/worst (mejor/media/peor individuo),
sat_acc_mean y pct_satisfiers (para marcar top50/70/80/90).

Salidas:
  example/explore_temp.csv          (1 fila por run)
  example/explore_temp_curves.csv   (1 fila por (run, generación))

Lanzar:
    python explore_temp.py
"""
import os
import time

import numpy as np
import pandas as pd

from id_recovery import IDRecovery

# ─── DISEÑO DEL BARRIDO ─────────────────────────────────────────────────────────
# Orden de optimizadores: rápidos primero, KEDA (el caro) al final.
OPTIMIZERS    = ['umda', 'emna', 'egna', 'keda']
FITNESS_TYPES = ['binary', 'margin', 'softmax', 'regret', 'entropy']
# Temperaturas en escala geométrica (factor 2), simétricas en log en torno a 1.
CHANCE_TEMPS  = [0.25, 0.5, 1.0, 2.0, 4.0]
UTILITY_TEMPS = [0.25, 0.5, 1.0, 2.0, 4.0]
RULE_PCTS     = [10, 20]
STOP_MODE     = 'top90'
MIN_ITER      = 50
MAX_ITER      = 100
# KEDA necesita población > nº de variables para que gaussian_kde no sea singular
# (bypass mode=both ~66 libres); el resto van bien con 50.
SIZE_GEN_PER_OPTIMIZER = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}
TARGET_FITNESS = 1e-5
BASE_SEED     = 42
N_REPS        = 5

NETWORKS = {
    'bypass': {
        'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
        'rules_csv': r'example\bypass2\reglas_generadas.csv',
        'mode':      'both',
    },
}

COMMON = dict(
    min_max_ut=True, u_range=(0, 10), alpha=0.5, elite_factor=0.0,
    symmetric_sampling=False,
)

RAW_CSV    = r'example\explore_temp.csv'
CURVES_CSV = r'example\explore_temp_curves.csv'


def total_rules_of(net_cfg):
    probe = IDRecovery(xdsl_path=net_cfg['xdsl_path'], rules_csv=net_cfg['rules_csv'],
                       mode=net_cfg['mode'], n_decision_rules=-1,
                       fitness_type='binary', optimizer_type='umda', **COMMON)
    return len(probe.all_rules)


def n_rules_for_pct(total, pct):
    return max(1, int(round(pct / 100.0 * total)))


def run_one(net, cfg, opt, fit, tc, tu, pct, n_rules, total, rep, seed):
    exp = IDRecovery(
        xdsl_path=cfg['xdsl_path'], rules_csv=cfg['rules_csv'], mode=cfg['mode'],
        n_decision_rules=n_rules, stop_mode=STOP_MODE, optimizer_type=opt,
        fitness_type=fit, chance_temperature=tc, utility_temperature=tu,
        random_seed=seed, **COMMON,
    )
    try:
        exp.run(g=SIZE_GEN_PER_OPTIMIZER.get(opt, 50), i=MAX_ITER,
                target_fitness=TARGET_FITNESS, min_iter=MIN_ITER)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {net} {opt} {fit} Tc={tc} Tu={tu} {pct}% rep={rep} aborted: {type(e).__name__}")
        return None, None
    if not exp.history:
        return None, None

    last = exp.history[-1]
    pop_fit = np.asarray(last['fitness'], dtype=float)
    pop_acc = np.asarray(last['accuracies'], dtype=float)
    gen_cpus = [float(h.get('gen_cpu_time', np.nan)) for h in exp.history]
    sat_acc = pop_acc[pop_fit <= 1e-9]

    ident = dict(network=net, optimizer=opt, fitness_type=fit,
                 chance_temperature=tc, utility_temperature=tu,
                 pct=pct, n_decision_rules=n_rules, total_rules=total, rep=rep, seed=seed)
    row = {**ident,
           'acc_best': float(np.max(pop_acc)), 'acc_mean': float(np.mean(pop_acc)),
           'acc_worst': float(np.min(pop_acc)),
           'sat_acc_mean': float(np.mean(sat_acc)) if sat_acc.size else float('nan'),
           'n_satisfiers': int(sat_acc.size),
           'stop_generation': int(last['gen']),
           'cpu_total': float(np.nansum(gen_cpus)) if gen_cpus else float('nan')}

    curve_rows = []
    for h in exp.history:
        acc_g = np.asarray(h['accuracies'], dtype=float)
        fit_g = np.asarray(h['fitness'], dtype=float)
        sat_g = acc_g[fit_g <= 1e-9]
        pop = int(acc_g.size)
        curve_rows.append({**ident, 'generation': int(h['gen']),
                           'acc_best': float(np.max(acc_g)) if pop else float('nan'),
                           'acc_mean': float(np.mean(acc_g)) if pop else float('nan'),
                           'acc_worst': float(np.min(acc_g)) if pop else float('nan'),
                           'sat_acc_mean': float(np.mean(sat_g)) if sat_g.size else float('nan'),
                           'n_satisfiers': int(sat_g.size), 'pop_size': pop,
                           'pct_satisfiers': float(100.0 * sat_g.size / pop) if pop else float('nan')})
    return row, curve_rows


def load_existing():
    """Reanudable y robusto: si algún CSV está ilegible (p.ej. corrupto), se
    empieza de cero (mejor que arrastrar datos rotos)."""
    try:
        rows = pd.read_csv(RAW_CSV).to_dict('records') if os.path.exists(RAW_CSV) else []
        curve_rows = pd.read_csv(CURVES_CSV).to_dict('records') if os.path.exists(CURVES_CSV) else []
    except Exception as e:
        print(f"  !! CSV ilegible ({e}); se empieza de cero.")
        return [], [], set()
    done = set((r['network'], r['optimizer'], r['fitness_type'],
                r['chance_temperature'], r['utility_temperature'], r['pct'], r['rep'])
               for r in rows)
    return rows, curve_rows, done


def _atomic_write(path, df):
    """Escritura atómica: vuelca a .tmp y renombra (os.replace es atómico). Evita
    ficheros a medio escribir y la corrupción si el proceso muere a mitad."""
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def persist(rows, curve_rows):
    _atomic_write(RAW_CSV, pd.DataFrame(rows))
    _atomic_write(CURVES_CSV, pd.DataFrame(curve_rows))


def main():
    rows, curve_rows, done = load_existing()
    totals = {net: total_rules_of(cfg) for net, cfg in NETWORKS.items()}
    print('Reglas totales por red:', totals)

    plan = []
    for net, cfg in NETWORKS.items():
        for pct in RULE_PCTS:                 # primero todo el 10%, luego el 20%
            for opt in OPTIMIZERS:
                for fit in FITNESS_TYPES:
                    for tc in CHANCE_TEMPS:
                        for tu in UTILITY_TEMPS:
                            for rep in range(N_REPS):
                                plan.append((net, cfg, opt, fit, tc, tu, pct, rep))
    todo = [p for p in plan if (p[0], p[2], p[3], p[4], p[5], p[6], p[7]) not in done]
    print(f"Plan total: {len(plan)} runs · hechos: {len(done)} · pendientes: {len(todo)}")

    t0 = time.time()
    for k, (net, cfg, opt, fit, tc, tu, pct, rep) in enumerate(todo, 1):
        total = totals[net]
        n_rules = n_rules_for_pct(total, pct)
        row, crows = run_one(net, cfg, opt, fit, tc, tu, pct, n_rules, total, rep, BASE_SEED + rep)
        if row is not None:
            rows.append(row); curve_rows.extend(crows)
            msg = (f"acc[w/m/b]={row['acc_worst']:.0f}/{row['acc_mean']:.0f}/{row['acc_best']:.0f} "
                   f"gen={row['stop_generation']}")
        else:
            msg = 'ABORTED'
        elapsed = time.time() - t0
        eta = elapsed / k * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {net} {opt} {fit} Tc={tc} Tu={tu} {pct}% rep={rep}: {msg}  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        persist(rows, curve_rows)

    print(f"\n=== DONE en {(time.time()-t0)/60:.1f} min ===")


if __name__ == '__main__':
    main()
