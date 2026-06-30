"""Mini-estudio: criterio de parada (top X%) vs calidad de los individuos que
SÍ cumplen todas las reglas de entrenamiento.

Idea
----
Con UMDA y fitness ``binary`` (penalización = nº de reglas de entrenamiento NO
satisfechas, luego penalización 0 ⇔ cumple TODAS las reglas de entrenamiento),
se barre el criterio de parada top10/30/50/70/90. En cada parada se mira la
POBLACIÓN FINAL y, **solo entre los individuos que cumplen todas las reglas de
entrenamiento**, se calcula la accuracy sobre el conjunto COMPLETO de reglas:
``mean``, ``std``, ``best`` (max) y ``worst`` (min). Así se ve la degeneración:
varios individuos cumplen las reglas que se les dieron pero difieren en las que
no han visto — alguno puede ir "muy desencaminado". También se guardan las
mismas métricas sobre TODA la población, para contraste.

Barrido
-------
  optimizador : umda, emna
  fitness     : binary
  stop_mode   : top10, top30, top50, top70, top90
  % reglas    : 5, 10, 20, 30
  redes       : bypass (mode=both, 10 reps) y nhlv1 (mode=utility_only, 5 reps)
  max_iter    : 100
  reps        : cada rep usa una muestra de reglas aleatoria y distinta (seed=42+rep)

Salidas
-------
  example/explore_stopmode.csv          (1 fila por run: info a nivel de ejecución)
  example/explore_stopmode_long.csv     (1 fila por individuo que cumple las reglas)
  example/explore_stopmode_curves.csv   (1 fila por (run, generación): curva best/mean/worst)
  example/explore_stopmode_summary.csv  (agregado por red × %reglas × stop_mode)

Lanzar:
    python explore_stopmode.py
"""
import os
import time

import numpy as np
import pandas as pd

from id_recovery import IDRecovery

# ─── DISEÑO DEL BARRIDO ─────────────────────────────────────────────────────────
OPTIMIZERS = ['umda', 'emna', 'egna', 'keda']
STOP_MODES = ['top10', 'top30', 'top50', 'top70', 'top90']
RULE_PCTS  = [5, 10, 20, 30]
MAX_ITER   = 100
SIZE_GEN   = 50
# KEDA necesita población > nº de variables (gaussian_kde); el resto va con 50.
SIZE_GEN_PER_OPTIMIZER = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}
TARGET_FITNESS = 1e-5   # binary: el target real es 0.0 (lo fija _compute_target)
BASE_SEED  = 42

# Una entrada por red. `reps` = repeticiones por experimento; cada repetición usa
# random_seed = BASE_SEED + rep, que siembra `random` y `numpy` → un subconjunto
# de reglas de entrenamiento ALEATORIO y DISTINTO (y reproducible) en cada rep.
# bypass es barato (10 reps); nhlv1 (utility_only) es caro (5 reps).
NETWORKS = {
    'bypass': {
        'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
        'rules_csv': r'example\bypass2\reglas_generadas.csv',
        'mode':      'both',
        'reps':      10,
        # EGNA/KEDA solo se añaden en bypass y con top90 (el notebook usa top90; con
        # min_iter=50 los 5 stop-modes son la misma trayectoria → redundantes).
        'optimizers': ['umda', 'emna', 'egna', 'keda'],
        'stop_modes': ['top90'],
        'min_iter':  50,                     # mínimo 50 generaciones (curvas no cortas)
    },
    'nhlv1': {
        'xdsl_path': r'example\nhlv1\network-nhlv1.xdsl',
        'rules_csv': r'example\nhlv1\reglas_generadas.csv',
        'mode':      'utility_only',
        'reps':      5,
        'optimizers': ['umda', 'emna', 'egna'],   # nhlv1: +EGNA (KEDA aparte: ~30h y degenera)
        'min_iter':  50,                     # también mínimo 50 generaciones
        # Truco: con un solo run top90 basta. El criterio de parada no cambia la
        # trayectoria del EDA (misma semilla → mismas generaciones), así que el
        # run top90 contiene los puntos de top50/top70: la columna pct_satisfiers
        # marca en qué generación se cruza cada umbral. 1 run en vez de 5.
        'stop_modes': ['top90'],
    },
}

# Hiperparámetros fijos comunes (los del resto de estudios). optimizer_type se
# inyecta en cada run (se barren UMDA y EMNA).
COMMON = dict(
    min_max_ut=True, u_range=(0, 10), alpha=0.5, elite_factor=0.0,
    fitness_type='binary',
    chance_temperature=1.0, utility_temperature=1.0,
    symmetric_sampling=False,
)

RAW_CSV     = r'example\explore_stopmode.csv'
LONG_CSV    = r'example\explore_stopmode_long.csv'
SUMMARY_CSV = r'example\explore_stopmode_summary.csv'
CURVES_CSV  = r'example\explore_stopmode_curves.csv'   # 1 fila por (run, generación)


def total_rules_of(net_cfg):
    """Nº total de reglas de la red (instancia ligera con todas las reglas)."""
    probe = IDRecovery(xdsl_path=net_cfg['xdsl_path'], rules_csv=net_cfg['rules_csv'],
                       mode=net_cfg['mode'], n_decision_rules=-1, **COMMON)
    return len(probe.all_rules)


def n_rules_for_pct(total, pct):
    return max(1, int(round(pct / 100.0 * total)))


def run_one(net, net_cfg, optimizer, pct, n_rules, total, stop_mode, rep, seed):
    exp = IDRecovery(
        xdsl_path=net_cfg['xdsl_path'], rules_csv=net_cfg['rules_csv'],
        mode=net_cfg['mode'], n_decision_rules=n_rules, stop_mode=stop_mode,
        optimizer_type=optimizer, random_seed=seed, **COMMON,
    )
    try:
        exp.run(g=SIZE_GEN_PER_OPTIMIZER.get(optimizer, SIZE_GEN), i=MAX_ITER,
                target_fitness=TARGET_FITNESS, min_iter=net_cfg.get('min_iter', 1))
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {net} {optimizer} pct={pct} {stop_mode} rep={rep} aborted: {type(e).__name__}: {e}")
        return None, None, None
    if not exp.history:
        return None, None, None

    last = exp.history[-1]
    pop_fit = np.asarray(last['fitness'], dtype=float)
    pop_acc = np.asarray(last['accuracies'], dtype=float)
    gen_cpus = [float(h.get('gen_cpu_time', np.nan)) for h in exp.history]

    # Individuos que cumplen TODAS las reglas de entrenamiento (binary: penalización 0).
    sat_mask = pop_fit <= 1e-9
    sat_acc = pop_acc[sat_mask]
    n_sat = int(sat_mask.sum())

    def _stat(a, fn):
        return float(fn(a)) if a.size else float('nan')

    row = {
        'network': net, 'optimizer': optimizer, 'mode': net_cfg['mode'], 'pct': pct,
        'n_decision_rules': n_rules, 'total_rules': total,
        'stop_mode': stop_mode, 'rep': rep, 'seed': seed,
        'pop_size': int(pop_fit.size), 'n_satisfiers': n_sat,
        'frac_satisfiers': n_sat / pop_fit.size if pop_fit.size else float('nan'),
        # Métricas SOLO de los que cumplen todas las reglas de entrenamiento:
        'sat_acc_mean':  _stat(sat_acc, np.mean),
        'sat_acc_std':   _stat(sat_acc, np.std),
        'sat_acc_best':  _stat(sat_acc, np.max),
        'sat_acc_worst': _stat(sat_acc, np.min),
        # Contraste: toda la población final.
        'pop_acc_mean':  float(np.mean(pop_acc)),
        'pop_acc_best':  float(np.max(pop_acc)),
        'pop_acc_worst': float(np.min(pop_acc)),
        'best_fitness':  float(np.min(pop_fit)),
        'stop_generation': int(last['gen']),
        'cpu_total': float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }
    long_rows = [{'network': net, 'optimizer': optimizer, 'pct': pct, 'stop_mode': stop_mode,
                  'rep': rep, 'seed': seed, 'acc': float(a)} for a in sat_acc]

    # Curva por generación: best/mean/worst de la accuracy de la población en cada
    # generación + nº de individuos que cumplen todas las reglas de entrenamiento.
    curve_rows = []
    for h in exp.history:
        acc_g = np.asarray(h['accuracies'], dtype=float)
        fit_g = np.asarray(h['fitness'], dtype=float)
        sat_g = acc_g[fit_g <= 1e-9]
        pop = int(acc_g.size)
        curve_rows.append({
            'network': net, 'optimizer': optimizer, 'pct': pct, 'stop_mode': stop_mode,
            'rep': rep, 'seed': seed, 'generation': int(h['gen']),
            'acc_best':  float(np.max(acc_g)) if pop else float('nan'),
            'acc_mean':  float(np.mean(acc_g)) if pop else float('nan'),
            'acc_worst': float(np.min(acc_g)) if pop else float('nan'),
            'sat_acc_mean': float(np.mean(sat_g)) if sat_g.size else float('nan'),
            'n_satisfiers': int(sat_g.size),
            'pop_size': pop,
            # % de individuos que cumplen TODAS las reglas de entrenamiento en esta
            # generación. Cruzar 50/70/90% ⇔ donde pararía top50/top70/top90.
            'pct_satisfiers': float(100.0 * sat_g.size / pop) if pop else float('nan'),
            'best_fitness': float(np.min(fit_g)) if pop else float('nan'),
        })
    return row, long_rows, curve_rows


def load_existing():
    try:
        rows = pd.read_csv(RAW_CSV).to_dict('records') if os.path.exists(RAW_CSV) else []
        long_rows = pd.read_csv(LONG_CSV).to_dict('records') if os.path.exists(LONG_CSV) else []
        curve_rows = pd.read_csv(CURVES_CSV).to_dict('records') if os.path.exists(CURVES_CSV) else []
    except Exception as e:
        print(f"  !! CSV ilegible ({e}); se empieza de cero.")
        return [], [], [], set()
    done = set((r['network'], r['optimizer'], r['pct'], r['stop_mode'], r['rep']) for r in rows)
    return rows, long_rows, curve_rows, done


def build_summary(rows, long_rows):
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    lng = pd.DataFrame(long_rows) if long_rows else pd.DataFrame(columns=['network', 'optimizer', 'pct', 'stop_mode', 'acc'])
    out = []
    for (net, opt, pct, sm), g in raw.groupby(['network', 'optimizer', 'pct', 'stop_mode']):
        pool = lng[(lng['network'] == net) & (lng['optimizer'] == opt)
                   & (lng['pct'] == pct) & (lng['stop_mode'] == sm)]['acc'].values
        out.append({
            'network': net, 'optimizer': opt, 'pct': pct, 'stop_mode': sm,
            'n_runs': len(g),
            'runs_with_satisfiers': int((g['n_satisfiers'] > 0).sum()),
            'mean_n_satisfiers': float(g['n_satisfiers'].mean()),
            'total_satisfiers': int(len(pool)),
            # Pooled sobre TODOS los individuos que cumplen las reglas (todas las reps):
            'sat_pool_mean':  float(np.mean(pool)) if pool.size else float('nan'),
            'sat_pool_std':   float(np.std(pool))  if pool.size else float('nan'),
            'sat_pool_best':  float(np.max(pool))  if pool.size else float('nan'),
            'sat_pool_worst': float(np.min(pool))  if pool.size else float('nan'),
            # Media entre runs de los estadísticos por run:
            'sat_runmean_mean':  float(g['sat_acc_mean'].mean()),
            'sat_runbest_mean':  float(g['sat_acc_best'].mean()),
            'sat_runworst_mean': float(g['sat_acc_worst'].mean()),
            # Contraste con toda la población:
            'pop_acc_mean':  float(g['pop_acc_mean'].mean()),
            'pop_acc_best':  float(g['pop_acc_best'].mean()),
            'pop_acc_worst': float(g['pop_acc_worst'].mean()),
            'stop_gen_mean': float(g['stop_generation'].mean()),
            'cpu_total_mean': float(g['cpu_total'].mean()),
        })
    cols_order = ['network', 'optimizer', 'pct', 'stop_mode']
    return pd.DataFrame(out).sort_values(cols_order).reset_index(drop=True)


def _atomic_write(path, df):
    tmp = path + '.tmp'
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def persist(rows, long_rows, curve_rows):
    _atomic_write(RAW_CSV, pd.DataFrame(rows))
    _atomic_write(LONG_CSV, pd.DataFrame(long_rows))
    _atomic_write(CURVES_CSV, pd.DataFrame(curve_rows))
    _atomic_write(SUMMARY_CSV, build_summary(rows, long_rows))


def main():
    rows, long_rows, curve_rows, done = load_existing()
    totals = {net: total_rules_of(cfg) for net, cfg in NETWORKS.items()}
    print("Reglas totales por red:", totals)

    plan = []
    for net, cfg in NETWORKS.items():               # bypass primero (barato)
        for opt in cfg.get('optimizers', OPTIMIZERS):
            for pct in RULE_PCTS:                     # %reglas ascendente
                for sm in cfg.get('stop_modes', STOP_MODES):
                    for rep in range(cfg['reps']):
                        plan.append((net, cfg, opt, pct, sm, rep))
    todo = [p for p in plan if (p[0], p[2], p[3], p[4], p[5]) not in done]
    print(f"Plan total: {len(plan)} runs · ya hechos: {len(done)} · pendientes: {len(todo)}")

    t0 = time.time()
    for k, (net, cfg, opt, pct, sm, rep) in enumerate(todo, 1):
        total = totals[net]
        n_rules = n_rules_for_pct(total, pct)
        seed = BASE_SEED + rep
        row, lrows, crows = run_one(net, cfg, opt, pct, n_rules, total, sm, rep, seed)
        if row is not None:
            rows.append(row)
            long_rows.extend(lrows)
            curve_rows.extend(crows)
            msg = (f"sat={row['n_satisfiers']:>2}/{row['pop_size']} "
                   f"sat_acc[min/mean/max]={row['sat_acc_worst']:.0f}/"
                   f"{row['sat_acc_mean']:.0f}/{row['sat_acc_best']:.0f} "
                   f"gen={row['stop_generation']}")
        else:
            msg = "ABORTED/empty"
        elapsed = time.time() - t0
        eta = elapsed / k * (len(todo) - k)
        print(f"[{k}/{len(todo)}] {net} {opt} {pct}% {sm} rep={rep}: {msg}  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        persist(rows, long_rows, curve_rows)   # reanudable: persiste tras cada run

    print(f"\n=== DONE en {(time.time()-t0)/60:.1f} min ===")
    summ = build_summary(rows, long_rows)
    if not summ.empty:
        with pd.option_context('display.width', 200, 'display.max_rows', 200):
            print("\n=== Resumen: accuracy de los que cumplen las reglas (pooled) ===")
            print(summ[['network', 'optimizer', 'pct', 'stop_mode', 'mean_n_satisfiers',
                        'sat_pool_worst', 'sat_pool_mean', 'sat_pool_best',
                        'pop_acc_worst', 'stop_gen_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
