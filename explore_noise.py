"""Grid de ROBUSTEZ AL RUIDO: reglas de entrenamiento parcialmente erroneas.

Pregunta: en la vida real un experto da reglas con errores. Si una fraccion de
las reglas observadas prescribe la accion EQUIVOCADA, ¿recupera la EDA la
politica VERDADERA igualmente, o se rompe?

Diseno:
  - Red: bypass2 (mode='both'), entrenando con TODAS las reglas (n=-1).
  - Se corrompe una fraccion `noise` de las reglas de entrenamiento: su accion
    objetivo de fitness se cambia por una erronea aleatoria (rule_noise en
    id_recovery). La accion VERDADERA se conserva para medir accuracy.
  - Barremos noise = 0, 10, 20, 30, 40, 50 %.
  - Metrica clave: accuracy sobre las 20 reglas REALES (no las corrompidas).
    Si baja suavemente -> robusto; si colapsa -> fragil.

Grid: umda/emna/keda x regret/binary x 6 niveles de ruido x 10 reps = 360 runs.
Salida: example/explore_noise.csv
"""
import os
import time
import numpy as np
import pandas as pd

from id_recovery import IDRecovery


NET = {
    'name': 'bypass2',
    'xdsl_path': r'example\bypass2\network-bypass2.xdsl',
    'rules_csv': r'example\bypass2\reglas_generadas.csv',
    'mode': 'both',
}

OPTIMIZERS    = ['umda', 'emna', 'keda']
FITNESS_TYPES = ['regret', 'binary']
NOISE_LEVELS  = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'keda': 400}

COMMON = {
    'u_range':            (0, 10),
    'min_max_ut':         True,
    'alpha':              0.5,
    'elite_factor':       0.0,
    'stop_mode':          'top50',
    'symmetric_sampling': False,
    'chance_temperature': 1.0,
    'utility_temperature': 1.0,
    'n_decision_rules':   -1,      # entrenar con TODAS las reglas
}

MAX_ITER       = 60
TARGET_FITNESS = 1e-5
N_REPS         = 10
BASE_SEED      = 42

RAW_CSV = r'example\explore_noise.csv'


def run_one(optimizer, fitness_type, noise, rep, seed):
    sg = SIZE_GEN_PER_OPT[optimizer]
    params = dict(COMMON)
    params.update({
        'xdsl_path': NET['xdsl_path'],
        'rules_csv': NET['rules_csv'],
        'mode':      NET['mode'],
        'fitness_type':   fitness_type,
        'optimizer_type': optimizer,
        'random_seed':    seed,
        'rule_noise':     noise,
    })
    exp = IDRecovery(**params)
    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {optimizer}/{fitness_type}/noise={noise} rep={rep} "
              f"{type(e).__name__}")
        return None
    if not exp.history:
        return None
    last = exp.history[-1]
    gen_cpus = [float(h.get('gen_cpu_time', float('nan'))) for h in exp.history]
    return {
        'net':            NET['name'],
        'optimizer':      optimizer,
        'fitness_type':   fitness_type,
        'noise':          noise,
        'rep':            rep,
        'seed':           seed,
        'size_gen':       sg,
        'n_noisy_rules':  int(exp.n_noisy_rules),
        'total_rules':    len(exp.all_rules),
        'stop_generation': last['gen'],
        # accuracy sobre las reglas REALES (a_idx), no las corrompidas:
        'best_accuracy':  float(np.max(last['accuracies'])),
        'mean_accuracy':  float(np.mean(last['accuracies'])),
        'mse_chance':     float(np.min(last['errors_chance'])),
        'mse_utility':    float(np.min(last['errors_utility'])),
        'cpu_total':      float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df['optimizer'], df['fitness_type'], df['noise'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(o, f, n, r) for o in OPTIMIZERS
                            for f in FITNESS_TYPES
                            for n in NOISE_LEVELS
                            for r in range(N_REPS)]
    total = len(combos)
    todo  = total - len(done)
    print(f"Plan ruido: {total} runs ({todo} pendientes, {len(done)} hechos)")

    t0, completed = time.time(), 0
    for opt, fit, noise, rep in combos:
        if (opt, fit, noise, rep) in done:
            continue
        seed = BASE_SEED + rep
        r = run_one(opt, fit, noise, rep, seed)
        completed += 1
        if r is not None:
            rows.append(r)
            acc = r['best_accuracy']; nn = r['n_noisy_rules']
        else:
            acc, nn = float('nan'), 0
        elapsed = time.time() - t0
        eta = elapsed / completed * (todo - completed) if completed else 0
        print(f"[{completed}/{todo}] {opt}/{fit}/noise={noise:.0%} rep={rep}: "
              f"acc_real={acc:.0f}% ({nn} reglas ruidosas)  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed} runs en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    print("\n=== Accuracy REAL media por (optimizer, fitness, noise) ===")
    print(df.pivot_table(values='best_accuracy',
                         index=['optimizer', 'fitness_type'],
                         columns='noise', aggfunc='mean').round(1).to_string())


if __name__ == '__main__':
    main()
