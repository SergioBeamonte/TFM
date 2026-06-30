"""Grid de ROBUSTEZ AL RUIDO con SUPERVISION PARCIAL.

Variante del grid de ruido: en vez de entrenar con TODAS las reglas, se entrena
con un SUBCONJUNTO (un porcentaje) y, dentro de el, una fraccion es erronea.
Asi varia el NUMERO DE REGLAS CORRECTAS de entrenamiento, que es lo que de
verdad determina si se recupera la politica real.

Cruzamos:
  - tamano del subconjunto: 20/40/60 % de las 20 reglas  -> 4/8/12 reglas
  - nivel de ruido:         0/25/50 %  (fraccion de ese subconjunto que es erronea)
  => numero de CORRECTAS = n_train - n_ruidosas  (se registra explicito)
  - los 4 EDAs: umda, emna, egna, keda
  - fitness regret, 10 reps

Metrica clave: accuracy sobre las 20 reglas REALES (no las corrompidas).
La idea: ver si lo que manda es el numero ABSOLUTO de reglas correctas,
independientemente de cuanta supervision ruidosa las acompane.

Salida: example/explore_noise_subset.csv
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
    'total_rules': 20,
}

OPTIMIZERS    = ['umda', 'emna', 'egna', 'keda']      # los 4 EDAs
RULES_PCT     = [20, 40, 60]                          # subconjunto: 4/8/12 reglas
NOISE_LEVELS  = [0.0, 0.25, 0.50]
FITNESS       = 'regret'
SIZE_GEN_PER_OPT = {'umda': 50, 'emna': 50, 'egna': 50, 'keda': 400}

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
}

MAX_ITER       = 60
TARGET_FITNESS = 1e-5
N_REPS         = 10
BASE_SEED      = 42

RAW_CSV = r'example\explore_noise_subset.csv'


def n_rules_for(pct):
    return max(1, round(NET['total_rules'] * pct / 100))


def run_one(optimizer, pct, noise, rep, seed):
    sg = SIZE_GEN_PER_OPT[optimizer]
    n_rules = n_rules_for(pct)
    params = dict(COMMON)
    params.update({
        'xdsl_path': NET['xdsl_path'],
        'rules_csv': NET['rules_csv'],
        'mode':      NET['mode'],
        'optimizer_type':   optimizer,
        'random_seed':      seed,
        'n_decision_rules': n_rules,    # SUBCONJUNTO, no todas
        'rule_noise':       noise,
    })
    exp = IDRecovery(**params)
    try:
        exp.run(g=sg, i=MAX_ITER, target_fitness=TARGET_FITNESS)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"    !! {optimizer}/pct={pct}/noise={noise} rep={rep} {type(e).__name__}")
        return None
    if not exp.history:
        return None
    last = exp.history[-1]
    n_noisy = int(exp.n_noisy_rules)
    n_correct = n_rules - n_noisy
    gen_cpus = [float(h.get('gen_cpu_time', float('nan'))) for h in exp.history]
    return {
        'net':            NET['name'],
        'optimizer':      optimizer,
        'fitness_type':   FITNESS,
        'rules_pct':      pct,
        'n_train_rules':  n_rules,
        'noise':          noise,
        'n_noisy_rules':  n_noisy,
        'n_correct_rules': n_correct,      # <-- variable clave
        'rep':            rep,
        'seed':           seed,
        'size_gen':       sg,
        'total_rules':    NET['total_rules'],
        'stop_generation': last['gen'],
        'best_accuracy':  float(np.max(last['accuracies'])),   # sobre las 20 REALES
        'mean_accuracy':  float(np.mean(last['accuracies'])),
        'mse_chance':     float(np.min(last['errors_chance'])),
        'mse_utility':    float(np.min(last['errors_utility'])),
        'cpu_total':      float(np.nansum(gen_cpus)) if gen_cpus else float('nan'),
    }


def load_existing():
    if os.path.exists(RAW_CSV):
        df = pd.read_csv(RAW_CSV)
        done = set(zip(df['optimizer'], df['rules_pct'], df['noise'], df['rep']))
        return df.to_dict('records'), done
    return [], set()


def main():
    rows, done = load_existing()
    combos = [(o, p, n, r) for o in OPTIMIZERS
                           for p in RULES_PCT
                           for n in NOISE_LEVELS
                           for r in range(N_REPS)]
    total = len(combos)
    todo  = total - len(done)
    print(f"Plan ruido-subset: {total} runs ({todo} pendientes, {len(done)} hechos)")

    t0, completed = time.time(), 0
    for opt, pct, noise, rep in combos:
        if (opt, pct, noise, rep) in done:
            continue
        seed = BASE_SEED + rep
        r = run_one(opt, pct, noise, rep, seed)
        completed += 1
        if r is not None:
            rows.append(r)
            acc = r['best_accuracy']; nc = r['n_correct_rules']; nt = r['n_train_rules']
        else:
            acc, nc, nt = float('nan'), 0, 0
        elapsed = time.time() - t0
        eta = elapsed / completed * (todo - completed) if completed else 0
        print(f"[{completed}/{todo}] {opt}/pct={pct}%/noise={noise:.0%} rep={rep}: "
              f"correctas={nc}/{nt} -> acc_real={acc:.0f}%  "
              f"({elapsed/60:.1f}min / ETA {eta/60:.1f}min)")
        pd.DataFrame(rows).to_csv(RAW_CSV, index=False)

    print(f"\n=== DONE: {completed} runs en {(time.time()-t0)/60:.1f} min ===")
    df = pd.DataFrame(rows)
    if df.empty:
        return
    print("\n=== Accuracy real media por (optimizer, nº correctas) ===")
    print(df.pivot_table(values='best_accuracy', index='optimizer',
                         columns='n_correct_rules', aggfunc='mean').round(0).to_string())


if __name__ == '__main__':
    main()
