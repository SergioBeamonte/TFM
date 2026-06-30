"""Grid search COMPLETO sobre las dos redes.

Diseñado para argumentar las conclusiones del TFM cruzando las dimensiones que
importan:
    optimizador (4) x fitness (5) x supervision/reglas (4) x sampling (2) x red (2)

Fijos (ya estudiados en experimentos focalizados aparte): stop=top50, T=1,
min_max_ut=True, identifiable_chance=True (default), alpha=0.5, elite=0, 5 reps.
size_gen: 50 (umda/emna/egna), 400 (keda).

Reutiliza TODA la maquinaria de grid_search.py: resume por combinacion, CSVs de
resultados agregados + curvas, y medicion de CPU por generacion. Escribe en
subcarpetas grid_full/ de cada red para no tocar los grids viejos del dashboard.

Es idempotente: si muere (suspension, OOM), al relanzarlo retoma donde quedo.
"""
import os
import grid_search as gs


# ── Fijos del grid ──────────────────────────────────────────────────────────────
gs.STOP_MODES           = ['top50']
gs.CHANCE_TEMPERATURES  = [1.0]
gs.UTILITY_TEMPERATURES = [1.0]
gs.N_REPETITIONS        = 5
gs.BASE_SEED            = 42
# gs.SIZE_GEN_PER_OPTIMIZER ya trae keda=400, resto=50.

# ── Diseño POR RED ───────────────────────────────────────────────────────────────
# bypass2 (rapido): grid completo. nhlv1 (~70x mas lento): grid REDUCIDO enfocado
# en argumentar el escalado (mejores fitness, sin egna que seria inviable).
NETS = [
    {'name': 'bypass2',
     'xdsl':  r'example\bypass2\network-bypass2.xdsl',
     'rules': r'example\bypass2\reglas_generadas.csv',
     'mode':  'both',
     'fitness':  ["binary", "margin", "softmax", "regret", "entropy"],
     'rules_pct': [5, 10, 20, 40],
     'sampling': ['non_symmetric', 'symmetric'],
     'opts':     ['umda', 'emna', 'keda', 'egna'],
     'n_reps':   10},
    # nhlv1 desactivado para esta tanda (re-run solo bypass2 a 10 reps con datos
    # crudos por-rep). Sus agregados de 5 reps ya estan en nhlv1/grid_full.
    # {'name': 'nhlv1',
    #  'xdsl':  r'example\nhlv1\network-nhlv1.xdsl',
    #  'rules': r'example\nhlv1\reglas_generadas.csv',
    #  'mode':  'utility_only',
    #  'fitness':  ["regret", "binary"],
    #  'rules_pct': [5, 20, 40],
    #  'sampling': ['non_symmetric'],
    #  'opts':     ['umda', 'emna'],
    #  'n_reps':   5},
]


def main():
    for net in NETS:
        gs.MODES             = [net['mode']]
        gs.FITNESS_TYPES     = net['fitness']
        gs.RULES_PERCENTAGES = net['rules_pct']
        gs.SAMPLING_MODES    = net['sampling']
        gs.N_REPETITIONS     = net['n_reps']
        # Carpeta NUEVA (grid_raw): conserva intactos los CSVs de grid_full.
        out_folder = os.path.join(os.path.dirname(net['xdsl']), 'grid_raw')
        os.makedirs(out_folder, exist_ok=True)
        for opt in net['opts']:
            cfg = {
                'xdsl_path':     net['xdsl'],
                'rules_csv':     net['rules'],
                'min_max_ut':    True,
                'u_range':       (0, 10),
                'alpha':         0.5,
                'elite_factor':  0.0,
                'optimizer_type': opt,
            }
            print(f"\n{'#'*70}\n##### {net['name'].upper()} / {opt.upper()}  (mode={net['mode']})\n{'#'*70}")
            gs.run_grid_for_optimizer(cfg, out_folder)


if __name__ == '__main__':
    main()
