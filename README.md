# TFM — Assigning Probabilities and Utilities in Influence Diagrams using EDAs

Trabajo de Fin de Máster (MSc Data Science). Recuperación de los parámetros
(probabilidades y utilidades) de un **Diagrama de Influencia** a partir de un
conjunto de **reglas de decisión**, planteada como un problema de optimización
resuelto con **Algoritmos de Estimación de Distribuciones (EDAs)**: UMDA, EGNA,
EMNA y KEDA.

La memoria completa está en [`memoria/`](memoria/) (fuentes LaTeX y PDF compilado).

## Estructura del repositorio

```
.
├── grid_search.py          # Barrido sistemático de hiperparámetros (punto de entrada principal)
├── run_full_grid.py        # Orquestación del grid completo
├── id_recovery.py          # Núcleo: recuperación del ID mediante EDAs
├── rule_generator.py       # Generación de reglas de decisión a partir de una red
├── pysmile_license.py      # Licencia de PySMILE (motor SMILE de BayesFusion)
├── evaluation_code/        # Motor de evaluación de diagramas de influencia
├── converter/              # Conversión de modelos (ID_gen.py)
├── vendor/                 # Wheel de PySMILE vendorizado (Windows, cp313)
├── example/                # Instancias del problema (redes + reglas de entrada)
│   ├── bypass2/            #   red network-bypass2.xdsl + reglas y utilidades
│   ├── nhlv1/              #   red network-nhlv1.xdsl + reglas y utilidades
│   └── VentureTwoWay/      #   red de ejemplo clásica
├── memoria/                # Memoria del TFM (LaTeX + figuras + PDF)
├── bibliography/           # PDFs de las referencias citadas
└── .github/workflows/      # CI: ejecución del grid_search en GitHub Actions
```

Los ficheros de cada instancia en `example/` son **entradas** del método
(`network-*.xdsl`, `reglas_generadas.csv`, `rule_mappings.txt`,
`utility_tables.txt`). Los resultados (`grid_search_*.csv`) se regeneran con el
código y no se versionan de forma permanente.

## Instalación

Entorno de referencia: **Windows 64-bit + Python 3.13**.

```bash
pip install -r requirements.txt
# PySMILE no está en PyPI; se instala desde el wheel vendorizado:
pip install vendor/pysmile-2.4.0-cp313-cp313-win_amd64.whl
```

En otros sistemas operativos o versiones de Python hace falta el wheel de
PySMILE correspondiente, disponible en [BayesFusion](https://www.bayesfusion.com).

## Uso

```bash
# Barrido por defecto sobre la red del script
py grid_search.py

# Elegir optimizador y carpeta de salida
py grid_search.py --optimizer_type umda --base_folder example/nhlv1

# Especificar red y reglas explícitamente
py grid_search.py --xdsl_path example/bypass2/network-bypass2.xdsl \
                  --rules_csv example/bypass2/reglas_generadas.csv \
                  --optimizer_type keda --min_max_ut True
```

Cada ejecución produce `grid_search_results_*.csv` (estadísticas agregadas por
combinación) y `grid_search_curves_*.csv` (curvas de convergencia promediadas).

El workflow `.github/workflows/grid_search.yml` permite lanzar el mismo barrido
en GitHub Actions (`workflow_dispatch`) para cada optimizador.

## Autor

Sergio Beamonte González — MSc Data Science.
