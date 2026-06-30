"""Vigilante: espera a que TODO UMDA termine (nhlv1 umda = 100 runs) y entonces
para el experimento (para que NO arranque EMNA-nhlv1) y avisa. Resumible: los
CSV quedan intactos, así que EMNA se puede lanzar luego sin repetir UMDA."""
import time
import subprocess

import pandas as pd

CSV = 'example/explore_stopmode.csv'
TARGET = 20  # nhlv1 umda con top90-only: 4 pct x 1 stop x 5 rep

KILL = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object {$_.CommandLine -like '*explore_stopmode*'} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }")

while True:
    try:
        d = pd.read_csv(CSV)
        n = int(len(d[(d.network == 'nhlv1') & (d.optimizer == 'umda')
                      & (d.stop_mode == 'top90')]))
    except Exception:
        n = -1
    if n >= TARGET:
        subprocess.run(['powershell', '-NoProfile', '-Command', KILL])
        print(f'UMDA_DONE: nhlv1 umda = {n}. Experimento detenido antes de EMNA.')
        break
    time.sleep(150)
