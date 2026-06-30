"""Mantiene el equipo despierto mientras corren experimentos largos.

Llama a SetThreadExecutionState con ES_CONTINUOUS|ES_SYSTEM_REQUIRED, que le
indica a Windows que NO entre en suspension por inactividad mientras este
proceso siga vivo (independientemente de la config de energia). Re-afirma el
estado periodicamente por seguridad. Al morir, restaura el estado normal.

Lanzar en segundo plano y matarlo cuando terminen los experimentos.
"""
import ctypes
import time

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

kernel32 = ctypes.windll.kernel32

print("keep_awake: activado (el equipo no se suspendera por inactividad)")
try:
    while True:
        kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        time.sleep(30)
finally:
    kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("keep_awake: desactivado")
