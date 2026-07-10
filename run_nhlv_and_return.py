"""Segundo ordenador: corre el experimento de NHLv y, al terminar, DEVUELVE los
resultados por git (commit + pull --rebase + push) para que la máquina principal
los recoja con un simple `git pull`.

Uso (en el segundo equipo, tras el setup del RUNBOOK):
    python run_nhlv_and_return.py

Es reanudable: si se corta, relanzarlo continúa el experimento y vuelve a intentar
la devolución al final.
"""
import subprocess
import sys

RESULTS = ['example/explore_variance_nhlv.csv',
           'example/explore_variance_nhlv_summary.csv']


def sh(*args, check=False):
    print('>>', ' '.join(args), flush=True)
    return subprocess.run(args, check=check)


def main():
    # 1) Correr el experimento (bloquea hasta terminar; reanudable).
    r = sh(sys.executable, 'explore_variance_nhlv.py')
    if r.returncode != 0:
        print('!! el experimento no terminó bien; NO se devuelven resultados.', flush=True)
        sys.exit(1)

    # 2) Resumen (genera el _summary.csv).
    sh(sys.executable, 'explore_variance_nhlv.py', '--summary')

    # 3) Devolver por git. Si git no tiene identidad configurada, esto fallará:
    #    configúrala una vez con  git config user.name / user.email  (ver RUNBOOK).
    sh('git', 'add', *RESULTS)
    # commit puede no tener nada nuevo (p.ej. re-run ya devuelto): no es error fatal.
    sh('git', 'commit',
       '-m', 'nhlv variance: resultados del segundo equipo',
       '-m', 'Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')
    # traer lo que haya subido la máquina principal (tocan ficheros distintos ->
    # merge/rebase limpio) y luego empujar.
    sh('git', 'pull', '--rebase')
    p = sh('git', 'push')
    if p.returncode == 0:
        print('\n== Resultados DEVUELTOS (git push OK). En la máquina principal: git pull ==', flush=True)
    else:
        print('\n!! push falló. Revisa credenciales/red y reintenta:  git push', flush=True)


if __name__ == '__main__':
    main()
