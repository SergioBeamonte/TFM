"""
Two evaluation engines for influence diagrams (.xdsl / GeNIe-SMILE format):

    • ShachterEngine   – exact arc-reversal algorithm (Shachter 1986)
    • MonteCarloEngine – approximate forward-sampling with greedy rollout

Usage
-----
    result = run_evaluation(
        xdsl_path    = "nhl_lymphoma.xdsl",
        patient_data = {"age": "v20_29", "clinical_stage": "I"},
        engine       = "shachter",   # or "montecarlo"
        n_samples    = 20_000,       # MC only
        mc_seed      = 42,           # MC only – for reproducibility
    )
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import pysmile
import pysmile_license

from evaluation_code.models import EvaluationResult
from evaluation_code.extractor import NetworkExtractor
from evaluation_code.engines import BaseEngine, ShachterEngine, MonteCarloEngine
from evaluation_code.printer import ResultPrinter
from rule_generator import RuleGenerator


# ══════════════════════════════════════════════════════════════════════════════
# 8  MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation(
    xdsl_path   : str,
    patient_data: Dict[str, str],
    engine      : str           = "shachter",
    n_samples   : int           = 10_000,
    n_inner     : int           = 100,
    mc_seed     : Optional[int] = None,
    silent      : bool          = False,
) -> EvaluationResult:
    """
    Load an influence diagram, set patient evidence, and evaluate it.

    Parameters
    ----------
    xdsl_path    : path to the .xdsl file (GeNIe / SMILE format)
    patient_data : observed node states, e.g. ``{"age": "v20_29"}``
    engine       : ``"shachter"`` for exact arc-reversal evaluation, or
                   ``"montecarlo"`` (aliases: ``"mc"``) for approximation
    n_samples    : number of outer MC trajectories  [MC only]
    n_inner      : inner samples per action lookahead [MC only]
    mc_seed      : RNG seed for reproducibility      [MC only]
    silent       : suppress printed output

    Returns
    -------
    EvaluationResult
        Contains optimal decisions, expected utilities, posterior
        probabilities, max utility, and per-engine metadata.
    """
    # ── load ──────────────────────────────────────────────────────────────────
    if not os.path.exists(xdsl_path):
        candidate = os.path.join(os.path.dirname(__file__), xdsl_path)
        if os.path.exists(candidate):
            xdsl_path = candidate

    net = pysmile.Network()
    net.read_file(xdsl_path)

    # ── extract internal representation ──────────────────────────────────────
    nodes = NetworkExtractor.extract(net)

    # ── select engine ─────────────────────────────────────────────────────────
    key = engine.strip().lower()
    if key in ("shachter", "exact", "arc_reversal"):
        eng: BaseEngine = ShachterEngine()
    elif key in ("montecarlo", "mc", "monte_carlo", "approximate"):
        eng = MonteCarloEngine(n_samples=n_samples, n_inner=n_inner, seed=mc_seed)
    else:
        raise ValueError(
            f"Unknown engine '{engine}'.  "
            f"Choose 'shachter' (exact) or 'montecarlo' (approximate)."
        )

    # ── evaluate ──────────────────────────────────────────────────────────────
    result = eng.evaluate(nodes, patient_data, net)

    # ── print ─────────────────────────────────────────────────────────────────
    if not silent:
        ResultPrinter.print(result, patient_data)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9  OPTIONAL: SIDE-BY-SIDE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def compare_engines(
    xdsl_path   : str,
    patient_data: Dict[str, str],
    n_samples   : int           = 20_000,
    mc_seed     : Optional[int] = 0,
) -> Tuple[EvaluationResult, EvaluationResult]:
    """
    Run both engines on the same network and print a comparison table.

    Returns
    -------
    (shachter_result, mc_result)
    """
    print("\n" + "▓" * 60)
    print("  RUNNING SHACHTER (EXACT) ENGINE …")
    print("▓" * 60)
    res_s = run_evaluation(xdsl_path, patient_data, engine="shachter", silent=False)

    print("\n" + "▓" * 60)
    print(f"  RUNNING MONTE CARLO ENGINE  (n={n_samples:,}) …")
    print("▓" * 60)
    res_m = run_evaluation(xdsl_path, patient_data, engine="montecarlo",
                           n_samples=n_samples, mc_seed=mc_seed, silent=False)

    # ── comparison summary ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  COMPARISON SUMMARY")
    print("═" * 60)
    print(f"  {'Metric':<35} {'Shachter':>10}  {'MC':>10}")
    print("  " + "-" * 56)
    print(f"  {'Max Expected Utility':<35} {res_s.max_utility:>10.4f}"
          f"  {res_m.max_utility:>10.4f}")
    for dname in res_s.optimal_decisions:
        a_s = res_s.optimal_decisions.get(dname, "—")
        a_m = res_m.optimal_decisions.get(dname, "—")
        match = "✓" if a_s == a_m else "✗"
        print(f"  {'Decision: ' + dname:<35} {a_s:>10}  {a_m:>10}  {match}")
    print("═" * 60)

    return res_s, res_m


def generate_rules(
    xdsl_path: str,
    n_rules: int,
    output_path: str = "reglas_generadas.csv",
    fixed_states: Optional[Dict[str, str]] = None,
    respect_probs: bool = True
):
    """
    Genera un conjunto de reglas aleatorias y las guarda en un CSV codificado.

    Parameters
    ----------
    xdsl_path     : ruta al archivo .xdsl
    n_rules       : número total de reglas/filas a generar
    output_path   : ruta del archivo CSV de salida
    fixed_states  : diccionario de nodos con estados fijos (ej. {"PAIN": "PRESENT"})
    respect_probs : True para muestrear según CPT, False para equiprobable
    """
    gen = RuleGenerator(xdsl_path)
    gen.generate_csv(n_rules, output_path, fixed_states, respect_probs)
    
    # Generar archivo de equivalencias en el mismo directorio que el CSV
    mapping_path = os.path.splitext(output_path)[0] + "_mapping.txt"
    gen.export_mappings(mapping_path)


if __name__ == "__main__":
    """
    INSTRUCCIONES DE PRUEBA:
    1. Para evaluar un caso concreto: Usa run_evaluation()
    2. Para generar un dataset de reglas: Usa generate_rules()
    """
    test_path = "example/network-bypass2.xdsl"
    
    # Sample evidence for a patient
    sample_data = {
        "PAIN": "PRESENT",
        "HEARTDISEASE": "PRESENT",
        "ANGIOGRAM": "POSITIVE"
    }
    
    # Run exact evaluation
    run_evaluation(test_path, sample_data, engine="shachter")

    # --- NEW: Rule Generation Example ---
    # print("\n" + "#" * 60)
    # print("  GENERATING RANDOM RULES (CSV) ...")
    # print("#" * 60)
    # # generate_rules(
    #     xdsl_path=test_path,
    #     n_rules=10,
    #     output_path="reglas_ejemplo.csv",
    #     respect_probs=False  # Equiprobable for diversity in small sample
    # )