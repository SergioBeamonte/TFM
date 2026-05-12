from typing import Dict
from evaluation_code.models import EvaluationResult

# ==============================================================================
# 7  RESULT PRINTER
# ==============================================================================

class ResultPrinter:

    W = 60   # column width

    @classmethod
    def print(cls, result: EvaluationResult, evidence: Dict[str, str]) -> None:
        sep = "=" * cls.W
        print(f"\n{sep}")
        print(f"  ENGINE : {result.engine}")
        print(sep)

        print("\n-- Evidence ---------------------------------------------")
        for k, v in evidence.items():
            print(f"   {k:30s} = {v}")

        if result.optimal_decisions:
            print("\n-- Decisions --------------------------------------------")
            for dname, best in result.optimal_decisions.items():
                eu_dict = result.expected_utilities.get(dname, {})
                print(f"\n   [{dname}]")
                for action, eu in sorted(eu_dict.items(), key=lambda x: -x[1]):
                    marker = "  * OPTIMAL" if action == best else ""
                    print(f"      {action:22s}  EU = {eu:10.4f}{marker}")

        if result.posterior_probs:
            print("\n-- Posterior Probabilities ------------------------------")
            for cname, probs in result.posterior_probs.items():
                print(f"\n   [{cname}]")
                for state, p in probs.items():
                    bar = "#" * int(p * 20)
                    print(f"      {state:22s}  {p:6.2%}  {bar}")

        print(f"\n-- Max Expected Utility ---------------------------------")
        print(f"   {result.max_utility:.6f}")

        if result.metadata:
            print("\n-- Metadata ---------------------------------------------")
            for k, v in result.metadata.items():
                print(f"   {k:25s}: {v}")

        print(f"\n{sep}")

        # CSV summary line
        ev_str  = " AND ".join(f"{k}={v}" for k, v in evidence.items())
        dec_str = " AND ".join(f"{k}={v}" for k, v in result.optimal_decisions.items())
        print("\n-- CSV Rule ---------------------------------------------")
        print("EVIDENCE,OPTIMAL_DECISIONS,MAX_EXPECTED_UTILITY")
        print(f"{ev_str},{dec_str},{result.max_utility:.6f}")
        print()