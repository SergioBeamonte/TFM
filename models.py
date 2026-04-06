from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1  DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════

class NodeKind(Enum):
    CHANCE   = "CHANCE"
    DECISION = "DECISION"
    UTILITY  = "UTILITY"


@dataclass
class IDNode:
    """
    Internal representation of one influence-diagram node.

    Table layout
    ------------
    CHANCE   ndarray shape (*parent_dims, n_own_states)
             axes list  = node.parents + [node.name]
    UTILITY  ndarray shape (*parent_dims,)
             axes list  = list(node.parents)
    DECISION table = axes = None
    """
    name    : str
    kind    : NodeKind
    states  : List[str]           # empty for UTILITY
    parents : List[str]
    table   : Optional[np.ndarray] = None
    axes    : Optional[List[str]]  = None   # one label per array dimension

    def successors(self, nodes: Dict[str, 'IDNode']) -> List[str]:
        return [n for n, nd in nodes.items() if self.name in nd.parents]

    def copy(self) -> 'IDNode':
        return IDNode(
            name    = self.name,
            kind    = self.kind,
            states  = list(self.states),
            parents = list(self.parents),
            table   = self.table.copy() if self.table is not None else None,
            axes    = list(self.axes)   if self.axes  is not None else None,
        )


@dataclass
class DecisionPolicy:
    """
    Optimal action for every context (parent-value assignment) of one
    decision node, extracted by the Shachter engine.
    """
    node    : str
    # key   = tuple(sorted( (parent_name, state) pairs ))
    # value = (optimal_action, expected_utility_at_that_context)
    mapping : Dict[Tuple, Tuple[str, float]] = field(default_factory=dict)

    def best(self, parent_vals: Dict[str, str]) -> Tuple[str, float]:
        """Return (best_action, EU) for a given parent assignment."""
        key = tuple(sorted(parent_vals.items()))
        return self.mapping.get(key, (None, float("nan")))


@dataclass
class EvaluationResult:
    engine             : str
    optimal_decisions  : Dict[str, str]
    expected_utilities : Dict[str, Dict[str, float]]   # node → {action: EU}
    posterior_probs    : Dict[str, Dict[str, float]]   # chance node → {state: P}
    max_utility        : float
    policies           : List[DecisionPolicy] = field(default_factory=list)
    metadata           : Dict[str, Any]       = field(default_factory=dict)