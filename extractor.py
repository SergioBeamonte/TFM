from typing import Dict
import numpy as np
import pysmile

from models import IDNode, NodeKind

# ══════════════════════════════════════════════════════════════════════════════
# 2  NETWORK EXTRACTOR  (pysmile.Network → {name: IDNode})
# ══════════════════════════════════════════════════════════════════════════════

class NetworkExtractor:
    """Convert a loaded pysmile.Network into an ``{name: IDNode}`` mapping."""

    @staticmethod
    def extract(net: pysmile.Network) -> Dict[str, IDNode]:
        nodes: Dict[str, IDNode] = {}

        # ── pass 1 : skeleton (no tables yet) ───────────────────────────────
        for h in net.get_all_nodes():
            name   = net.get_node_id(h)
            ntype  = net.get_node_type(h)
            n_out  = net.get_outcome_count(h)
            states = [net.get_outcome_id(h, i) for i in range(n_out)]
            pnames = [net.get_node_id(ph) for ph in net.get_parents(h)]

            if   ntype == pysmile.NodeType.CPT:      kind = NodeKind.CHANCE
            elif ntype == pysmile.NodeType.DECISION:  kind = NodeKind.DECISION
            elif ntype == pysmile.NodeType.UTILITY:
                kind, states = NodeKind.UTILITY, []
            else:
                continue   # unknown node type – skip

            nodes[name] = IDNode(name=name, kind=kind, states=states, parents=pnames)

        # ── pass 2 : fill tables ─────────────────────────────────────────────
        for h in net.get_all_nodes():
            name = net.get_node_id(h)
            nd   = nodes[name]
            if nd.kind == NodeKind.DECISION:
                continue   # decision nodes carry no probability table

            raw    = np.array(list(net.get_node_definition(h)), dtype=float)
            psizes = [len(nodes[p].states) for p in nd.parents]

            if nd.kind == NodeKind.CHANCE:
                # shape: (*parent_dims, n_own_states)
                nd.table = raw.reshape(psizes + [len(nd.states)])
                nd.axes  = nd.parents + [name]
            else:  # UTILITY
                # shape: (*parent_dims,)
                nd.table = raw.reshape(psizes if psizes else [1])
                nd.axes  = list(nd.parents)

        return nodes