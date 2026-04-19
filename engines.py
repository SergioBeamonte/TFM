import itertools
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pysmile

from models import IDNode, NodeKind, DecisionPolicy, EvaluationResult
from math_utils import _expand, _sum_out, _max_out

# ══════════════════════════════════════════════════════════════════════════════
# 4  BASE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class BaseEngine(ABC):

    @abstractmethod
    def evaluate(self,
                 nodes   : Dict[str, IDNode],
                 evidence: Dict[str, str],
                 net     : pysmile.Network) -> EvaluationResult:
        ...

    # ── shared utilities ─────────────────────────────────────────────────────

    @staticmethod
    def _posteriors(net     : pysmile.Network,
                    evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Standard belief propagation (delegated to pysmile)."""
        net.clear_all_evidence()
        for nid, state in evidence.items():
            net.set_evidence(nid, state)
        net.update_beliefs()
        out: Dict[str, Dict[str, float]] = {}
        for h in net.get_all_nodes():
            if net.get_node_type(h) != pysmile.NodeType.CPT:
                continue
            name  = net.get_node_id(h)
            vals  = net.get_node_value(h)
            n_out = net.get_outcome_count(h)
            out[name] = {net.get_outcome_id(h, i): float(vals[i])
                         for i in range(n_out)}
        return out

    @staticmethod
    def _topo_sort(nodes: Dict[str, IDNode]) -> List[str]:
        """Kahn's algorithm – topological ordering respecting parent→child."""
        in_deg: Dict[str, int] = {n: 0 for n in nodes}
        for nd in nodes.values():
            for p in nd.parents:
                if p in nodes:
                    in_deg[nd.name] += 1
        q     = deque(n for n, d in in_deg.items() if d == 0)
        order : List[str] = []
        while q:
            n = q.popleft()
            order.append(n)
            for s in nodes[n].successors(nodes):
                if s in in_deg:
                    in_deg[s] -= 1
                    if in_deg[s] == 0:
                        q.append(s)
        return order


# ══════════════════════════════════════════════════════════════════════════════
# 5  SHACHTER ENGINE  (exact arc-reversal)
# ══════════════════════════════════════════════════════════════════════════════

class ShachterEngine(BaseEngine):
    """
    Evaluates an influence diagram exactly using Shachter's (1986) algorithm:

    1. Add memory arcs to enforce a total temporal order on decisions.
    2. Iteratively eliminate nodes by
         a. marginalising out barren chance nodes,
         b. maximising over decision nodes (and recording the policy), or
         c. reversing arcs via Bayes' theorem until (a) or (b) applies.
    3. Posterior probabilities are computed by pysmile belief propagation.

    Complexity is exponential in the tree-width of the moral graph, but
    exact in every case (no approximation).
    """

    def evaluate(self, nodes_orig, evidence, net):
        # Deep-copy so the caller's network is not modified
        nodes = {n: nd.copy() for n, nd in nodes_orig.items()}
        sizes = {n: len(nd.states)
                 for n, nd in nodes.items() if nd.kind != NodeKind.UTILITY}

        util_name = next(n for n, nd in nodes.items()
                         if nd.kind == NodeKind.UTILITY)
        util_nd   = nodes[util_name]

        # ── 1. condition on evidence via point-mass CPTs ─────────────────────
        for var, state in evidence.items():
            if var in nodes and nodes[var].kind == NodeKind.CHANCE:
                nd  = nodes[var]
                idx = nd.states.index(state)
                pm  = np.zeros_like(nd.table)
                pm[..., idx] = 1.0
                nd.table = pm

        # ── 2. add memory arcs ───────────────────────────────────────────────
        self._add_memory_arcs(nodes)

        # ── 3. initialise the working utility factor u = {d, ax} ─────────────
        #    d  : numpy array with the current utility values
        #    ax : list of variable names labelling each dimension of d
        u: Dict[str, Any] = {
            "d" : util_nd.table.copy(),
            "ax": list(util_nd.axes),
        }

        # ── 4. remove irrelevant sinks ───────────────────────────────────────
        self._elim_sinks(nodes, util_name)

        policies: List[DecisionPolicy] = []

        # ── 5. main elimination loop ─────────────────────────────────────────
        for _ in range(len(nodes_orig) ** 2):   # upper bound
            if not util_nd.parents:
                break
            if   self._try_elim_chance(nodes, util_name, util_nd, u, sizes):
                pass
            elif self._try_elim_decision(nodes, util_name, util_nd, u, sizes, policies):
                pass
            else:
                # Must reverse an arc to unlock a further elimination step
                self._reverse_arc(nodes, util_name, u, sizes)
            self._elim_sinks(nodes, util_name)

        # If u["d"] is still an array, it means some nodes couldn't be eliminated
        # typically because of missing temporal links or cycles in the logic.
        final_d = np.squeeze(u["d"])
        if final_d.ndim > 0:
            # Fallback: take the maximum value if we failed to reduce to scalar
            # (though this indicates an issue in the model's structural sequence)
            print("[!] Warning: Final utility factor is not scalar. Taking maximum value as fallback.")
            max_eu = float(np.max(final_d))
        else:
            max_eu = float(final_d)

        # ── 6. aggregate per-policy results ──────────────────────────────────
        optimal: Dict[str, str]              = {}
        eus_out: Dict[str, Dict[str, float]] = {}

        for pol in policies:
            actions = {a for _, (a, _) in pol.mapping.items()}
            # Best global action = highest EU across all contexts
            best_a = max(
                actions,
                key=lambda a: max(
                    (eu for _, (aa, eu) in pol.mapping.items() if aa == a),
                    default=float("-inf"),
                ),
            )
            optimal[pol.node] = best_a
            eus_out[pol.node] = {
                a: max(
                    (eu for _, (aa, eu) in pol.mapping.items() if aa == a),
                    default=float("nan"),
                )
                for a in actions
            }

        return EvaluationResult(
            engine             = "Shachter — Exact Arc-Reversal (1986)",
            optimal_decisions  = optimal,
            expected_utilities = eus_out,
            posterior_probs    = self._posteriors(net, evidence),
            max_utility        = max_eu,
            policies           = policies,
            metadata           = {
                "algorithm"       : "Shachter 1986",
                "nodes_in_model"  : len(nodes_orig),
                "evidence_set"    : list(evidence.keys()),
            },
        )

    # ─────────────────────────────────────────────────────────────────────────
    # private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _add_memory_arcs(nodes: Dict[str, IDNode]) -> None:
        """
        Enforce the "no-forgetting" property by ensuring that each decision
        node "remembers" all previous decisions and their respective information
        sets.  This converts a regular ID into a sequential one.
        """
        # 1. Identify all decision nodes and find their temporal order
        decisions = [n for n, nd in nodes.items() if nd.kind == NodeKind.DECISION]
        if not decisions:
            return

        # Simple reachability-based order: Di is before Dj if there's a path Di->Dj
        # In a regular ID, this represents a total order.
        def has_path(start, end):
            q, visited = [end], {end}
            while q:
                curr = q.pop(0)
                if curr == start: return True
                for p in nodes[curr].parents:
                    if p not in visited:
                        visited.add(p)
                        q.append(p)
            return False

        # Sort decisions such that if there's a path from Di to Dj, i < j.
        # This is a topological sort restricted to decision nodes.
        # We'll use a bubble-sort-like approach for simplicity given typically small N.
        ordered_ds = list(decisions)
        for i in range(len(ordered_ds)):
            for j in range(i + 1, len(ordered_ds)):
                if has_path(ordered_ds[j], ordered_ds[i]):
                    ordered_ds[i], ordered_ds[j] = ordered_ds[j], ordered_ds[i]

        # 2. Sequential propagation: Di remembers all parents of D_{i-1} + D_{i-1} itself
        for i in range(1, len(ordered_ds)):
            curr = nodes[ordered_ds[i]]
            prev = nodes[ordered_ds[i-1]]
            
            # Current decision must observe previous decision and all its inputs
            new_parents = set(curr.parents) | set(prev.parents) | {ordered_ds[i-1]}
            # Filter out itself (just in case)
            new_parents.discard(ordered_ds[i])
            curr.parents = list(new_parents)

    @staticmethod
    def _elim_sinks(nodes: Dict[str, IDNode], util_name: str) -> None:
        """
        Remove non-utility nodes with no successors (they cannot affect
        the utility value so they are irrelevant).  Repeats until stable.
        """
        changed = True
        while changed:
            changed = False
            for name in list(nodes.keys()):
                if name == util_name:
                    continue
                if not nodes[name].successors(nodes):
                    del nodes[name]
                    for nd in nodes.values():
                        if name in nd.parents:
                            nd.parents.remove(name)
                    changed = True
                    break   # restart scan after any deletion

    @staticmethod
    def _try_elim_chance(nodes, util_name, util_nd, u, sizes) -> bool:
        """
        Eliminate a chance node X whose only successor is the utility node
        by marginalising it out of the utility factor.

        u["d"] and u["ax"] are updated in place.
        """
        for name in list(nodes.keys()):
            nd = nodes[name]
            if nd.kind != NodeKind.CHANCE:
                continue
            if nd.successors(nodes) != [util_name]:
                continue
            if name not in u["ax"]:
                continue

            # Union of axes: utility ∪ CPT(X)
            cpt_ax   = nd.parents + [name]
            all_ax   = list(dict.fromkeys(u["ax"] + cpt_ax))

            # Register any new parent sizes
            for p in nd.parents:
                if p not in sizes and p in nodes:
                    sizes[p] = len(nodes[p].states)

            # Σ_x  P(X=x | pa(X)) · U(..., X=x, ...)
            product      = (_expand(u["d"],    u["ax"],  all_ax, sizes) *
                            _expand(nd.table,  cpt_ax,   all_ax, sizes))
            new_d, new_ax = _sum_out(product, all_ax, name)

            u["d"]  = new_d
            u["ax"] = new_ax

            # Update utility node's parent list
            util_nd.parents.remove(name)
            for p in nd.parents:
                if p not in util_nd.parents:
                    util_nd.parents.append(p)

            del nodes[name]
            return True

        return False

    @staticmethod
    def _try_elim_decision(nodes, util_name, util_nd, u, sizes, policies) -> bool:
        """
        Eliminate a decision node D when:
          • D is a parent of the utility node, AND
          • every other parent of the utility node is also a parent of D
            (i.e. D "sees" the full information set at the moment of choice).

        Records the optimal policy and updates u in place.
        """
        for name in list(nodes.keys()):
            nd = nodes[name]
            if nd.kind != NodeKind.DECISION:
                continue
            if name not in util_nd.parents:
                continue

            others = [p for p in util_nd.parents if p != name]
            if not all(o in nd.parents for o in others):
                continue
            if name not in u["ax"]:
                continue

            sizes[name] = len(nd.states)

            # max_d U(pa(D), d)  +  argmax for the policy
            max_d, arg_d, new_ax = _max_out(u["d"], u["ax"], name)

            # Build policy: enumerate all combinations of remaining axes
            pol       = DecisionPolicy(node=name)
            pa_axes   = new_ax   # axes after removing D
            pa_sizes  = [sizes[ax] for ax in pa_axes if ax in sizes]
            pa_axes_k = [ax for ax in pa_axes if ax in nodes]  # only real nodes

            ranges = [range(sizes[ax]) for ax in pa_axes_k]
            for combo in (itertools.product(*ranges) if ranges else [()]):
                parent_vals = {ax: nodes[ax].states[combo[i]]
                               for i, ax in enumerate(pa_axes_k)}
                # Index into argmax array (only over axes that are in sizes)
                full_combo  = tuple(
                    combo[pa_axes_k.index(ax)] if ax in pa_axes_k else 0
                    for ax in pa_axes
                )
                best_idx    = int(arg_d[full_combo]) if pa_axes else int(arg_d)
                best_eu     = float(max_d[full_combo]) if pa_axes else float(max_d)
                best_action = nd.states[best_idx]
                key         = tuple(sorted(parent_vals.items()))
                pol.mapping[key] = (best_action, best_eu)

            policies.append(pol)

            u["d"]  = max_d
            u["ax"] = new_ax

            util_nd.parents.remove(name)
            del nodes[name]
            return True

        return False

    @staticmethod
    def _reverse_arc(nodes, util_name, u, sizes) -> bool:
        """
        Find a chance node X with no decision successors and more than one
        chance successor, pick the first such successor Y, and reverse the
        arc X→Y using Bayes' theorem:

            P'(Y | pa(Y)∖X)    = Σ_x P(Y|X, pa(Y)∖X) · P(X|pa(X))
            P'(X | Y, pa(X))   = P(Y|X, pa(X)) · P(X|pa(X)) / P'(Y | ·)
        """
        for xname in list(nodes.keys()):
            xnd   = nodes[xname]
            if xnd.kind != NodeKind.CHANCE:
                continue
            succs = xnd.successors(nodes)
            if any(nodes[s].kind == NodeKind.DECISION for s in succs):
                continue
            chance_succs = [s for s in succs
                            if s != util_name and nodes[s].kind == NodeKind.CHANCE]
            if not chance_succs:
                continue

            yname = chance_succs[0]
            ynd   = nodes[yname]

            # ── parent partitions ────────────────────────────────────────────
            pa_x    = set(xnd.parents)
            pa_y_no = set(ynd.parents) - {xname}   # pa(Y) without X
            common  = pa_x & pa_y_no
            only_x  = pa_x  - common
            only_y  = pa_y_no - common

            # Register sizes for all involved nodes
            for v in list(common) + list(only_x) + list(only_y) + [xname, yname]:
                if v not in sizes and v in nodes:
                    sizes[v] = len(nodes[v].states)

            # ── joint axes: common, only_x, only_y, X, Y ────────────────────
            joint_ax = (sorted(common) + sorted(only_x) +
                        sorted(only_y) + [xname, yname])

            x_exp   = _expand(xnd.table, xnd.parents + [xname], joint_ax, sizes)
            y_exp   = _expand(ynd.table, ynd.parents + [yname], joint_ax, sizes)
            joint   = x_exp * y_exp   # P(X, Y | pa_x, pa_y\X)

            # ── new marginal P'(Y | pa_x, pa_y\X) by summing X out ─────────
            x_idx     = joint_ax.index(xname)
            marg_y    = np.sum(joint, axis=x_idx)
            marg_y_ax = [a for a in joint_ax if a != xname]

            # ── new P'(X | Y, pa(X)) via Bayes ──────────────────────────────
            marg_y_exp = _expand(marg_y, marg_y_ax, joint_ax, sizes)
            with np.errstate(invalid="ignore", divide="ignore"):
                new_x_cpt = np.where(
                    marg_y_exp > 0,
                    joint / marg_y_exp,
                    1.0 / sizes[xname],   # uniform fallback on zero marginals
                )

            new_x_parents = sorted(common) + sorted(only_y) + sorted(only_x) + [yname]
            new_x_axes    = new_x_parents + [xname]
            new_y_parents = sorted(common) + sorted(only_x) + sorted(only_y)
            new_y_axes    = new_y_parents + [yname]

            xnd.table   = _expand(new_x_cpt, joint_ax, new_x_axes, sizes)
            xnd.axes    = new_x_axes
            xnd.parents = new_x_parents

            ynd.table   = _expand(marg_y, marg_y_ax, new_y_axes, sizes)
            ynd.axes    = new_y_axes
            ynd.parents = new_y_parents

            # Utility factor may now reference X via the reversed CPT
            # (no structural change to u needed – axes are unchanged)
            return True

        return False


# ══════════════════════════════════════════════════════════════════════════════
# 6  MONTE CARLO ENGINE  (forward-sampling with greedy rollout)
# ══════════════════════════════════════════════════════════════════════════════

class MonteCarloEngine(BaseEngine):
    """
    Approximate ID evaluation via forward Monte Carlo simulation.

    For each of ``n_samples`` trajectories:
      1. Sample all unobserved chance nodes in topological order.
      2. At each decision node, perform a greedy one-step lookahead:
           for every available action, estimate the downstream EU by
           completing the trajectory with inner samples, then pick the
           action with the highest estimated EU.
      3. Read off the utility at the end of the trajectory.

    The final expected utility is the average over all trajectories.
    The recommended action for each decision node is the one chosen most
    frequently across all trajectories.

    Parameters
    ----------
    n_samples      : outer trajectories (accuracy ↑ with more samples)
    n_inner        : inner samples per action during greedy lookahead
    seed           : optional RNG seed for reproducibility
    """

    def __init__(self,
                 n_samples : int           = 10_000,
                 n_inner   : int           = 100,
                 seed      : Optional[int] = None):
        self.n_samples = n_samples
        self.n_inner   = n_inner
        self.rng       = np.random.default_rng(seed)

    # ── public ───────────────────────────────────────────────────────────────

    def evaluate(self, nodes_orig, evidence, net):
        nodes     = nodes_orig   # read-only reference
        topo      = self._topo_sort(nodes)
        util_name = next(n for n, nd in nodes.items() if nd.kind == NodeKind.UTILITY)
        util_nd   = nodes[util_name]

        # Accumulators
        action_counts: Dict[str, Dict[str, int]]   = {
            n: {s: 0 for s in nd.states}
            for n, nd in nodes.items() if nd.kind == NodeKind.DECISION
        }
        action_eu_sum: Dict[str, Dict[str, float]] = {
            n: {s: 0.0 for s in nd.states}
            for n, nd in nodes.items() if nd.kind == NodeKind.DECISION
        }
        total_utility = 0.0

        for _ in range(self.n_samples):
            sample = dict(evidence)   # start with observed variables

            for vname in topo:
                nd = nodes[vname]
                if nd.kind == NodeKind.UTILITY:
                    continue

                if nd.kind == NodeKind.CHANCE:
                    if vname in sample:
                        continue   # already fixed by evidence
                    if not all(p in sample for p in nd.parents):
                        continue   # parent not yet sampled (shouldn't happen in topo)
                    idx = self._sample_idx(nd, sample, nodes)
                    sample[vname] = nd.states[idx]

                elif nd.kind == NodeKind.DECISION:
                    best_a, best_eu = self._greedy_action(
                        nd, sample, util_nd, nodes, topo)
                    sample[vname] = best_a
                    action_counts[vname][best_a] += 1
                    action_eu_sum[vname][best_a] += best_eu

            # Record utility for this trajectory
            if all(p in sample for p in util_nd.parents):
                idx     = tuple(nodes[p].states.index(sample[p])
                                for p in util_nd.parents)
                utility = float(util_nd.table[idx])
            else:
                utility = 0.0
            total_utility += utility

        mean_eu = total_utility / self.n_samples

        # ── aggregate ─────────────────────────────────────────────────────────
        optimal: Dict[str, str]              = {}
        eus_out: Dict[str, Dict[str, float]] = {}

        for dname, counts in action_counts.items():
            best_a = max(counts, key=counts.get)
            optimal[dname] = best_a
            eus_out[dname] = {
                a: (action_eu_sum[dname][a] / max(1, counts[a]))
                for a in counts
            }

        return EvaluationResult(
            engine             = "Monte Carlo — Forward Sampling (Approximate)",
            optimal_decisions  = optimal,
            expected_utilities = eus_out,
            posterior_probs    = self._posteriors(net, evidence),
            max_utility        = mean_eu,
            metadata           = {
                "n_samples" : self.n_samples,
                "n_inner"   : self.n_inner,
                "note"      : "Increase n_samples / n_inner for better accuracy.",
            },
        )

    # ── private helpers ───────────────────────────────────────────────────────

    def _sample_idx(self, nd: IDNode,
                    sample: Dict[str, str],
                    nodes : Dict[str, IDNode]) -> int:
        """Sample one state index from ``nd``'s CPT given current ``sample``."""
        if not nd.parents:
            probs = nd.table  # marginal; shape (n_states,)
        else:
            idx   = tuple(nodes[p].states.index(sample[p]) for p in nd.parents)
            probs = nd.table[idx]
        return int(self.rng.choice(len(nd.states), p=probs))

    def _get_utility(self, util_nd: IDNode,
                     sample: Dict[str, str],
                     nodes : Dict[str, IDNode]) -> Optional[float]:
        """Return the utility value for a complete sample, or None if missing."""
        if not all(p in sample for p in util_nd.parents):
            return None
        idx = tuple(nodes[p].states.index(sample[p]) for p in util_nd.parents)
        return float(util_nd.table[idx])

    def _greedy_action(self,
                       dec_nd  : IDNode,
                       sample  : Dict[str, str],
                       util_nd : IDNode,
                       nodes   : Dict[str, IDNode],
                       topo    : List[str]) -> Tuple[str, float]:
        """
        For each available action of ``dec_nd``:
          • If the utility can be read off directly (all utility parents
            are already observed), compute it exactly.
          • Otherwise, run ``n_inner`` inner forward samples to estimate EU.
        Return the action with the highest estimated EU.
        """
        best_a, best_eu = None, float("-inf")

        for action in dec_nd.states:
            s2 = {**sample, dec_nd.name: action}

            u = self._get_utility(util_nd, s2, nodes)
            if u is not None:
                eu = u
            else:
                eu = self._estimate_eu_inner(s2, util_nd, nodes, topo)

            if eu > best_eu:
                best_eu, best_a = eu, action

        return best_a, best_eu

    def _estimate_eu_inner(self,
                           base_sample: Dict[str, str],
                           util_nd    : IDNode,
                           nodes      : Dict[str, IDNode],
                           topo       : List[str]) -> float:
        """
        Complete ``base_sample`` by sampling unobserved chance nodes
        ``n_inner`` times; return the mean utility.
        """
        total = 0.0
        for _ in range(self.n_inner):
            s = dict(base_sample)
            for vname in topo:
                nd = nodes[vname]
                if nd.kind == NodeKind.CHANCE and vname not in s:
                    if all(p in s for p in nd.parents):
                        s[vname] = nd.states[self._sample_idx(nd, s, nodes)]
            u = self._get_utility(util_nd, s, nodes)
            total += u if u is not None else 0.0
        return total / self.n_inner