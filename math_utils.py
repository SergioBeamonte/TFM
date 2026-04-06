from typing import Dict, List, Tuple
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 3  NAMED-AXIS FACTOR ARITHMETIC  (pure numpy helpers)
# ══════════════════════════════════════════════════════════════════════════════

def _expand(data: np.ndarray,
            cur_axes: List[str],
            tgt_axes: List[str],
            sizes   : Dict[str, int]) -> np.ndarray:
    """
    Expand ``data`` (dimensions labelled ``cur_axes``) to cover all
    ``tgt_axes`` in that order.  Missing axes are broadcast from size-1.
    Returns a *copy* with shape ``[sizes[ax] for ax in tgt_axes]``.
    """
    tmp, tmp_ax = data, list(cur_axes)
    for ax in tgt_axes:
        if ax not in tmp_ax:
            tmp = np.expand_dims(tmp, axis=len(tmp_ax))
            tmp_ax.append(ax)
    perm = [tmp_ax.index(ax) for ax in tgt_axes]
    tmp  = np.transpose(tmp, perm)
    return np.broadcast_to(tmp, [sizes[ax] for ax in tgt_axes]).copy()


def _sum_out(data: np.ndarray,
             axes: List[str],
             var : str) -> Tuple[np.ndarray, List[str]]:
    """Marginalise ``var`` out of factor by summation."""
    idx = axes.index(var)
    return np.sum(data, axis=idx), [a for a in axes if a != var]


def _max_out(data: np.ndarray,
             axes: List[str],
             var : str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Maximise over ``var``.
    Returns ``(max_values, argmax_indices, remaining_axes)``.
    """
    idx      = axes.index(var)
    new_axes = [a for a in axes if a != var]
    return np.max(data, axis=idx), np.argmax(data, axis=idx), new_axes