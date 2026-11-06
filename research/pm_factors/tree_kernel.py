r"""LightGBM text models flattened for a compiled (numba) predictor that can also sum just a subset of the trees.

``ic_score`` changes one model column at a time, so a candidate only moves the trees that split on that column:
prediction = (the current prediction) - (those trees on the old column) + (those trees on the new column). ``FlatModel``
keeps every tree in flat arrays and ``trees_of[j]`` lists the trees that use feature j; ``predict(X, rows, trees)`` walks
only the given trees for the given rows, in parallel over rows (``NUMBA_NUM_THREADS``). The node logic -- missing
types, default direction, NaN as 0 unless NaN is the missing type, categorical bitsets -- is ``ic_score.TreeModel``'s,
which matches LightGBM's own predictions; the output is the raw score (the sum of leaf values).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def _predict(X, rows, trees, node_off, n_nodes, feat, thr, dt, left, right, leaf_off, leaf, cb_off, cb, ct_off, ct):  # pragma: no cover
    out = np.zeros(rows.size)
    for r in prange(rows.size):
        i = rows[r]
        s = 0.0
        for q in range(trees.size):
            t = trees[q]
            if n_nodes[t] == 0:
                s += leaf[leaf_off[t]]
                continue
            node = 0
            while True:
                k = node_off[t] + node
                x = X[i, feat[k]]
                d = dt[k]
                isnan = np.isnan(x)
                missing = (d >> 2) & 3  # 0 none, 1 zero, 2 nan
                if (d & 1) == 1:  # categorical: bit x of the node's bitset
                    ci = int(thr[k])
                    xi = -1 if isnan else int(x)
                    go_left = False
                    if xi >= 0:
                        word = cb[cb_off[t] + ci] + xi // 32
                        if word < cb[cb_off[t] + ci + 1]:
                            go_left = ((ct[ct_off[t] + word] >> (xi % 32)) & 1) == 1
                else:
                    xz = 0.0 if (isnan and missing != 2) else x
                    if (missing == 2 and isnan) or (missing == 1 and abs(xz) <= 1e-35):
                        go_left = ((d >> 1) & 1) == 1
                    else:
                        go_left = xz <= thr[k]
                nxt = left[k] if go_left else right[k]
                if nxt < 0:
                    s += leaf[leaf_off[t] + ~nxt]
                    break
                node = nxt
        out[r] = s
    return out


class FlatModel:
    """One model's trees as flat arrays; ``trees_of[j]`` = the trees that split on feature j."""

    def __init__(self, trees: list[dict]) -> None:
        n = len(trees)
        self.n_nodes = np.array([t["feat"].size for t in trees], np.int64)
        self.node_off = np.concatenate([[0], np.cumsum(self.n_nodes)[:-1]]).astype(np.int64)
        cat = lambda k, dtype: np.concatenate([t[k] for t in trees]).astype(dtype) if n else np.zeros(0, dtype)  # noqa: E731
        self.feat, self.thr, self.dt = cat("feat", np.int64), cat("thr", np.float64), cat("dt", np.int64)
        self.left, self.right = cat("left", np.int64), cat("right", np.int64)
        nl = np.array([t["leaf"].size for t in trees], np.int64)
        self.leaf_off = np.concatenate([[0], np.cumsum(nl)[:-1]]).astype(np.int64)
        self.leaf = cat("leaf", np.float64)
        ncb = np.array([t["cat_b"].size for t in trees], np.int64)
        nct = np.array([t["cat_t"].size for t in trees], np.int64)
        self.cb_off = np.concatenate([[0], np.cumsum(ncb)[:-1]]).astype(np.int64)
        self.ct_off = np.concatenate([[0], np.cumsum(nct)[:-1]]).astype(np.int64)
        self.cb = cat("cat_b", np.int64) if ncb.sum() else np.zeros(1, np.int64)
        self.ct = cat("cat_t", np.int64) if nct.sum() else np.zeros(1, np.int64)
        self.all = np.arange(n, dtype=np.int64)
        uses: dict[int, list[int]] = {}
        for i, t in enumerate(trees):
            for f in np.unique(t["feat"]):
                uses.setdefault(int(f), []).append(i)
        self.trees_of = {f: np.array(v, np.int64) for f, v in uses.items()}

    @classmethod
    def from_file(cls, path: Path) -> FlatModel:
        """Parse a LightGBM text model with ``ic_score.TreeModel``'s reader."""
        from ic_score import TreeModel  # the same parser the pure-numpy predictor uses

        return cls(TreeModel(path).trees)

    def predict(self, X: np.ndarray, rows: np.ndarray | None = None, trees: np.ndarray | None = None) -> np.ndarray:
        """Raw score of ``rows`` of X (all rows by default) summed over ``trees`` (all by default)."""
        rows = np.arange(X.shape[0], dtype=np.int64) if rows is None else rows.astype(np.int64)
        trees = self.all if trees is None else trees
        if trees.size == 0:
            return np.zeros(rows.size)
        return _predict(X, rows, trees, self.node_off, self.n_nodes, self.feat, self.thr, self.dt, self.left, self.right,
                        self.leaf_off, self.leaf, self.cb_off, self.cb, self.ct_off, self.ct)
