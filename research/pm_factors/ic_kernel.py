r"""Compiled (numba) per-mark kernels for ``ic_score``: the cross-sectional Spearman IC and the cross-sectional transforms,
over a panel whose rows are grouped contiguously by mark (``starts[k]:ends[k]`` = the rows of mark k), in parallel over marks.

* ``seg_spearman``: Spearman correlation per mark of two columns over the rows where both are finite (average ranks for
  ties; NaN with fewer than ``min_n`` rows or a constant first column);
* ``seg_z``: z-score per mark (population std; NaN with fewer than 2 values or zero spread);
* ``seg_rank``: average rank per mark scaled to [0, 1] (``center`` subtracts 0.5);
* ``seg_demean``: minus the mean of the row's sub-group inside its mark (``sub`` < 0 = no group; groups of fewer than
  3 values give NaN) -- sub = 0 everywhere is the mark's cross-sectional mean.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True)
def _avg_rank(a):  # pragma: no cover
    """0-based ranks, ties sharing their average (PM's fa is printed to 5 decimals: many ties)."""
    n = a.size
    o = np.argsort(a, kind="mergesort")
    out = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[o[j + 1]] == a[o[i]]:
            j += 1
        for q in range(i, j + 1):
            out[o[q]] = (i + j) / 2.0
        i = j + 1
    return out


@njit(parallel=True, cache=True)
def seg_spearman(p, r, starts, ends, min_n):  # pragma: no cover
    out = np.full(starts.size, np.nan)
    for k in prange(starts.size):
        s, e = starts[k], ends[k]
        n = 0
        for i in range(s, e):
            if np.isfinite(p[i]) and np.isfinite(r[i]):
                n += 1
        if n < min_n:
            continue
        a = np.empty(n)
        b = np.empty(n)
        j = 0
        for i in range(s, e):
            if np.isfinite(p[i]) and np.isfinite(r[i]):
                a[j] = p[i]
                b[j] = r[i]
                j += 1
        if a.std() == 0.0:
            continue
        ra = _avg_rank(a)
        rb = _avg_rank(b)
        mu = (n - 1) / 2.0
        num = 0.0
        da = 0.0
        db = 0.0
        for i in range(n):
            x = ra[i] - mu
            y = rb[i] - mu
            num += x * y
            da += x * x
            db += y * y
        if da > 0 and db > 0:
            out[k] = num / np.sqrt(da * db)
    return out


@njit(cache=True)
def ema_irregular(t, x, half_life):  # pragma: no cover
    """EMA after each update: weight 1 per update, decayed by 2^(-dt / half_life) between updates (sum / wgt, as PM's
    engine state); NaN updates are skipped."""
    out = np.empty(t.size)
    s = 0.0
    w = 0.0
    last = t[0] if t.size else 0.0
    k = np.log(2.0) / half_life
    for i in range(t.size):
        d = np.exp(-(t[i] - last) * k) if t[i] > last else 1.0
        s *= d
        w *= d
        last = t[i]
        if np.isfinite(x[i]):
            s += x[i]
            w += 1.0
        out[i] = s / w if w > 0 else np.nan
    return out


@njit(cache=True)
def ema_seeded(t, x, half_life, x0, w0, t0):  # pragma: no cover
    """ema_irregular started from a prior state: value x0 with weight w0 at time t0 (yesterday's state carried over)."""
    out = np.empty(t.size)
    s = x0 * w0
    w = w0
    last = t0
    k = np.log(2.0) / half_life
    for i in range(t.size):
        d = np.exp(-(t[i] - last) * k) if t[i] > last else 1.0
        s *= d
        w *= d
        last = t[i]
        if np.isfinite(x[i]):
            s += x[i]
            w += 1.0
        out[i] = s / w if w > 0 else np.nan
    return out


@njit(parallel=True, cache=True)
def seg_z(v, starts, ends):  # pragma: no cover
    out = np.full(v.size, np.nan)
    for k in prange(starts.size):
        s, e = starts[k], ends[k]
        n = 0
        sm = 0.0
        for i in range(s, e):
            if np.isfinite(v[i]):
                n += 1
                sm += v[i]
        if n < 2:
            continue
        mu = sm / n
        ss = 0.0
        for i in range(s, e):
            if np.isfinite(v[i]):
                ss += (v[i] - mu) ** 2
        sd = np.sqrt(ss / n)
        if sd > 0:
            for i in range(s, e):
                if np.isfinite(v[i]):
                    out[i] = (v[i] - mu) / sd
    return out


@njit(parallel=True, cache=True)
def seg_rank(v, starts, ends, center):  # pragma: no cover
    out = np.full(v.size, np.nan)
    for k in prange(starts.size):
        s, e = starts[k], ends[k]
        n = 0
        for i in range(s, e):
            if np.isfinite(v[i]):
                n += 1
        if n < 2:
            continue
        idx = np.empty(n, np.int64)
        val = np.empty(n)
        j = 0
        for i in range(s, e):
            if np.isfinite(v[i]):
                idx[j] = i
                val[j] = v[i]
                j += 1
        o = np.argsort(val, kind="mergesort")
        a = 0
        while a < n:  # ties share their average rank
            b = a
            while b + 1 < n and val[o[b + 1]] == val[o[a]]:
                b += 1
            rk = (a + b) / 2.0  # 0-based average rank
            for q in range(a, b + 1):
                out[idx[o[q]]] = rk / (n - 1) - (0.5 if center else 0.0)
            a = b + 1
    return out


@njit(parallel=True, cache=True)
def seg_demean(v, sub, starts, ends, nsub):  # pragma: no cover
    out = np.full(v.size, np.nan)
    for k in prange(starts.size):
        s, e = starts[k], ends[k]
        sm = np.zeros(nsub)
        cnt = np.zeros(nsub, np.int64)
        for i in range(s, e):
            if np.isfinite(v[i]) and sub[i] >= 0:
                sm[sub[i]] += v[i]
                cnt[sub[i]] += 1
        for i in range(s, e):
            g = sub[i]
            if np.isfinite(v[i]) and g >= 0 and cnt[g] >= 3:
                out[i] = v[i] - sm[g] / cnt[g]
    return out
