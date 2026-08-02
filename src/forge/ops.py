"""Research-only transforms.

**These are not factors.** Nothing in this module computes a signal; everything
reshapes one that prism already produced. The distinction is the second build
invariant: if a transform has to run in production, it belongs in prism, not
here. A z-score applied in research and re-derived slightly differently in C++
is a live/research divergence that no backtest will ever surface.

Cross-sectional ops operate **within a timestamp**. That grouping is not
configurable, because pooling across time is the single most common way to
leak: standardising a factor using the full sample's mean subtracts a quantity
that was not knowable at ``t``.

Time-series ops operate within a symbol over a **trailing** window, for the
same reason.

Every op is a pure ``(DataFrame, **params) -> DataFrame`` registered by name, so
a config file can declare a pipeline without importing anything.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = [
    "REGISTRY",
    "register",
    "get",
    "apply_pipeline",
    "winsorize",
    "zscore",
    "rank_to_normal",
    "neutralize",
    "rolling_zscore",
    "rolling_quantile",
    "ema_ratio",
]

REGISTRY: dict[str, Callable[..., Frame]] = {}
"""Name -> transform. Populated by :func:`register`."""


def register(name: str) -> Callable[[Callable[..., Frame]], Callable[..., Frame]]:
    """Register a transform under ``name``."""

    def deco(fn: Callable[..., Frame]) -> Callable[..., Frame]:
        if name in REGISTRY:
            raise ValueError(f"transform {name!r} is already registered")
        REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Callable[..., Frame]:
    """Look up a registered transform, with a helpful error."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown transform {name!r}; registered: {sorted(REGISTRY)}") from None


def apply_pipeline(df: Frame, steps: Sequence[Mapping[str, Any]]) -> Frame:
    """Run a declared sequence of transforms.

    Each step is ``{"op": name, **params}``. Steps are applied in order and
    each sees the previous step's output, so a pipeline reads top to bottom::

        apply_pipeline(df, [
            {"op": "winsorize", "column": "alpha", "method": "mad", "k": 3.0},
            {"op": "neutralize", "column": "alpha",
             "by": ["industry"], "size_col": "market_cap"},
            {"op": "zscore", "column": "alpha"},
        ])

    Winsorize before neutralize before standardize is the conventional order:
    outliers would otherwise dominate the regression, and standardising first
    would be undone by the residualisation.
    """
    out = df
    for i, step in enumerate(steps):
        params = dict(step)
        try:
            name = params.pop("op")
        except KeyError:
            raise KeyError(f"pipeline step {i} has no 'op' key: {step!r}") from None
        out = get(str(name))(out, **params)
    return out


def _target(column: str, out: str | None) -> str:
    return column if out is None else out


@register("winsorize")
def winsorize(
    df: Frame,
    column: str,
    *,
    method: str = "mad",
    k: float = 3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    out: str | None = None,
    by: str = TS,
) -> Frame:
    """Clip cross-sectional outliers.

    Parameters
    ----------
    method:
        ``"mad"`` clips to ``median +/- k * 1.4826 * MAD``. ``"quantile"``
        clips to the ``lower``/``upper`` quantiles.
    k:
        MAD multiplier. 3.0 is the convention and corresponds to roughly 3
        standard deviations for a Gaussian, but unlike a standard-deviation
        rule it is not itself dragged around by the outliers it is meant to
        catch -- which is why MAD is the default.

    Notes
    -----
    Clipping, not dropping. A dropped name silently leaves the cross-section
    and changes the universe from one timestamp to the next.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (by, column), where="winsorize")
    tgt = _target(column, out)

    if method == "mad":
        med = pl.col(column).median().over(by)
        mad = (pl.col(column) - med).abs().median().over(by) * 1.4826
        lo, hi = med - k * mad, med + k * mad
    elif method == "quantile":
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"need 0 <= lower < upper <= 1, got {lower}, {upper}")
        lo = pl.col(column).quantile(lower).over(by)
        hi = pl.col(column).quantile(upper).over(by)
    else:
        raise ValueError(f"method must be 'mad' or 'quantile', got {method!r}")

    # A degenerate cross-section (zero spread) would clip everything to the
    # median; leave it untouched instead.
    res = lf.with_columns(
        pl.when(hi > lo).then(pl.col(column).clip(lo, hi)).otherwise(pl.col(column)).alias(tgt)
    )
    return restore(res, want)


@register("zscore")
def zscore(
    df: Frame,
    column: str,
    *,
    out: str | None = None,
    by: str = TS,
    ddof: int = 1,
) -> Frame:
    """Cross-sectional standardisation within each timestamp.

    Returns null where the cross-section has zero dispersion, rather than
    dividing by zero and producing infinities that poison every downstream
    aggregate.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (by, column), where="zscore")
    tgt = _target(column, out)

    mu = pl.col(column).mean().over(by)
    sd = pl.col(column).std(ddof=ddof).over(by)
    res = lf.with_columns(
        pl.when(sd > 0).then((pl.col(column) - mu) / sd).otherwise(None).alias(tgt)
    )
    return restore(res, want)


@register("rank_to_normal")
def rank_to_normal(
    df: Frame,
    column: str,
    *,
    out: str | None = None,
    by: str = TS,
) -> Frame:
    """Map cross-sectional ranks onto a standard normal.

    Uses the Blom transform ``Phi^-1((r - 3/8) / (n + 1/4))``. The offset keeps
    the extreme ranks off +/-infinity, which a naive ``r / (n + 1)`` would not.

    This is the most aggressive way to tame a heavy-tailed factor: it discards
    magnitude entirely and keeps only ordering. That is usually right for
    A-share factor data and always worth stating out loud, because a factor
    whose edge lived in its magnitudes will lose it here.
    """
    from scipy.special import ndtri

    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (by, column), where="rank_to_normal")
    tgt = _target(column, out)

    ranked = lf.with_columns(
        _r=pl.col(column).rank("average").over(by),
        _n=pl.col(column).is_not_null().sum().over(by),
    )
    p = ((pl.col("_r") - 0.375) / (pl.col("_n") + 0.25)).clip(1e-9, 1 - 1e-9)
    vals = ranked.with_columns(_p=pl.when(pl.col("_n") > 1).then(p).otherwise(None))

    arr = vals["_p"].to_numpy()
    z = np.full(arr.shape, np.nan)
    ok = np.isfinite(arr)
    z[ok] = ndtri(arr[ok])

    res = (
        vals.with_columns(pl.Series(tgt, z))
        .with_columns(pl.col(tgt).fill_nan(None))
        .drop("_r", "_n", "_p")
    )
    return restore(res, want)


@register("neutralize")
def neutralize(
    df: Frame,
    column: str,
    *,
    by: Sequence[str] = ("industry",),
    size_col: str | None = "market_cap",
    out: str | None = None,
    ts: str = TS,
    min_names: int = 10,
) -> Frame:
    """Residualise a factor against industry dummies and log size.

    Fits ``factor ~ industry dummies + log(size)`` by OLS **within each
    timestamp** and returns the residual. This removes the part of the factor
    explained by sector membership and market cap, which is nearly always the
    part that is already priced.

    Parameters
    ----------
    by:
        Categorical columns to dummy out. Pass ``()`` for size-only
        neutralisation.
    size_col:
        Raw market cap. Logged internally -- size effects are log-linear, and
        regressing on raw cap lets a handful of megacaps set the slope.
    min_names:
        Cross-sections with fewer valid names return null rather than a
        residual. An OLS with fewer observations than dummy levels is
        unidentified and would return an exact zero residual, which looks like
        a perfectly neutral factor rather than a failure.

    Notes
    -----
    The design matrix is solved with ``lstsq``, so a rank-deficient cross
    section (an industry with one member, a constant size column) degrades to a
    least-norm solution instead of raising.
    """
    want = flavor(df)
    lf = to_polars(df)
    need = [ts, column, *by] + ([size_col] if size_col else [])
    require_columns(lf, need, where="neutralize")
    tgt = _target(column, out)

    resid = np.full(lf.height, np.nan)
    lf = lf.with_row_index("_row")

    for part in lf.partition_by(ts, maintain_order=True):
        valid = part.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        if size_col:
            valid = valid.filter(pl.col(size_col).is_not_null() & (pl.col(size_col) > 0))
        if valid.height < min_names:
            continue

        y = valid[column].to_numpy().astype(float)
        cols: list[np.ndarray] = [np.ones(valid.height)]
        for cat in by:
            codes = valid[cat].cast(pl.Categorical).to_physical().to_numpy()
            levels = np.unique(codes)
            # Drop one level: the intercept already spans it, and keeping all
            # of them makes the design matrix singular by construction.
            for lv in levels[1:]:
                cols.append((codes == lv).astype(float))
        if size_col:
            cols.append(np.log(valid[size_col].to_numpy().astype(float)))

        x = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid[valid["_row"].to_numpy()] = y - x @ beta

    # NaN and null are distinct in polars: a NaN survives `is_not_null()` and
    # then poisons every downstream mean. Cross-sections that were skipped must
    # come back as null, matching the documented contract.
    res = lf.with_columns(pl.Series(tgt, resid)).with_columns(
        pl.col(tgt).fill_nan(None)
    ).drop("_row")
    return restore(res, want)


@register("rolling_zscore")
def rolling_zscore(
    df: Frame,
    column: str,
    *,
    window: int,
    out: str | None = None,
    over: str = SYMBOL,
    ts: str = TS,
    min_periods: int | None = None,
) -> Frame:
    """Trailing z-score within each symbol.

    The window ends at ``t`` inclusive, so the statistic is knowable at ``t``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    want = flavor(df)
    lf = to_polars(df).sort([over, ts], maintain_order=True)
    require_columns(lf, (ts, over, column), where="rolling_zscore")
    tgt = _target(column, out)
    mp = window if min_periods is None else min_periods

    mu = pl.col(column).rolling_mean(window_size=window, min_samples=mp).over(over)
    sd = pl.col(column).rolling_std(window_size=window, min_samples=mp).over(over)
    res = lf.with_columns(
        pl.when(sd > 0).then((pl.col(column) - mu) / sd).otherwise(None).alias(tgt)
    )
    return restore(res.sort([ts, over], maintain_order=True), want)


@register("rolling_quantile")
def rolling_quantile(
    df: Frame,
    column: str,
    *,
    window: int,
    quantile: float = 0.5,
    out: str | None = None,
    over: str = SYMBOL,
    ts: str = TS,
    min_periods: int | None = None,
) -> Frame:
    """Trailing quantile within each symbol."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f"quantile must be in [0, 1], got {quantile}")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    want = flavor(df)
    lf = to_polars(df).sort([over, ts], maintain_order=True)
    require_columns(lf, (ts, over, column), where="rolling_quantile")
    tgt = _target(column, out)
    mp = window if min_periods is None else min_periods

    res = lf.with_columns(
        pl.col(column)
        .rolling_quantile(quantile=quantile, window_size=window, min_samples=mp)
        .over(over)
        .alias(tgt)
    )
    return restore(res.sort([ts, over], maintain_order=True), want)


@register("ema_ratio")
def ema_ratio(
    df: Frame,
    column: str,
    *,
    fast: int,
    slow: int,
    out: str | None = None,
    over: str = SYMBOL,
    ts: str = TS,
) -> Frame:
    """Ratio of a fast to a slow exponential moving average, minus one.

    A scale-free trend measure: dividing rather than subtracting keeps the
    output comparable across instruments trading at 5 CNY and 500 CNY, which a
    raw EMA difference is not.
    """
    if fast < 1 or slow < 1:
        raise ValueError(f"spans must be >= 1, got fast={fast}, slow={slow}")
    if fast >= slow:
        raise ValueError(f"fast span must be shorter than slow, got {fast} >= {slow}")

    want = flavor(df)
    lf = to_polars(df).sort([over, ts], maintain_order=True)
    require_columns(lf, (ts, over, column), where="ema_ratio")
    tgt = _target(column, out)

    f = pl.col(column).ewm_mean(span=fast, ignore_nulls=True).over(over)
    s = pl.col(column).ewm_mean(span=slow, ignore_nulls=True).over(over)
    res = lf.with_columns(pl.when(s != 0).then(f / s - 1.0).otherwise(None).alias(tgt))
    return restore(res.sort([ts, over], maintain_order=True), want)
