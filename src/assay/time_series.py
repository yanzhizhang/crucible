"""Time-series metrics for single-instrument research.

The cross-sectional half of :mod:`assay` answers "does this factor rank names
correctly". This half answers "does this return stream make money", which is
the right question for futures and single-instrument work.

Annualisation is explicit everywhere. ``periods_per_year`` must match the
sampling grain -- :data:`SESSIONS_PER_YEAR` for daily A-share data, and that
times the bars per session for intraday. Defaulting it silently is how two
tearsheets of the same strategy come to disagree by a factor of 15.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from itertools import product

import numpy as np
import polars as pl

from assay.results import PerformanceResult, SurfaceResult
from crucible.frames import TS, Frame, require_columns, to_polars

__all__ = [
    "SESSIONS_PER_YEAR",
    "sharpe",
    "max_drawdown",
    "drawdown_duration",
    "calmar",
    "monthly_winrate",
    "performance",
    "rolling_corr",
    "conditional_return_by_quantile",
    "parameter_surface",
    "periods_per_year_for",
]

SESSIONS_PER_YEAR = 244.0
"""Mainland A-share trading sessions per year, after CSRC holidays."""

_NEGLIGIBLE = 1e-9
"""Dispersion below this multiple of the mean counts as no dispersion.

Guards every ratio that divides by a standard deviation. Floating-point noise
in a constant series is ~1e-19, not 0, so an exact-zero test lets a degenerate
stream report an astronomic Sharpe.
"""


def periods_per_year_for(bars_per_session: int = 1) -> float:
    """Annualisation constant for a given intraday grain.

    ``periods_per_year_for(240)`` for one-minute bars, ``periods_per_year_for(4800)``
    for 3-second slots.
    """
    if bars_per_session < 1:
        raise ValueError(f"bars_per_session must be >= 1, got {bars_per_session}")
    return SESSIONS_PER_YEAR * bars_per_session


def _returns(x: Frame | pl.Series | np.ndarray, column: str = "ret") -> np.ndarray:
    """Coerce a return stream to a finite 1-D array."""
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, pl.Series):
        arr = x.to_numpy()
    else:
        df = to_polars(x)
        require_columns(df, (column,), where="time-series metric")
        arr = df[column].to_numpy()
    arr = np.asarray(arr, dtype=float)
    return arr[np.isfinite(arr)]


def sharpe(
    returns: Frame | pl.Series | np.ndarray,
    *,
    periods_per_year: float = SESSIONS_PER_YEAR,
    rf: float = 0.0,
    column: str = "ret",
) -> float:
    """Annualised Sharpe ratio of a simple-return stream.

    ``rf`` is the risk-free rate **per period**, not annualised. Chinese
    research conventionally uses zero, which is what the default assumes.
    """
    r = _returns(returns, column)
    if r.size < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    mean = float(r.mean())
    # `sd <= 0` is not a sufficient guard. A constant series computes a
    # standard deviation of ~1e-19 rather than exactly zero, and the ratio then
    # reports a Sharpe of 1e16 -- which looks like a spectacular strategy
    # instead of a degenerate return stream. Compare against the scale of the
    # returns themselves, not against zero.
    if sd <= _NEGLIGIBLE * max(abs(mean), 1e-12):
        return 0.0
    return float((mean - rf) / sd * math.sqrt(periods_per_year))


def _equity(r: np.ndarray) -> np.ndarray:
    """Compounded equity curve starting at 1.0."""
    return np.cumprod(1.0 + r)


def max_drawdown(
    returns: Frame | pl.Series | np.ndarray, *, column: str = "ret"
) -> float:
    """Worst peak-to-trough decline, as a negative fraction.

    Computed on the **compounded** curve. Summing arithmetic returns instead
    understates drawdown precisely when it matters most, in the large-loss tail.
    """
    r = _returns(returns, column)
    if r.size == 0:
        return 0.0
    eq = _equity(r)
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def drawdown_duration(
    returns: Frame | pl.Series | np.ndarray, *, column: str = "ret"
) -> int:
    """Longest run of periods spent below a previous peak.

    Often the more decisive statistic than depth: a 12% drawdown lasting three
    days is noise, the same 12% lasting fourteen months ends mandates.
    """
    r = _returns(returns, column)
    if r.size == 0:
        return 0
    eq = _equity(r)
    peak = np.maximum.accumulate(eq)
    underwater = eq < peak
    longest = run = 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def calmar(
    returns: Frame | pl.Series | np.ndarray,
    *,
    periods_per_year: float = SESSIONS_PER_YEAR,
    column: str = "ret",
) -> float:
    """Annualised return divided by the absolute maximum drawdown."""
    r = _returns(returns, column)
    if r.size == 0:
        return 0.0
    mdd = abs(max_drawdown(r))
    if mdd <= 0:
        return 0.0
    years = r.size / periods_per_year
    if years <= 0:
        return 0.0
    ann = _equity(r)[-1] ** (1.0 / years) - 1.0
    return float(ann / mdd)


def monthly_winrate(df: Frame, *, column: str = "ret", ts: str = TS) -> float:
    """Fraction of calendar months with a positive compounded return.

    Needs real timestamps, so it takes a frame rather than a bare array.
    """
    lf = to_polars(df)
    require_columns(lf, (ts, column), where="monthly_winrate")
    monthly = (
        lf.filter(pl.col(column).is_not_null() & pl.col(column).is_finite())
        .group_by(pl.col(ts).dt.truncate("1mo").alias("_m"))
        .agg(((pl.col(column) + 1.0).product() - 1.0).alias("m_ret"))
    )
    if monthly.height == 0:
        return 0.0
    return float((monthly["m_ret"].to_numpy() > 0).mean())


def performance(
    df: Frame,
    *,
    column: str = "ret",
    ts: str = TS,
    periods_per_year: float = SESSIONS_PER_YEAR,
) -> PerformanceResult:
    """Full performance summary for a return stream.

    Returns
    -------
    PerformanceResult
        Carrying ``n`` and an ``is_thin`` flag, so a nine-month backtest
        reporting a Sharpe of 3 announces its own fragility.
    """
    lf = to_polars(df)
    require_columns(lf, (column,), where="performance")
    r = _returns(lf, column)
    n = int(r.size)

    if n == 0:
        return PerformanceResult(0, periods_per_year, 0, 0, 0, 0, 0, 0, 0, 0, pl.DataFrame())

    eq = _equity(r)
    years = n / periods_per_year
    total = float(eq[-1] - 1.0)
    ann = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else 0.0

    equity_df = (
        lf.select(ts).head(n).with_columns(equity=pl.Series("equity", eq))
        if ts in lf.columns
        else pl.DataFrame({"equity": eq})
    )
    wr = monthly_winrate(lf, column=column, ts=ts) if ts in lf.columns else 0.0

    return PerformanceResult(
        n=n,
        periods_per_year=periods_per_year,
        total_return=total,
        ann_return=ann,
        ann_vol=vol,
        sharpe=sharpe(r, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(r),
        drawdown_duration=drawdown_duration(r),
        calmar=calmar(r, periods_per_year=periods_per_year),
        monthly_winrate=wr,
        equity=equity_df,
    )


def rolling_corr(
    df: Frame,
    a: str,
    b: str,
    window: int,
    *,
    ts: str = TS,
    min_periods: int | None = None,
) -> pl.DataFrame:
    """Trailing correlation between two columns.

    The window is **trailing**, so the value at ``t`` uses only observations at
    or before ``t`` and is safe to use as a feature. A centred window would
    read the future.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    lf = to_polars(df).sort(ts, maintain_order=True)
    require_columns(lf, (ts, a, b), where="rolling_corr")
    mp = window if min_periods is None else min_periods
    return lf.select(
        ts,
        pl.rolling_corr(pl.col(a), pl.col(b), window_size=window, min_samples=mp).alias("corr"),
    )


def conditional_return_by_quantile(
    df: Frame,
    signal: str,
    label: str,
    n: int = 5,
    *,
    min_obs: int = 30,
) -> pl.DataFrame:
    """Mean forward return conditioned on the signal's own historical quantile.

    The single-instrument analogue of :func:`assay.quantile_returns`. Buckets
    are formed over the **whole sample**, which makes this an in-sample
    descriptive tool: it shows the shape of the conditional relationship, not a
    tradable result. For a tradable version the breakpoints must be estimated
    on a trailing window only.

    Returns
    -------
    ``bucket``, ``lo``, ``hi``, ``mean_ret``, ``n``, sorted by bucket.
    """
    if n < 2:
        raise ValueError(f"need at least 2 buckets, got {n}")
    lf = to_polars(df).filter(
        pl.col(signal).is_not_null()
        & pl.col(label).is_not_null()
        & pl.col(signal).is_finite()
        & pl.col(label).is_finite()
    )
    if lf.height < min_obs:
        raise ValueError(f"need at least {min_obs} observations, got {lf.height}")

    return (
        lf.with_columns(
            ((pl.col(signal).rank("ordinal") - 1) * n // pl.len())
            .clip(0, n - 1)
            .cast(pl.Int32)
            .alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.col(signal).min().alias("lo"),
            pl.col(signal).max().alias("hi"),
            pl.col(label).mean().alias("mean_ret"),
            pl.len().alias("n"),
        )
        .sort("bucket")
    )


def parameter_surface(
    grid: Mapping[str, Sequence[object]],
    score_fn: Callable[..., float],
    *,
    plateau_threshold: float = 0.6,
) -> SurfaceResult:
    """Evaluate a parameter grid and judge whether the best point is robust.

    A genuine edge is a *plateau*: neighbouring parameter values score
    similarly, because the effect is real and not tuned to the sample. An
    isolated peak -- one combination scoring far above every neighbour -- is the
    signature of a sweep that has fitted noise. This function reports which one
    you have.

    Parameters
    ----------
    grid:
        ``{param_name: [values]}``. Evaluated as the full Cartesian product.
    score_fn:
        Called with each combination as keyword arguments; returns the score to
        maximise (Sharpe, IC, whatever). Must be deterministic, or the
        neighbourhood comparison is meaningless.
    plateau_threshold:
        Neighbour-to-peak score ratio below which the peak is called isolated.

    Returns
    -------
    SurfaceResult
        With ``is_isolated_peak`` set. Treat that as a rejection: the correct
        response is to distrust the parameterisation, not to tune it further.
    """
    names = list(grid)
    if not names:
        raise ValueError("grid is empty")
    axes = [list(grid[k]) for k in names]

    rows: list[dict[str, object]] = []
    for combo in product(*axes):
        kwargs = dict(zip(names, combo))
        rows.append({**kwargs, "score": float(score_fn(**kwargs))})

    table = pl.DataFrame(rows)
    scores = table["score"].to_numpy()
    best_i = int(np.nanargmax(scores))
    best_score = float(scores[best_i])
    best = {k: table[k][best_i] for k in names}

    # Neighbours differ by one step on exactly one axis.
    idx_of = {k: axes[j].index(best[k]) for j, k in enumerate(names)}
    neighbours: list[float] = []
    for j, k in enumerate(names):
        for step in (-1, 1):
            pos = idx_of[k] + step
            if 0 <= pos < len(axes[j]):
                probe = dict(best)
                probe[k] = axes[j][pos]
                match = table
                for kk, vv in probe.items():
                    match = match.filter(pl.col(kk) == vv)
                if match.height:
                    neighbours.append(float(match["score"][0]))

    if not neighbours or best_score <= 0:
        plateau = 0.0
    else:
        plateau = float(np.mean(neighbours) / best_score)

    return SurfaceResult(
        grid=table,
        best=best,  # type: ignore[arg-type]
        best_score=best_score,
        plateau_score=plateau,
        is_isolated_peak=bool(neighbours) and plateau < plateau_threshold,
        n_points=table.height,
    )
