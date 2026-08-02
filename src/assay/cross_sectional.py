"""Cross-sectional single-factor evaluation.

Every metric here treats all names at one timestamp as **one sample**. That is
the grouping the build contract mandates and it is not a formality: pooling
observations across time and computing a single correlation over the whole
panel mixes cross-sectional signal with time-series drift, and the resulting
number answers no question anyone asked.

So each function produces a *time series of cross-sectional statistics*, then
summarises that series. The summary carries its own sample count, because the
number of cross-sections is the real degrees of freedom -- not the number of
rows.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from assay.results import DecayResult, ICResult, QuantileResult, TurnoverResult
from crucible.frames import SYMBOL, TS, Frame, require_columns, to_polars

__all__ = ["ic", "ic_summary", "quantile_returns", "decay", "turnover"]


def _clean(df: pl.DataFrame, factor: str, label: str) -> pl.DataFrame:
    """Drop rows where either side is missing or non-finite.

    Masked observations arrive as null by design (see :mod:`almanac.masks`).
    They must leave the correlation entirely rather than being filled, since
    any fill value invents a rank.
    """
    return df.filter(
        pl.col(factor).is_not_null()
        & pl.col(label).is_not_null()
        & pl.col(factor).is_finite()
        & pl.col(label).is_finite()
    )


def ic(
    df: Frame,
    factor: str,
    label: str,
    *,
    method: str = "spearman",
    min_names: int = 20,
) -> ICResult:
    """Cross-sectional IC per timestamp, plus its summary.

    Parameters
    ----------
    method:
        ``"spearman"`` (default) or ``"pearson"``. Rank correlation is the
        default because factor distributions in A-shares are heavy-tailed and a
        handful of extreme values otherwise dominate a Pearson estimate.
    min_names:
        Cross-sections with fewer valid names produce null rather than a
        correlation. A 4-name "cross-section" yields a number between -1 and 1
        that carries no information, and averaging those in corrupts the mean.

    Returns
    -------
    ICResult
        With the per-timestamp series retained.

    Point-in-time contract
    ----------------------
    Purely descriptive: reads whatever ``factor`` and ``label`` already hold.
    Whether the pair is PIT-valid is decided upstream, by how the label was
    built -- see :func:`horizon.forward_return`.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}")

    lf = to_polars(df)
    require_columns(lf, (TS, factor, label), where="ic")

    series = (
        _clean(lf, factor, label)
        .group_by(TS)
        .agg(
            pl.corr(factor, label, method=method).alias("ic"),
            pl.len().alias("n"),
        )
        .with_columns(pl.when(pl.col("n") >= min_names).then(pl.col("ic")).alias("ic"))
        .sort(TS)
    )
    return ic_summary(series, method=method)


def ic_summary(
    series: Frame,
    *,
    method: str = "spearman",
    newey_west_lags: int = 0,
) -> ICResult:
    """Summarise an IC series into mean, ICIR, t-stat and hit rate.

    Parameters
    ----------
    newey_west_lags:
        Autocorrelation lags to correct the t-statistic for. **Set this to
        ``horizon - 1`` whenever the label windows overlap.** Consecutive
        5-day forward returns sampled daily share four days of data, so the IC
        series is strongly autocorrelated and the naive
        ``sqrt(n)`` t-stat overstates significance -- frequently by a factor of
        two or more, which is the difference between "publishable" and
        "nothing".

    Returns
    -------
    ICResult
        ``t_stat`` is Newey-West corrected when ``newey_west_lags > 0``.
    """
    s = to_polars(series)
    require_columns(s, ("ic",), where="ic_summary")

    vals = s.filter(pl.col("ic").is_not_null())["ic"].to_numpy()
    n = int(vals.size)
    breadth = float(s["n"].mean()) if "n" in s.columns and s.height else 0.0

    if n == 0:
        return ICResult(s, 0.0, 0.0, 0.0, 0.0, 0.0, 0, breadth, method, newey_west_lags)

    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if n > 1 else 0.0
    icir = mean / std if std > 0 else 0.0

    if std <= 0 or n < 2:
        t_stat = 0.0
    elif newey_west_lags > 0:
        t_stat = mean / _newey_west_se(vals, newey_west_lags)
    else:
        t_stat = icir * math.sqrt(n)

    return ICResult(
        series=s,
        mean=mean,
        std=std,
        icir=icir,
        t_stat=float(t_stat),
        positive_rate=float((vals > 0).mean()),
        n_periods=n,
        mean_breadth=breadth,
        method=method,
        newey_west_lags=newey_west_lags,
    )


def _newey_west_se(x: np.ndarray, lags: int) -> float:
    """Newey-West standard error of the mean, Bartlett-kernel weighted."""
    n = x.size
    dev = x - x.mean()
    gamma0 = float(dev @ dev) / n
    total = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        gamma_k = float(dev[k:] @ dev[:-k]) / n
        total += 2.0 * (1.0 - k / (lags + 1.0)) * gamma_k
    total = max(total, 1e-24)
    return math.sqrt(total / n)


def quantile_returns(
    df: Frame,
    factor: str,
    label: str,
    n: int = 10,
    *,
    min_names: int | None = None,
    weight: str | None = None,
) -> QuantileResult:
    """Bucket names by factor rank each period and track forward returns.

    Parameters
    ----------
    n:
        Number of buckets. Bucket 0 holds the lowest factor values.
    min_names:
        Skip cross-sections thinner than this. Defaults to ``2 * n``, so every
        bucket can hold at least two names -- below that, a single outlier
        becomes an entire bucket's return.
    weight:
        Optional column for weighting within a bucket (e.g. market cap).
        Defaults to equal weight.

    Returns
    -------
    QuantileResult
        Per-bucket means, cumulative curves, the long-short spread, and a
        monotonicity score.

    Notes
    -----
    Ranks are computed **within each timestamp**, never pooled. Pooling would
    let a period of generally high factor values populate the top bucket
    regardless of relative standing.
    """
    if n < 2:
        raise ValueError(f"need at least 2 buckets, got {n}")
    floor = 2 * n if min_names is None else min_names

    lf = _clean(to_polars(df), factor, label)
    require_columns(lf, (TS, SYMBOL, factor, label), where="quantile_returns")

    w = pl.col(weight) if weight is not None else pl.lit(1.0)

    bucketed = (
        lf.with_columns(pl.len().over(TS).alias("_n"))
        .filter(pl.col("_n") >= floor)
        .with_columns(
            (
                (pl.col(factor).rank("ordinal").over(TS) - 1)
                * n
                // pl.col("_n")
            )
            .clip(0, n - 1)
            .cast(pl.Int32)
            .alias("bucket")
        )
        .with_columns(_w=w)
    )

    per_period = (
        bucketed.group_by([TS, "bucket"])
        .agg(
            ((pl.col(label) * pl.col("_w")).sum() / pl.col("_w").sum()).alias("ret"),
            pl.len().alias("n"),
        )
        .sort([TS, "bucket"])
    )

    curves = per_period.with_columns(
        cum_ret=(pl.col("ret") + 1.0).cum_prod().over("bucket") - 1.0
    )

    by_bucket = (
        per_period.group_by("bucket")
        .agg(pl.col("ret").mean().alias("mean_ret"), pl.col("n").sum().alias("n"))
        .sort("bucket")
    )

    wide = per_period.pivot(on="bucket", index=TS, values="ret").sort(TS)
    top, bot = str(n - 1), "0"
    if top in wide.columns and bot in wide.columns:
        spread = wide.select(
            TS, (pl.col(top) - pl.col(bot)).alias("spread")
        ).drop_nulls()
    else:
        spread = pl.DataFrame({TS: [], "spread": []})
    spread = spread.with_columns(cum_spread=(pl.col("spread") + 1.0).cum_prod() - 1.0)

    sv = spread["spread"].to_numpy()
    spread_mean = float(sv.mean()) if sv.size else 0.0
    spread_sd = float(sv.std(ddof=1)) if sv.size > 1 else 0.0
    spread_sharpe = spread_mean / spread_sd if spread_sd > 0 else 0.0

    mono = 0.0
    if by_bucket.height > 1:
        mono_df = by_bucket.select(
            pl.corr("bucket", "mean_ret", method="spearman").alias("m")
        )
        mono = float(mono_df["m"][0] or 0.0)

    return QuantileResult(
        by_bucket=by_bucket,
        curves=curves,
        spread=spread,
        monotonicity=mono,
        spread_mean=spread_mean,
        spread_sharpe=spread_sharpe,
        n_buckets=n,
        n_periods=int(spread.height),
    )


def decay(
    df: Frame,
    factor: str,
    labels_by_horizon: Mapping[int, str],
    *,
    method: str = "spearman",
    min_names: int = 20,
) -> DecayResult:
    """IC against horizon, with a half-life estimate.

    Parameters
    ----------
    labels_by_horizon:
        ``{horizon: label_column}``. Build these with
        :func:`horizon.forward_return` at each horizon; they must all share the
        same entry lag or the curve compares different things.

    Returns
    -------
    DecayResult
        The curve plus the interpolated half-life.

    Notes
    -----
    Each horizon's t-stat is Newey-West corrected with ``horizon - 1`` lags,
    since longer-horizon labels overlap more and would otherwise look
    artificially more significant than short ones -- exactly backwards.
    """
    if not labels_by_horizon:
        raise ValueError("labels_by_horizon is empty")

    lf = to_polars(df)
    rows: list[dict[str, float | int]] = []
    for h in sorted(labels_by_horizon):
        col = labels_by_horizon[h]
        raw = ic(lf, factor, col, method=method, min_names=min_names)
        corrected = ic_summary(raw.series, method=method, newey_west_lags=max(0, h - 1))
        rows.append(
            {
                "horizon": h,
                "ic": corrected.mean,
                "t_stat": corrected.t_stat,
                "n_periods": corrected.n_periods,
            }
        )

    curve = pl.DataFrame(rows).sort("horizon")
    ics = curve["ic"].to_numpy()
    horizons = curve["horizon"].to_numpy()

    k = int(np.argmax(np.abs(ics)))
    peak_h, peak_ic = int(horizons[k]), float(ics[k])
    return DecayResult(
        curve=curve,
        peak_horizon=peak_h,
        peak_ic=peak_ic,
        half_life=_half_life(horizons, ics, k),
    )


def _half_life(horizons: np.ndarray, ics: np.ndarray, peak: int) -> float:
    """Linearly interpolated horizon where |IC| first halves after the peak."""
    target = abs(ics[peak]) / 2.0
    if target <= 0:
        return float("inf")
    for i in range(peak + 1, len(ics)):
        if abs(ics[i]) <= target:
            x0, x1 = float(horizons[i - 1]), float(horizons[i])
            y0, y1 = abs(float(ics[i - 1])), abs(float(ics[i]))
            if y0 == y1:
                return x1
            return x0 + (y0 - target) * (x1 - x0) / (y0 - y1)
    return float("inf")


def turnover(
    df: Frame,
    factor: str,
    *,
    method: str = "spearman",
    min_names: int = 20,
) -> TurnoverResult:
    """Rank turnover: ``1 - corr(rank[t], rank[t-1])``.

    Consecutive cross-sections are matched **by symbol**, so a name that enters
    or leaves the universe contributes to neither side rather than shifting the
    alignment. Positional differencing would report spurious turnover every
    time the universe changes size.

    Returns
    -------
    TurnoverResult
        The per-period series plus mean and implied holding period.
    """
    lf = to_polars(df).filter(pl.col(factor).is_not_null() & pl.col(factor).is_finite())
    require_columns(lf, (TS, SYMBOL, factor), where="turnover")

    ranked = lf.with_columns(
        _rank=pl.col(factor).rank("average").over(TS),
        _n=pl.len().over(TS),
    ).filter(pl.col("_n") >= min_names)

    stamps = ranked[TS].unique().sort().to_list()
    rows: list[dict[str, object]] = []
    for prev, cur in zip(stamps[:-1], stamps[1:]):
        a = ranked.filter(pl.col(TS) == prev).select(SYMBOL, _prev="_rank")
        b = ranked.filter(pl.col(TS) == cur).select(SYMBOL, _cur="_rank")
        joined = a.join(b, on=SYMBOL, how="inner")
        if joined.height < min_names:
            continue
        corr = joined.select(pl.corr("_prev", "_cur", method=method).alias("c"))["c"][0]
        if corr is None:
            continue
        rows.append({TS: cur, "turnover": 1.0 - float(corr), "n": joined.height})

    series = pl.DataFrame(rows) if rows else pl.DataFrame({TS: [], "turnover": [], "n": []})
    vals = series["turnover"].to_numpy() if series.height else np.array([])

    return TurnoverResult(
        series=series,
        mean=float(vals.mean()) if vals.size else 0.0,
        std=float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        n_periods=int(vals.size),
    )
