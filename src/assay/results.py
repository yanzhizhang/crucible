"""Result types for factor evaluation.

Every metric in :mod:`assay` returns one of these rather than a bare float.
The reason is sample size. An IC of 0.09 computed over 2000 cross-sections is a
finding; the same 0.09 over 11 cross-sections is noise, and a float cannot tell
you which one you are looking at. Carrying ``n`` in the type makes a thin slice
visible at the call site instead of three plots later.

Each result also keeps the underlying series where one exists, so a caller can
plot or re-aggregate without recomputing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

__all__ = [
    "ICResult",
    "QuantileResult",
    "DecayResult",
    "TurnoverResult",
    "PerformanceResult",
    "SurfaceResult",
]


@dataclass(frozen=True)
class ICResult:
    """Cross-sectional information coefficient over time.

    Attributes
    ----------
    series:
        ``ts``, ``ic``, ``n`` -- one row per cross-section, with the number of
        names that entered that correlation.
    mean, std:
        Moments of the IC series.
    icir:
        ``mean / std``. The IC's own signal-to-noise ratio.
    t_stat:
        ``icir * sqrt(n_periods)``, optionally Newey-West corrected.
    newey_west_lags:
        Lags used for the autocorrelation correction. Non-zero whenever the
        label horizon overlaps between consecutive observations -- without it
        the t-stat on an overlapping label is inflated, often by 2-3x.
    positive_rate:
        Fraction of cross-sections with IC > 0. A factor with a good mean but a
        50% hit rate is carried by a few periods.
    """

    series: pl.DataFrame
    mean: float
    std: float
    icir: float
    t_stat: float
    positive_rate: float
    n_periods: int
    mean_breadth: float
    method: str = "spearman"
    newey_west_lags: int = 0

    @property
    def is_thin(self) -> bool:
        """Whether the sample is too small to interpret (< 30 cross-sections)."""
        return self.n_periods < 30

    def summary(self) -> dict[str, float | int | str | bool]:
        """Flat mapping for logging or a report table."""
        return {
            "method": self.method,
            "mean": self.mean,
            "std": self.std,
            "icir": self.icir,
            "t_stat": self.t_stat,
            "positive_rate": self.positive_rate,
            "n_periods": self.n_periods,
            "mean_breadth": self.mean_breadth,
            "newey_west_lags": self.newey_west_lags,
            "is_thin": self.is_thin,
        }

    def __repr__(self) -> str:
        warn = "  [THIN SAMPLE]" if self.is_thin else ""
        return (
            f"ICResult({self.method}: mean={self.mean:+.4f} ICIR={self.icir:+.3f} "
            f"t={self.t_stat:+.2f} pos={self.positive_rate:.1%} "
            f"periods={self.n_periods} breadth={self.mean_breadth:.0f}){warn}"
        )


@dataclass(frozen=True)
class QuantileResult:
    """Bucketed forward returns and the long-short spread.

    Attributes
    ----------
    by_bucket:
        ``bucket``, ``mean_ret``, ``n`` -- average forward return per quantile.
    curves:
        ``ts``, ``bucket``, ``ret``, ``cum_ret`` -- cumulative curve per bucket.
    spread:
        ``ts``, ``spread``, ``cum_spread`` -- top minus bottom bucket.
    monotonicity:
        Spearman correlation between bucket ordinal and mean return, in
        ``[-1, 1]``. This matters more than the spread: a factor whose extreme
        buckets work but whose middle is scrambled is usually picking up an
        outlier effect, not a monotone exposure.
    """

    by_bucket: pl.DataFrame
    curves: pl.DataFrame
    spread: pl.DataFrame
    monotonicity: float
    spread_mean: float
    spread_sharpe: float
    n_buckets: int
    n_periods: int

    def __repr__(self) -> str:
        return (
            f"QuantileResult(q={self.n_buckets} spread={self.spread_mean:+.4%} "
            f"sharpe={self.spread_sharpe:+.2f} monotonicity={self.monotonicity:+.2f} "
            f"periods={self.n_periods})"
        )


@dataclass(frozen=True)
class DecayResult:
    """IC as a function of holding horizon.

    Attributes
    ----------
    curve:
        ``horizon``, ``ic``, ``t_stat``, ``n_periods``.
    half_life:
        Horizon at which IC falls to half its peak, linearly interpolated.
        ``inf`` when the curve never decays within the horizons tested --
        which usually means the horizons are too short, not that the factor is
        immortal.
    """

    curve: pl.DataFrame
    peak_horizon: int
    peak_ic: float
    half_life: float

    def __repr__(self) -> str:
        hl = "inf" if math.isinf(self.half_life) else f"{self.half_life:.1f}"
        return (
            f"DecayResult(peak={self.peak_ic:+.4f}@h{self.peak_horizon} half_life={hl})"
        )


@dataclass(frozen=True)
class TurnoverResult:
    """Rank turnover between consecutive cross-sections.

    ``1 - corr(rank[t], rank[t-1])``. Zero means the ordering is unchanged;
    one means it was fully reshuffled.

    ``implied_holding_periods`` is ``1 / mean`` -- roughly how many slots a name
    survives in the ranking. Compare it against the decay half-life: a factor
    that turns over faster than its own alpha decays is paying costs for
    nothing.
    """

    series: pl.DataFrame
    mean: float
    std: float
    n_periods: int

    @property
    def implied_holding_periods(self) -> float:
        """Average slots a name persists in the ranking."""
        return float("inf") if self.mean <= 0 else 1.0 / self.mean

    def __repr__(self) -> str:
        return (
            f"TurnoverResult(mean={self.mean:.3f} "
            f"holding={self.implied_holding_periods:.1f} periods={self.n_periods})"
        )


@dataclass(frozen=True)
class PerformanceResult:
    """Time-series performance statistics for a return stream.

    All ratios are annualised using ``periods_per_year``, which must match the
    sampling grain -- 244 for A-share daily sessions, 244*240 for one-minute
    bars. Getting it wrong scales every ratio by a constant and is the most
    common reason two tearsheets disagree.
    """

    n: int
    periods_per_year: float
    total_return: float
    ann_return: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    drawdown_duration: int
    calmar: float
    monthly_winrate: float
    equity: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)

    @property
    def is_thin(self) -> bool:
        """Fewer than one year of observations."""
        return self.n < self.periods_per_year

    def summary(self) -> dict[str, float | int | bool]:
        """Flat mapping for logging or a report table."""
        return {
            "n": self.n,
            "total_return": self.total_return,
            "ann_return": self.ann_return,
            "ann_vol": self.ann_vol,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "calmar": self.calmar,
            "monthly_winrate": self.monthly_winrate,
            "is_thin": self.is_thin,
        }

    def __repr__(self) -> str:
        warn = "  [< 1y]" if self.is_thin else ""
        return (
            f"PerformanceResult(sharpe={self.sharpe:+.2f} ann={self.ann_return:+.2%} "
            f"mdd={self.max_drawdown:.2%} calmar={self.calmar:+.2f} n={self.n}){warn}"
        )


@dataclass(frozen=True)
class SurfaceResult:
    """A parameter sweep, scored for robustness.

    Attributes
    ----------
    grid:
        One row per parameter combination with its score.
    best:
        The highest-scoring combination.
    plateau_score:
        Mean score of the best point's immediate neighbours divided by the best
        score itself. Near 1.0 means the peak sits on a plateau; near 0 means
        it is a spike.
    is_isolated_peak:
        True when the best point's neighbourhood collapses. A real edge is
        robust to a small parameter change; an isolated peak is almost always
        the sweep having fitted the sample. Treat it as a rejection, not a
        result to tune further.
    """

    grid: pl.DataFrame
    best: dict[str, float | int | str]
    best_score: float
    plateau_score: float
    is_isolated_peak: bool
    n_points: int

    def __repr__(self) -> str:
        verdict = "ISOLATED PEAK -- likely overfit" if self.is_isolated_peak else "plateau"
        return (
            f"SurfaceResult(best={self.best_score:+.4f} plateau={self.plateau_score:.2f} "
            f"points={self.n_points} -> {verdict})"
        )
