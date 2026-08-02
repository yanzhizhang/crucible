"""assay -- single-factor evaluation.

Cross-sectional metrics treat all names at one timestamp as one sample and
produce a time series of statistics; time-series metrics evaluate a single
return stream. Every metric returns a dataclass carrying its sample count, so a
result computed on eleven cross-sections cannot masquerade as one computed on
two thousand.
"""

from __future__ import annotations

from assay.cross_sectional import decay, ic, ic_summary, quantile_returns, turnover
from assay.results import (
    DecayResult,
    ICResult,
    PerformanceResult,
    QuantileResult,
    SurfaceResult,
    TurnoverResult,
)
from assay.time_series import (
    SESSIONS_PER_YEAR,
    calmar,
    conditional_return_by_quantile,
    drawdown_duration,
    max_drawdown,
    monthly_winrate,
    parameter_surface,
    performance,
    periods_per_year_for,
    rolling_corr,
    sharpe,
)

__all__ = [
    "SESSIONS_PER_YEAR",
    "DecayResult",
    "ICResult",
    "PerformanceResult",
    "QuantileResult",
    "SurfaceResult",
    "TurnoverResult",
    "calmar",
    "conditional_return_by_quantile",
    "decay",
    "drawdown_duration",
    "ic",
    "ic_summary",
    "max_drawdown",
    "monthly_winrate",
    "parameter_surface",
    "performance",
    "periods_per_year_for",
    "quantile_returns",
    "rolling_corr",
    "sharpe",
    "turnover",
]
