"""forge -- research-only feature transforms.

Reshapes factors that prism already computed. Nothing here creates a signal,
and nothing here may be relied on in production: if a transform must exist
live, it belongs in prism. Cross-sectional ops group within a timestamp and
time-series ops use trailing windows, both by construction rather than by
convention.
"""

from __future__ import annotations

from forge.ops import (
    REGISTRY,
    apply_pipeline,
    ema_ratio,
    get,
    neutralize,
    rank_to_normal,
    register,
    rolling_quantile,
    rolling_zscore,
    winsorize,
    zscore,
)

__all__ = [
    "REGISTRY",
    "apply_pipeline",
    "ema_ratio",
    "get",
    "neutralize",
    "rank_to_normal",
    "register",
    "rolling_quantile",
    "rolling_zscore",
    "winsorize",
    "zscore",
]
