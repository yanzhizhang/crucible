"""ballast -- portfolio construction.

Scores to weights, constraints applied by projection rather than rejection, and
index-futures hedging. Covariance estimators are separated out because the
sample covariance of a wide cross-section is singular and its extreme
eigenvalues are noise -- handing it to an optimizer maximises estimation error.
"""

from __future__ import annotations

from ballast.covariance import factor_covariance, ledoit_wolf, nearest_psd, sample_covariance
from ballast.portfolio import (
    INDEX_MULTIPLIERS,
    Constraints,
    HedgeResult,
    apply_constraints,
    combine_scores,
    market_neutral,
    score_to_weight,
)

__all__ = [
    "INDEX_MULTIPLIERS",
    "Constraints",
    "HedgeResult",
    "apply_constraints",
    "combine_scores",
    "factor_covariance",
    "ledoit_wolf",
    "market_neutral",
    "nearest_psd",
    "sample_covariance",
    "score_to_weight",
]
