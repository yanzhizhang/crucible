"""kiln -- models.

Purged walk-forward validation with embargo, LightGBM fitting with determinism
forced on, and a phase-randomized noise benchmark. Random K-fold is not
supported and will not be: all names at one timestamp are a single sample, and
label windows overlap across time.
"""

from __future__ import annotations

from kiln.models import (
    DEFAULT_LGBM_PARAMS,
    FitResult,
    NoiseBenchmarkResult,
    WalkForwardResult,
    assert_no_feature_leakage,
    fit_lgbm,
    noise_benchmark,
    phase_randomize,
    walk_forward,
)
from kiln.splits import CombinatorialPurgedCV, PurgedWalkForward, assert_no_leakage

__all__ = [
    "DEFAULT_LGBM_PARAMS",
    "CombinatorialPurgedCV",
    "FitResult",
    "NoiseBenchmarkResult",
    "PurgedWalkForward",
    "WalkForwardResult",
    "assert_no_feature_leakage",
    "assert_no_leakage",
    "fit_lgbm",
    "noise_benchmark",
    "phase_randomize",
    "walk_forward",
]
