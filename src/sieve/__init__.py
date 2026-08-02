"""sieve -- screening and orthogonalization.

Decides whether a candidate factor adds independent information to a pool, and
records why. Admission is a three-way decision -- admit, reject as duplicate,
or admit after residualising -- because discarding a correlated factor outright
throws away whatever genuinely new component it carries.
"""

from __future__ import annotations

from sieve.screen import (
    AdmissionRecord,
    FactorPool,
    ScreenResult,
    correlation_matrix,
    orthogonalize,
    screen,
)

__all__ = [
    "AdmissionRecord",
    "FactorPool",
    "ScreenResult",
    "correlation_matrix",
    "orthogonalize",
    "screen",
]
