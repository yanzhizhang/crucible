"""Shared primitives for the crucible research stack.

This package holds only what every component needs: the frame bridge, the
invariant-violation exceptions, and determinism helpers. It carries no
research logic, so a subpackage that is later split into its own repo takes
this along as a small dependency rather than inheriting a framework.
"""

from __future__ import annotations

from crucible.determinism import DEFAULT_SEED, rng, seed_all, stable_hash
from crucible.errors import (
    CrucibleError,
    LeakageError,
    ParityError,
    PointInTimeError,
    SchemaMismatch,
    UniverseError,
)
from crucible.frames import (
    SYMBOL,
    TS,
    Flavor,
    Frame,
    canonical_sort,
    flavor,
    frame_op,
    require_columns,
    restore,
    to_pandas,
    to_polars,
)

__all__ = [
    "DEFAULT_SEED",
    "SYMBOL",
    "TS",
    "CrucibleError",
    "Flavor",
    "Frame",
    "LeakageError",
    "ParityError",
    "PointInTimeError",
    "SchemaMismatch",
    "UniverseError",
    "canonical_sort",
    "flavor",
    "frame_op",
    "require_columns",
    "restore",
    "rng",
    "seed_all",
    "stable_hash",
    "to_pandas",
    "to_polars",
]
