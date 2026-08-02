"""Shared setup for the example scripts.

Generates (or reuses) one month of synthetic A-share data under
``examples/_out/store`` so every example is runnable end to end with no
external data. Swap :func:`open_store` for a real ``quarry.open_db(root)`` and
the rest of each script is unchanged -- that is the point of keeping loaders
schema-driven.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from crucible.determinism import seed_all
from quarry.db import open_db
from quarry.synth import SynthTruth, make_store

OUT = Path(__file__).parent / "_out"
STORE = OUT / "store"


def open_store(*, n_symbols: int = 60, with_ticks: bool = False) -> tuple[object, SynthTruth]:
    """Return ``(connection, truth)`` for the example store, building it once."""
    seed_all()
    OUT.mkdir(parents=True, exist_ok=True)
    truth = make_store(
        STORE,
        n_symbols=n_symbols,
        bars_per_day=60,
        with_ticks=with_ticks,
        tick_symbols=3,
    )
    return open_db(STORE), truth


def quiet_close_to_close() -> warnings.catch_warnings:
    """Suppress the close-to-close label warning in examples that use it knowingly."""
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.simplefilter("ignore", UserWarning)
    return ctx


def rule(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")
