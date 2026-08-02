"""Determinism helpers: same inputs must give byte-identical outputs.

Two things break this in practice and both are silent:

* ``hash()`` on ``str``/``bytes`` is salted per process (PYTHONHASHSEED), so
  any fingerprint built from it changes between runs. Use :func:`stable_hash`.
* Library RNGs seeded from entropy. Use :func:`seed_all` at every entry point
  that touches a model, a resample, or a randomized search.

There is deliberately no ``now()`` helper anywhere in crucible. Wall-clock in a
computation makes a backtest unreproducible; timestamps come from the data.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

__all__ = ["DEFAULT_SEED", "seed_all", "stable_hash", "rng"]

DEFAULT_SEED = 20240101
"""Fixed project-wide seed. Chosen once; never varied to 'see if it holds'."""


def seed_all(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG crucible can reach.

    Call at the top of any script or fit routine. Seeding numpy alone is not
    enough: LightGBM's feature subsampling reads its own seed, and any pure
    Python shuffling reads :mod:`random`.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng(seed: int = DEFAULT_SEED) -> np.random.Generator:
    """A fresh, explicitly seeded Generator.

    Preferred over the legacy global ``np.random`` functions: passing a
    Generator makes the randomness a visible argument instead of ambient state.
    """
    return np.random.default_rng(seed)


def stable_hash(*parts: Any) -> str:
    """A process-independent hex digest of ``parts``.

    Values are rendered with ``repr`` and joined with a separator that cannot
    appear in an identifier, so ``("ab", "c")`` and ``("a", "bc")`` differ.
    """
    payload = "\x1f".join(repr(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
