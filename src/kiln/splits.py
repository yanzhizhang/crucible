"""Purged walk-forward cross-validation.

Random K-fold on financial panel data is not a weaker validation scheme, it is
a broken one, and it fails in two compounding ways.

**Cross-sectional leakage.** All 5000 names at time ``t`` are one sample, not
5000. Splitting them across folds puts near-identical market states on both
sides of the partition, and the model scores its own training data.

**Temporal leakage.** A label at ``t`` observes returns through ``t + h``. A
training row at ``t`` therefore overlaps any test row within ``h`` slots of it,
even when their timestamps differ. Purging removes exactly those rows.

**Embargo.** Serial correlation extends the contamination a little past the
test window even without label overlap, so a further gap is dropped after each
test fold.

Splits here are over **unique timestamps**, then expanded to row indices. That
is what enforces the "one timestamp is one group" rule structurally rather than
by convention.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from crucible.errors import LeakageError

__all__ = ["PurgedWalkForward", "CombinatorialPurgedCV", "assert_no_leakage"]


def _stamp_codes(ts: Sequence[object] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map timestamps to dense ordinal codes.

    Returns ``(codes, uniques)`` where ``codes[i]`` is the ordinal position of
    row ``i``'s timestamp. Working in ordinals makes purge and embargo
    arithmetic exact regardless of grain -- sessions, minutes or 3-second slots.
    """
    arr = np.asarray(ts)
    uniques, codes = np.unique(arr, return_inverse=True)
    return codes.astype(np.int64), uniques


@dataclass(frozen=True)
class PurgedWalkForward:
    """Walk-forward splits with label purging and an embargo.

    Parameters
    ----------
    train_span:
        Training window length, in **unique timestamps**.
    test_span:
        Test window length, in unique timestamps.
    embargo:
        Timestamps dropped from training immediately after each test window.
    label_horizon:
        How many slots forward the label looks. Training rows whose label
        window reaches into the test window are purged. Set this to the ``n``
        used in :func:`horizon.forward_return` **plus its entry lag**; leaving
        it at 1 for a 10-day label silently readmits nine days of overlap.
    step:
        Slots advanced between folds. Defaults to ``test_span`` (non-overlapping
        test windows).
    expanding:
        When True, training starts at the beginning of the sample each fold
        instead of sliding. Use it when the process is stable and more data
        helps; keep it False when regime change makes old data misleading.

    Notes
    -----
    Deterministic: no shuffling, no randomness anywhere. Two runs on the same
    timestamps yield identical folds.
    """

    train_span: int
    test_span: int
    embargo: int = 0
    label_horizon: int = 1
    step: int | None = None
    expanding: bool = False

    def __post_init__(self) -> None:
        if self.train_span < 1:
            raise ValueError(f"train_span must be >= 1, got {self.train_span}")
        if self.test_span < 1:
            raise ValueError(f"test_span must be >= 1, got {self.test_span}")
        if self.embargo < 0:
            raise ValueError(f"embargo must be >= 0, got {self.embargo}")
        if self.label_horizon < 1:
            raise ValueError(f"label_horizon must be >= 1, got {self.label_horizon}")

    def n_splits(self, ts: Sequence[object] | np.ndarray) -> int:
        """Number of folds this configuration yields on ``ts``."""
        return sum(1 for _ in self.split(ts))

    def split(
        self, ts: Sequence[object] | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` row-index arrays.

        Every row sharing a timestamp lands in the same side of the split, so a
        cross-section is never divided.

        Raises
        ------
        ValueError
            If the sample is too short to produce even one fold -- an empty
            iterator would otherwise read as "no leakage found".
        """
        codes, uniques = _stamp_codes(ts)
        n = len(uniques)
        step = self.step or self.test_span

        if n < self.train_span + self.test_span:
            raise ValueError(
                f"sample has {n} unique timestamps but train_span + test_span = "
                f"{self.train_span + self.test_span}. Shorten the spans or lengthen "
                f"the sample; an empty split iterator looks like a clean validation."
            )

        produced = 0
        start = 0
        while start + self.train_span + self.test_span <= n:
            tr_lo = 0 if self.expanding else start
            tr_hi = start + self.train_span  # exclusive
            te_lo = tr_hi
            te_hi = min(te_lo + self.test_span, n)

            # Purge: a training row at u labels through u + label_horizon, so
            # anything within label_horizon of the test window overlaps it.
            purge_lo = te_lo - self.label_horizon
            # Embargo: drop a further gap after the test window.
            emb_hi = te_hi + self.embargo

            train_mask = (
                (codes >= tr_lo) & (codes < tr_hi) & (codes < purge_lo)
            ) | ((codes >= emb_hi) & (codes < tr_hi))
            test_mask = (codes >= te_lo) & (codes < te_hi)

            train_idx = np.flatnonzero(train_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx
                produced += 1

            start += step

        if produced == 0:
            raise ValueError(
                "no usable folds: purging removed every training row. Reduce "
                "label_horizon or embargo, or lengthen train_span."
            )


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """Combinatorial purged cross-validation for short samples.

    Splits the timeline into ``n_groups`` contiguous blocks and tests on every
    combination of ``n_test_groups`` of them, purging and embargoing around each
    selected block. This yields many more train/test paths than walk-forward
    from the same data, which matters when you have two years of history and
    walk-forward gives you four folds.

    The cost is that test blocks are reused across combinations, so fold scores
    are not independent. Treat the spread across paths as a robustness check,
    not as an ``n``-fold confidence interval.
    """

    n_groups: int = 6
    n_test_groups: int = 2
    embargo: int = 0
    label_horizon: int = 1

    def __post_init__(self) -> None:
        if self.n_groups < 2:
            raise ValueError(f"n_groups must be >= 2, got {self.n_groups}")
        if not 1 <= self.n_test_groups < self.n_groups:
            raise ValueError(
                f"n_test_groups must be in [1, {self.n_groups - 1}], got {self.n_test_groups}"
            )

    def n_splits(self) -> int:
        """Number of train/test combinations."""
        from math import comb

        return comb(self.n_groups, self.n_test_groups)

    def split(
        self, ts: Sequence[object] | np.ndarray
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_idx, test_idx)`` for every group combination."""
        codes, uniques = _stamp_codes(ts)
        n = len(uniques)
        if n < self.n_groups:
            raise ValueError(f"{n} timestamps cannot fill {self.n_groups} groups")

        edges = np.linspace(0, n, self.n_groups + 1).astype(int)
        blocks = [(int(edges[i]), int(edges[i + 1])) for i in range(self.n_groups)]

        for pick in combinations(range(self.n_groups), self.n_test_groups):
            test_mask = np.zeros(codes.shape, dtype=bool)
            drop_mask = np.zeros(codes.shape, dtype=bool)
            for g in pick:
                lo, hi = blocks[g]
                test_mask |= (codes >= lo) & (codes < hi)
                drop_mask |= (codes >= lo - self.label_horizon) & (codes < hi + self.embargo)

            train_idx = np.flatnonzero(~drop_mask)
            test_idx = np.flatnonzero(test_mask)
            if train_idx.size and test_idx.size:
                yield train_idx, test_idx


def assert_no_leakage(
    ts: Sequence[object] | np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    label_horizon: int = 1,
    embargo: int = 0,
) -> None:
    """Assert a split is free of temporal and cross-sectional contamination.

    Checks three things:

    1. **No shared timestamp.** A cross-section appearing on both sides means
       the grouping was broken.
    2. **No label overlap.** No training timestamp lies within ``label_horizon``
       before the test window.
    3. **Embargo respected.** No training timestamp lies within ``embargo``
       after the test window.

    Raises
    ------
    LeakageError
        Naming the specific violation and how many timestamps were involved.

    Notes
    -----
    Call this inside the fold loop, not once at the end. A split generator that
    is correct for fold 0 and wrong for fold 7 is the normal failure mode.
    """
    codes, _ = _stamp_codes(ts)
    tr, te = np.unique(codes[train_idx]), np.unique(codes[test_idx])
    if tr.size == 0 or te.size == 0:
        raise LeakageError("split has an empty side; nothing was validated")

    shared = np.intersect1d(tr, te)
    if shared.size:
        raise LeakageError(
            f"{shared.size} timestamp(s) appear in both train and test. All names at "
            f"one timestamp are a single sample and must not be split across folds."
        )

    te_lo, te_hi = int(te.min()), int(te.max())

    overlap = tr[(tr >= te_lo - label_horizon) & (tr < te_lo)]
    if overlap.size:
        raise LeakageError(
            f"{overlap.size} training timestamp(s) sit within label_horizon="
            f"{label_horizon} of the test window start. Their label windows observe "
            f"returns inside the test period, so the model is trained on its own "
            f"evaluation data. Increase the purge."
        )

    if embargo > 0:
        breach = tr[(tr > te_hi) & (tr <= te_hi + embargo)]
        if breach.size:
            raise LeakageError(
                f"{breach.size} training timestamp(s) fall inside the embargo of "
                f"{embargo} slots after the test window."
            )
