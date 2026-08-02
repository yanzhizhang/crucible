"""Failure modes that must never degrade into a warning.

Each invariant in the build contract gets an exception type. A research bug
that silently returns plausible numbers is worse than a crash, so every one of
these is raised, never logged.
"""

from __future__ import annotations


class CrucibleError(Exception):
    """Base for every crucible failure."""


class SchemaMismatch(CrucibleError):
    """A prism dump does not match the fingerprint the caller expects.

    Raised by :func:`quarry.validate_fingerprint`. Refusing a mismatched
    producer is the point: silently reading a renamed or reordered factor
    column yields research results attributed to the wrong factor.
    """


class PointInTimeError(CrucibleError):
    """A computation would read data that did not exist at the timestamp.

    Covers both directions: a feature reaching past ``t``, and a label window
    overlapping the feature timestamp.
    """


class LeakageError(CrucibleError):
    """A train/test split failed its leakage assertion.

    Raised by :mod:`kiln` when purge/embargo did not actually remove
    label-window overlap between train and test folds.
    """


class ParityError(CrucibleError):
    """Python PnL failed to reconcile against the C++ backtester.

    A mismatch is a build failure, not a rounding note.
    """


class UniverseError(CrucibleError):
    """A point-in-time universe query is unanswerable as asked.

    Most often: index membership without effective dates, which would force
    today's constituents onto a historical date.
    """
