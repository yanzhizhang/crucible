"""The polars/pandas boundary.

crucible is polars-native: every loader, alignment step, feature transform and
label lives in polars, because tick research means long frames with millions of
rows and the Arrow-backed columnar path is the only one that stays cheap there.

pandas is not second-class, it is *downstream*. ``statsmodels``, ``lightgbm``
and ``matplotlib`` all speak pandas/numpy, so :mod:`kiln`, :mod:`ballast` and
:mod:`herald` convert once at their own edge rather than threading two frame
types through the whole stack.

Public functions therefore accept either flavor and hand back the flavor they
were given. The rule for callers is simple: pass whatever you have, get back
what you passed.

Canonical layout is long format, sorted by ``(ts, symbol)``:

    ts      datetime[ns]  -- the observation timestamp, tz-naive exchange local
    symbol  str           -- 6-digit A-share code, or futures contract code
    ...     values

Wide format is a presentation detail. It is produced at the last moment inside
:mod:`assay`/:mod:`herald` and never persisted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal, ParamSpec, TypeAlias, TypeVar

import pandas as pd
import polars as pl

__all__ = [
    "TS",
    "SYMBOL",
    "Frame",
    "Flavor",
    "flavor",
    "to_polars",
    "to_pandas",
    "restore",
    "require_columns",
    "canonical_sort",
    "frame_op",
]

TS = "ts"
"""Canonical timestamp column name."""

SYMBOL = "symbol"
"""Canonical instrument column name."""

Frame: TypeAlias = pl.DataFrame | pd.DataFrame
Flavor: TypeAlias = Literal["polars", "pandas"]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def flavor(df: Frame) -> Flavor:
    """Report which library ``df`` came from."""
    if isinstance(df, pl.DataFrame):
        return "polars"
    if isinstance(df, pd.DataFrame):
        return "pandas"
    raise TypeError(f"expected a polars or pandas DataFrame, got {type(df).__name__}")


def to_polars(df: Frame) -> pl.DataFrame:
    """Coerce to polars.

    A pandas index is *not* metadata we can afford to lose silently: a
    ``(ts, symbol)`` MultiIndex carries the join keys. It is reset into real
    columns rather than dropped.
    """
    if isinstance(df, pl.DataFrame):
        return df
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"expected a polars or pandas DataFrame, got {type(df).__name__}")
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    return pl.from_pandas(df)


def to_pandas(df: Frame) -> pd.DataFrame:
    """Coerce to pandas, keeping a flat RangeIndex.

    Callers that want ``(ts, symbol)`` indexed frames set that up themselves;
    doing it here would make the round trip through :func:`restore` lossy.
    """
    if isinstance(df, pd.DataFrame):
        return df
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"expected a polars or pandas DataFrame, got {type(df).__name__}")
    return df.to_pandas()


def restore(df: pl.DataFrame, to: Flavor) -> Frame:
    """Convert a polars result back to the caller's flavor."""
    return df if to == "polars" else df.to_pandas()


def require_columns(df: pl.DataFrame, columns: Iterable[str], *, where: str) -> None:
    """Raise if any of ``columns`` is missing.

    ``where`` names the calling function so the message points at the real
    problem rather than at this helper.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{where}: missing required column(s) {missing}; have {df.columns}")


def canonical_sort(df: pl.DataFrame, *, by: Sequence[str] = (TS, SYMBOL)) -> pl.DataFrame:
    """Sort into canonical order, ignoring keys the frame does not carry.

    Determinism depends on this: two runs that differ only in row order produce
    different floating-point sums in any groupby aggregation, which breaks the
    byte-identical-output invariant.
    """
    keys = [c for c in by if c in df.columns]
    return df.sort(keys, maintain_order=True) if keys else df


def frame_op(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    """Mark a pure ``(DataFrame, **params) -> DataFrame`` transform.

    The wrapper coerces the first positional argument to polars and restores
    the caller's flavor on the way out, so a transform body only ever handles
    one frame type. Used by the :mod:`forge` registry.
    """

    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if not args:
            return fn(*args, **kwargs)
        first, *rest = args
        if not isinstance(first, (pl.DataFrame, pd.DataFrame)):
            return fn(*args, **kwargs)
        want = flavor(first)
        out = fn(to_polars(first), *rest, **kwargs)  # type: ignore[arg-type]
        if isinstance(out, pl.DataFrame) and want == "pandas":
            return out.to_pandas()  # type: ignore[return-value]
        return out

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
    return wrapper
