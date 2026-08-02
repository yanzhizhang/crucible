"""Loaders returning long-format ``(ts, symbol, ...)`` frames.

Every loader here pushes its date and symbol predicates into the SQL so DuckDB
prunes partitions before scanning. That is the difference between a query that
reads three files and one that reads the whole tree, and at tick resolution it
is the difference between working and not.

All loaders return polars. Convert at your own edge if you want pandas -- see
:mod:`crucible.frames`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import polars as pl

from crucible.frames import SYMBOL, TS
from quarry.db import stream
from quarry.schema import FactorSpec, SchemaFingerprint, validate_fingerprint

if TYPE_CHECKING:
    import duckdb

__all__ = [
    "view_columns",
    "dump_fingerprint",
    "load_factor_frame",
    "load_bars",
    "load_daily",
    "load_index",
    "load_ticks",
    "iter_ticks",
]

DateLike = dt.date | str


def _d(x: DateLike) -> dt.date:
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    s = str(x)
    return dt.date.fromisoformat(s) if "-" in s else dt.datetime.strptime(s, "%Y%m%d").date()


def _hive(x: DateLike) -> str:
    """Render a date as its hive partition value."""
    return _d(x).strftime("%Y%m%d")


def view_columns(conn: duckdb.DuckDBPyConnection, view: str) -> dict[str, str]:
    """Column name -> DuckDB type for ``view``."""
    rows = conn.execute(f"DESCRIBE {view}").fetchall()  # noqa: S608 -- view name is internal
    return {r[0]: r[1] for r in rows}


def _where(
    dates: Sequence[DateLike] | None,
    start: DateLike | None,
    end: DateLike | None,
    symbols: Sequence[str] | None,
) -> tuple[str, dict[str, Any]]:
    """Build a partition-pruning WHERE clause and its parameters.

    Predicates land on the hive columns ``date`` and ``symbol`` specifically,
    because those are the only ones DuckDB can use to skip files. Filtering on
    ``ts`` instead still works but reads every file first.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if dates is not None:
        params["dates"] = [_hive(d) for d in dates]
        clauses.append("list_contains($dates, date)")
    if start is not None:
        params["start"] = _hive(start)
        clauses.append("date >= $start")
    if end is not None:
        params["end"] = _hive(end)
        clauses.append("date <= $end")
    if symbols is not None:
        syms = list(dict.fromkeys(str(s) for s in symbols))
        if not syms:
            raise ValueError("symbols was empty; pass None to mean 'all symbols'")
        params["symbols"] = syms
        clauses.append("list_contains($symbols, symbol)")

    return ("WHERE " + " AND ".join(clauses) if clauses else ""), params


def _ts_expr(cols: dict[str, str]) -> str:
    """SQL producing the canonical ``ts`` column.

    Intraday datasets carry a real ``ts``. Daily datasets carry only the hive
    ``date``, which is promoted to a midnight timestamp so every frame in the
    stack has the same key regardless of frequency.
    """
    if TS in cols:
        return TS
    return "CAST(strptime(date, '%Y%m%d') AS TIMESTAMP) AS ts"


def dump_fingerprint(
    conn: duckdb.DuckDBPyConnection,
    view: str = "factor_frame",
    *,
    sidecar: str | None = None,
) -> SchemaFingerprint:
    """Read the fingerprint a dump declares about itself.

    Prefers a ``_fingerprint.json`` sidecar written by prism, which is the only
    source that knows factor **parameters**. Falls back to deriving names and
    dtypes from the view schema.

    The fallback cannot see parameters, so it will not catch a lookback change
    from 20 to 22 -- the columns are identical. Ship the sidecar if you want
    that class of drift caught.
    """
    if sidecar is not None:
        from pathlib import Path

        p = Path(sidecar)
        if p.is_file():
            return SchemaFingerprint.from_json(p.read_text(encoding="utf-8"))

    cols = view_columns(conn, view)
    skip = {TS, SYMBOL, "date"}
    return SchemaFingerprint(
        tuple(FactorSpec(name, {}, dtype) for name, dtype in cols.items() if name not in skip),
        producer="derived-from-schema",
    )


def load_factor_frame(
    conn: duckdb.DuckDBPyConnection,
    dates: Sequence[DateLike] | None = None,
    symbols: Sequence[str] | None = None,
    factors: Sequence[str] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    expected: SchemaFingerprint | None = None,
    sidecar: str | None = None,
    view: str = "factor_frame",
) -> pl.DataFrame:
    """Load prism factor values, refusing a mismatched producer.

    Parameters
    ----------
    dates:
        Explicit trading dates. Mutually usable with ``start``/``end``, which
        express a range; pass whichever is natural.
    symbols:
        Restrict to these instruments. ``None`` means all, which at tick
        resolution is rarely what you want.
    factors:
        Factor columns to read. ``None`` reads all of them. Naming them is
        cheaper -- Parquet is columnar, so an unread column is never scanned.
    expected:
        Fingerprint the caller requires. When supplied, the dump's own
        fingerprint must match exactly or :class:`SchemaMismatch` is raised.
        ``None`` skips the check; that is an explicit, visible opt-out.

    Returns
    -------
    Long-format ``(ts, symbol, *factors)`` sorted by ``(ts, symbol)``.

    Point-in-time contract
    ----------------------
    Returns exactly the rows prism stamped at each ``ts``. No forward fill, no
    reindexing, no interpolation -- a gap in the dump stays a gap, because
    filling it here would manufacture a value that did not exist at ``t``.
    """
    actual = dump_fingerprint(conn, view, sidecar=sidecar)
    validate_fingerprint(expected, actual, source=f"{view} dump")

    cols = view_columns(conn, view)
    if factors is not None:
        missing = [f for f in factors if f not in cols]
        if missing:
            raise KeyError(f"factor(s) {missing} not in {view}; available: {sorted(cols)}")
        picked = list(factors)
    else:
        picked = [c for c in cols if c not in {TS, SYMBOL, "date"}]

    where, params = _where(dates, start, end, symbols)
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])
    sql = f"SELECT {select} FROM {view} {where} ORDER BY ts, symbol"  # noqa: S608
    return conn.execute(sql, params).pl()


def load_bars(
    conn: duckdb.DuckDBPyConnection,
    freq: str = "1min",
    dates: Sequence[DateLike] | None = None,
    symbols: Sequence[str] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    fields: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Load OHLCV bars at ``freq`` from the ``bars_<freq>`` view.

    Bars are right-labelled: the row stamped 09:31:00 covers ``(09:30, 09:31]``.
    See :mod:`almanac.calendar` for why any other convention leaks the future.

    Point-in-time contract
    ----------------------
    A bar labelled ``t`` is fully known at ``t`` and may be read by a feature at
    ``t``. It must not be read by a feature at ``t - freq``.
    """
    view = f"bars_{freq}"
    cols = view_columns(conn, view)
    default = ["open", "high", "low", "close", "volume", "amount", "vwap"]
    picked = list(fields) if fields is not None else [c for c in default if c in cols]
    where, params = _where(dates, start, end, symbols)
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])
    sql = f"SELECT {select} FROM {view} {where} ORDER BY ts, symbol"  # noqa: S608
    return conn.execute(sql, params).pl()


def load_daily(
    conn: duckdb.DuckDBPyConnection,
    fields: Sequence[str] | None = None,
    dates: Sequence[DateLike] | None = None,
    symbols: Sequence[str] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    view: str = "daily",
) -> pl.DataFrame:
    """Load daily fields (prices, turnover, market cap, ST flags, listing info).

    Prices here are **unadjusted**. Adjustment is an explicit step via
    :func:`almanac.adjust`, because mask construction and cost modelling both
    need the traded price.
    """
    cols = view_columns(conn, view)
    picked = list(fields) if fields is not None else [c for c in cols if c not in {TS, SYMBOL, "date"}]
    missing = [f for f in picked if f not in cols]
    if missing:
        raise KeyError(f"field(s) {missing} not in {view}; available: {sorted(cols)}")
    where, params = _where(dates, start, end, symbols)
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])
    sql = f"SELECT {select} FROM {view} {where} ORDER BY ts, symbol"  # noqa: S608
    return conn.execute(sql, params).pl()


def load_index(
    conn: duckdb.DuckDBPyConnection,
    code: str,
    dates: Sequence[DateLike] | None = None,
    *,
    start: DateLike | None = None,
    end: DateLike | None = None,
    view: str = "index",
) -> pl.DataFrame:
    """Load one index's level series, e.g. ``"000300.SH"``.

    Returned in the same long shape as everything else, with ``symbol`` holding
    the index code, so an index joins against instrument frames without a
    special case.
    """
    cols = view_columns(conn, view)
    where, params = _where(dates, start, end, [code])
    picked = [c for c in cols if c not in {TS, SYMBOL, "date"}]
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])
    sql = f"SELECT {select} FROM {view} {where} ORDER BY ts"  # noqa: S608
    return conn.execute(sql, params).pl()


def load_ticks(
    conn: duckdb.DuckDBPyConnection,
    date: DateLike,
    symbols: Sequence[str] | None = None,
    *,
    fields: Sequence[str] | None = None,
    view: str = "ticks",
    allow_full_day: bool = False,
) -> pl.DataFrame:
    """Load one date of tick/snapshot data.

    Parameters
    ----------
    symbols:
        Required in practice. Passing ``None`` requests every symbol for the
        date, which for A-shares is roughly 5000 instruments at 3-second
        snapshots -- tens of millions of rows.
    allow_full_day:
        Must be set explicitly to permit ``symbols=None``. This guard exists
        because the full-day read is the single most common way to wedge a
        research box, and it should be a decision rather than an omission.

    Raises
    ------
    ValueError
        If ``symbols`` is None and ``allow_full_day`` is False.

    See Also
    --------
    iter_ticks : streaming form that never materialises the whole result.
    """
    if symbols is None and not allow_full_day:
        raise ValueError(
            "load_ticks() without `symbols` reads every instrument for the date "
            "(~5000 symbols x 4800 3s slots). Pass symbols=[...], use iter_ticks() "
            "to stream, or set allow_full_day=True if you really mean it."
        )
    cols = view_columns(conn, view)
    picked = list(fields) if fields is not None else [c for c in cols if c not in {TS, SYMBOL, "date"}]
    where, params = _where([date], None, None, symbols)
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])
    sql = f"SELECT {select} FROM {view} {where} ORDER BY ts, symbol"  # noqa: S608
    return conn.execute(sql, params).pl()


def iter_ticks(
    conn: duckdb.DuckDBPyConnection,
    dates: Sequence[DateLike],
    symbols: Sequence[str] | None = None,
    *,
    fields: Sequence[str] | None = None,
    view: str = "ticks",
    batch_rows: int = 1_000_000,
) -> Iterator[tuple[dt.date, pl.DataFrame]]:
    """Stream tick data one date at a time, in bounded batches.

    This is the intended entry point for tick research. Peak memory is one
    batch, not one day, so a multi-year sweep runs in constant space.

    Yields
    ------
    ``(date, chunk)`` pairs. A date with no data yields nothing rather than an
    empty frame, so a suspended-market day simply does not appear.
    """
    cols = view_columns(conn, view)
    picked = list(fields) if fields is not None else [c for c in cols if c not in {TS, SYMBOL, "date"}]
    select = ", ".join([_ts_expr(cols), SYMBOL, *picked])

    for day in dates:
        where, params = _where([day], None, None, symbols)
        sql = f"SELECT {select} FROM {view} {where} ORDER BY ts, symbol"  # noqa: S608
        for chunk in stream(conn, sql, params, batch_rows=batch_rows):
            if chunk.height:
                yield _d(day), chunk
