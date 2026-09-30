"""Named WindPy pulls used by the reproduction track, all through the cache.

Each function documents the Wind fields it asks for; column names are the Wind field names
lower-cased, plus ``code`` (Wind code, e.g. ``600000.SH``) and ``date`` / ``time`` where the
request is a series. Conversion to the quarry layout happens in ``research/ced``.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from wind.cache import cached_request

__all__ = [
    "daily_bars",
    "index_constituents",
    "minute_bars",
    "ticks",
    "trade_days",
]


def _lower(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.lower() for c in df.columns})


def trade_days(start: str, end: str) -> list[str]:
    """SSE trading days in ``[start, end]`` as ``YYYYMMDD`` strings."""
    df = cached_request("tdays", start, end, "")
    col = df.columns[-1]
    return [str(v)[:10].replace("-", "") for v in df[col].to_list()]


def daily_bars(
    codes: Sequence[str],
    start: str,
    end: str,
    fields: str = "open,high,low,close,volume,amt,pre_close,maxupordown,trade_status",
) -> pl.DataFrame:
    """Unadjusted daily bars, one request per code (wsd takes many fields only for one code).

    ``volume`` is in shares and ``amt`` in CNY, the exchange's own daily totals; these are
    what the quality standard's multi-source check compares against the tick stream.
    """
    out = []
    for c in codes:
        df = _lower(cached_request("wsd", c, fields, start, end, "PriceAdj=U"))
        out.append(df.rename({"index": "date"}).with_columns(pl.lit(c).alias("code")))
    return pl.concat(out, how="diagonal_relaxed") if out else pl.DataFrame()


def minute_bars(
    code: str, begin: str, end: str, fields: str = "open,high,low,close,volume,amt"
) -> pl.DataFrame:
    """1-minute bars for one code, ``begin``/``end`` as ``YYYY-MM-DD HH:MM:SS`` local time."""
    df = _lower(cached_request("wsi", code, fields, begin, end, "BarSize=1"))
    return df.rename({"index": "time"}).with_columns(pl.lit(code).alias("code"))


def ticks(
    code: str, begin: str, end: str, fields: str = "last,volume,amt,bid1,ask1,bsize1,asize1"
) -> pl.DataFrame:
    """Intraday tick snapshots for one code (Wind keeps only recent history)."""
    df = _lower(cached_request("wst", code, fields, begin, end, ""))
    return df.rename({"index": "time"}).with_columns(pl.lit(code).alias("code"))


def index_constituents(index_code: str, date: str) -> pl.DataFrame:
    """Constituents and weights of ``index_code`` (e.g. ``000300.SH``) as of ``date``."""
    return _lower(cached_request("wset", "indexconstituent", f"date={date};windcode={index_code}"))
