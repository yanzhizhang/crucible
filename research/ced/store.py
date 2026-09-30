"""CED storage: one hive-partitioned Parquet table per dataset (replaces the per-day ``.xr`` files).

Layout (root = ``$CRUCIBLE_DATA``, default ``/work/crucible_data``)::

    store/ced/<dataset>/date=YYYYMMDD/part.parquet     one trading day, one row per symbol
    store/ced/<dataset>/_static/<name>.parquet         undated tables (industry / concept catalogs)
    store/ced/calendar/exchange=<EX>.parquet           trading calendar cache

Rows are keyed by ``symbol`` (Wind code, e.g. ``600000.SH``); the ``date`` comes from the
partition, so a whole history is one scan: ``read("daily_sod", "20250101", "20251231")``.
What used to be a 2-D ``xr`` variable (ST types, dividend fields, index weights, industry
levels) is either plain columns or a long table -- see each dataset module. Integer columns
use real nulls instead of the ``-1`` sentinels the ``.xr`` files needed for C++.

``write`` is atomic (temp file + rename) and, like CED, warns on an empty panel unless the
dataset is one where "nothing today" is normal (dividends).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import polars as pl

DATA = Path(os.environ.get("CRUCIBLE_DATA", "/work/crucible_data"))
STORE = DATA / "store" / "ced"
log = logging.getLogger(__name__)


def path(dataset: str, date: str) -> Path:
    """Partition file of one trading day."""
    return STORE / dataset / f"date={date}" / "part.parquet"


def static_path(dataset: str, name: str) -> Path:
    """Undated table of a dataset (catalogs)."""
    return STORE / dataset / "_static" / f"{name}.parquet"


def _atomic(df: pd.DataFrame | pl.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp.{os.getpid()}")
    try:
        if isinstance(df, pl.DataFrame):
            df.write_parquet(tmp, compression="zstd")
        else:
            df.to_parquet(tmp, index=False, compression="zstd")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


def write(df: pd.DataFrame | pl.DataFrame, dataset: str, date: str, *, overwrite: bool = True,
          empty_ok: bool = False) -> Path:
    """Write one day of ``dataset``; skip (and say so) when it exists and ``overwrite`` is off."""
    p = path(dataset, date)
    if p.exists() and not overwrite:
        log.info("exists, skipped: %s", p)
        return p
    n = df.height if isinstance(df, pl.DataFrame) else len(df)
    if n == 0:
        (log.info if empty_ok else log.warning)(
            "%s %s: empty panel%s", dataset, date, "" if empty_ok else " -- upstream probably has no data")
    _atomic(df, p)
    log.info("wrote %s (%d rows)", p, n)
    return p


def write_static(df: pd.DataFrame | pl.DataFrame, dataset: str, name: str) -> Path:
    """Write an undated table (always overwritten)."""
    p = static_path(dataset, name)
    _atomic(df, p)
    log.info("wrote %s", p)
    return p


def exists(dataset: str, date: str) -> bool:
    return path(dataset, date).exists()


def dates(dataset: str) -> list[str]:
    """Trading days already written for ``dataset``."""
    root = STORE / dataset
    return sorted(p.name[5:] for p in root.glob("date=*") if (p / "part.parquet").exists())


def read(dataset: str, start: str | None = None, end: str | None = None) -> pl.LazyFrame:
    """All written days of ``dataset`` in ``[start, end]`` as one lazy frame with a ``date`` column."""
    lf = pl.scan_parquet(STORE / dataset / "date=*" / "part.parquet", hive_partitioning=True)
    lf = lf.with_columns(pl.col("date").cast(pl.String))
    if start:
        lf = lf.filter(pl.col("date") >= start)
    if end:
        lf = lf.filter(pl.col("date") <= end)
    return lf


def read_day(dataset: str, date: str) -> pd.DataFrame:
    """One day as pandas (what the builders and checks work with)."""
    return pd.read_parquet(path(dataset, date))
