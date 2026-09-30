"""Fetch public reference data through the a-stock-data skill and cache it as Parquet.

Extranet stand-ins for what the intranet gets from Wind SQL, used by the PM-reproduction
track until Wind is available:

======================  =============================================  ==================
dataset                 what                                           source
======================  =============================================  ==================
``tdx_daily``           one trading day, every SH/SZ/BJ security:      TDX official
                        prev_close, OHLC, volume (shares), amount      end-of-day package
``index_weights``       latest published constituents + weights        CSI (中证) official
``trading_calendar``    official month calendars                       SZSE official
``sw_industry``         Shenwan industry history (point-in-time)       Shenwan official
``st_list``             ST / *ST names, today's snapshot               Eastmoney
======================  =============================================  ==================

Layout: ``data/public_cache/<dataset>/...parquet``. Dated history (``tdx_daily``) is cached per
date and never refetched. "Latest" snapshots are keyed by the date the *source* stamps on the
file (not the fetch date), so re-running on the same publication is a no-op and a new
publication adds a file -- the history of snapshots accumulates point-in-time.

Usage (Windows or WSL, needs the ``public`` extra)::

    python research/astock/fetch.py --dates 20260615,20260805,20260921
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from astock import skill

CACHE = Path(__file__).resolve().parents[2] / "data" / "public_cache"
CST = dt.timezone(dt.timedelta(hours=8))


def _today_cst() -> dt.date:
    """Today in exchange local time (snapshot files are keyed by the market date)."""
    return dt.datetime.now(CST).date()


INDEXES = {"000300": "CSI 300", "000905": "CSI 500", "000852": "CSI 1000"}


def _to_polars(df: pd.DataFrame) -> pl.DataFrame:
    out = pl.from_pandas(df)
    # codes must stay 6-character strings (quarry rule: '000001' never becomes 1)
    for c in ("code",):
        if c in out.columns:
            out = out.with_columns(pl.col(c).cast(pl.String).str.zfill(6))
    return out


def _write(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return path


def _cached(path: Path, produce: Callable[[], pl.DataFrame], *, refresh: bool) -> pl.DataFrame:
    if path.exists() and not refresh:
        return pl.read_parquet(path)
    df = produce()
    _write(df, path)
    return df


def tdx_daily(date: str, *, refresh: bool = False) -> pl.DataFrame:
    """One day of every SH/SZ/BJ security from the TDX official end-of-day package."""
    ymd = date.replace("-", "")
    path = CACHE / "tdx_daily" / f"date={ymd}" / "part.parquet"
    return _cached(path, lambda: _to_polars(skill.tdx_daily_package(ymd)), refresh=refresh)


def index_weights(index_code: str) -> pl.DataFrame:
    """Latest published CSI constituents with weights, stored under the source's own date."""
    df = _to_polars(skill.index_weights(index_code, provider="csi"))
    asof = str(df["date"][0]).replace("-", "")
    _write(df, CACHE / "index_weights" / f"index={index_code}" / f"asof={asof}.parquet")
    return df


def trading_calendar(year: int) -> pl.DataFrame:
    """Every published month of ``year`` (unpublished future months are skipped, loudly)."""
    parts = []
    for m in range(1, 13):
        try:
            parts.append(_to_polars(skill.trading_calendar(year, m)))
        except RuntimeError as exc:
            print(f"  calendar {year}-{m:02d}: not published ({exc})")
    df = pl.concat(parts)
    _write(df, CACHE / "trading_calendar" / f"year={year}.parquet")
    return df


def sw_industry() -> pl.DataFrame:
    """Shenwan industry history; each row is a (code, industry, effective-from) change."""
    df = _to_polars(skill.sw_industry_history())
    _write(df, CACHE / "sw_industry" / f"fetched={_today_cst():%Y%m%d}.parquet")
    return df


def st_list() -> pl.DataFrame:
    """Today's ST / *ST snapshot."""
    df = _to_polars(skill.st_stock_list())
    _write(df, CACHE / "st_list" / f"date={_today_cst():%Y%m%d}.parquet")
    return df


def main() -> None:
    """CLI entry point: fetch everything the reproduction track needs right now."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="20260615,20260805,20260921")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--only", default="", help="comma list of datasets to run (default all)")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    run = lambda name: not only or name in only  # noqa: E731
    print(f"a-stock-data skill {skill.__skill_version__}; cache {CACHE}")
    failures = []

    def attempt(name: str, fn: Callable[[], pl.DataFrame]) -> None:
        try:
            df = fn()
            print(f"[ok]   {name}: {df.height:,} rows, cols {df.columns[:8]}")
        except Exception as exc:
            failures.append(name)
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    if run("tdx_daily"):
        for d in filter(None, a.dates.split(",")):
            attempt(f"tdx_daily {d}", lambda d=d: tdx_daily(d))
    if run("index_weights"):
        for code, name in INDEXES.items():
            attempt(f"index_weights {code} {name}", lambda c=code: index_weights(c))
    if run("trading_calendar"):
        attempt(f"trading_calendar {a.year}", lambda: trading_calendar(a.year))
    if run("sw_industry"):
        attempt("sw_industry", sw_industry)
    if run("st_list"):
        attempt("st_list", st_list)
    if failures:
        raise SystemExit(f"{len(failures)} source(s) failed: {failures}")


if __name__ == "__main__":
    main()
