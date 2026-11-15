"""Read-only: how often the official index is pushed, and what PM's index weight files (stsfile index-*.xr) hold.

1. Per index code and day: rows, median / p90 gap between consecutive records by exchange time and by arrival time,
   and the share of records whose last price changed -- 0430 from PM's own mp (pmmp_raw), the 224 days from dysl2_raw.
2. The stsfile index directory: the files, then one file opened as xarray (netCDF) if it is one, else its first bytes.

    python research/index_probe.py --days 20260430,20261112,20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/work/crucible_data")


def cadence(parts: list[Path], d: str) -> None:
    """Gap statistics per index code."""
    ix = pl.read_parquet(parts, columns=["symbol_id", "market_id", "time", "spider_ts", "last_price"]).unique(["symbol_id", "market_id", "time"])
    for (code, mk), g in sorted(ix.group_by("symbol_id", "market_id"), key=lambda kv: kv[0]):
        g = g.sort("time")
        g = g.filter(((pl.col("time") % (86400 * 10**9)) >= (1 * 3600 + 30 * 60) * 10**9) & ((pl.col("time") % (86400 * 10**9)) <= 7 * 3600 * 10**9))  # 09:30-15:00 CST in UTC
        if g.height < 10:
            continue
        ex = np.diff(g["time"].to_numpy()) / 1e9
        ar = np.diff(np.sort(g["spider_ts"].to_numpy())) / 1e9
        ch = float((g["last_price"].diff().fill_null(0) != 0).mean())
        print(f"  {d} {mk} {code:>6}: {g.height:>6} rows 09:30-15:00; exch gap median {np.median(ex):.2f}s p90 {np.quantile(ex, .9):.2f}s; "
              f"arrival gap median {np.median(ar):.2f}s p90 {np.quantile(ar, .9):.2f}s; last changed {ch:.2f}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", required=True)
    ap.add_argument("--sts", type=Path, default=DATA / "pm_stsfile" / "index")
    a = ap.parse_args()
    print("1 official index cadence")
    for d in a.days.split(","):
        for src in ("pmmp_raw", "dysl2_raw", "feitu_raw"):
            parts = list((DATA / src).glob(f"kind=index/date={d}/part-*.parquet"))
            if parts:
                print(f" {d} from {src}:")
                cadence(parts, d)
    print(f"2 {a.sts}")
    files = sorted(a.sts.glob("*")) if a.sts.exists() else []
    print(f"  {len(files)} files: {[f.name for f in files[:5]]} ... {[f.name for f in files[-5:]]}")
    pick = next((f for f in reversed(files) if f.is_file()), None)
    if pick is None:
        return
    print(f"  {pick} {pick.stat().st_size:,} bytes; first bytes {pick.read_bytes()[:64]!r}")
    try:
        import xarray as xr

        ds = xr.open_dataset(pick)
        print(f"  xarray: {ds}")
        for v in list(ds.data_vars)[:8]:
            print(f"   {v}: {ds[v].dims} {ds[v].dtype} e.g. {ds[v].values.ravel()[:6]}")
    except Exception as e:  # noqa: BLE001 -- a probe: say what it is not
        print(f"  not xarray-readable: {e!r}"[:300])


if __name__ == "__main__":
    main()
