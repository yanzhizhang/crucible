r"""What the D dimension of a samplerS file holds: which days a file carries, and whether a D slice is frozen across files.

For each family and export day: the distinct D values with their row counts and finite shares, and, against the previous
export, the share of cells where the SAME D slice is equal in both files (frozen history) or differs (re-estimated).

    python research/pm_factors/pm_slices.py --dates 20260923,20260924,20260929 --exports data/pm_export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import pm_frame


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    a = ap.parse_args()
    dates = sorted(x for x in a.dates.split(",") if x)
    names = sorted({f.stem.rsplit("_", 1)[0] for f in a.exports.glob("*.parquet")})
    rows, frozen = [], []
    for name in names:
        prev = None
        for d in dates:
            f = a.exports / f"{name}_{d}.parquet"
            if not f.exists():
                continue
            pm, vals, kind = pm_frame(f, None, None, None, None, None)
            if "date" not in pm.columns:
                rows.append({"file": name, "day": d, "D": "(no D dim)", "rows": pm.height, "finite": float(pm[vals[0]].is_finite().mean())})
                continue
            per = pm.group_by("date").agg(pl.len().alias("rows"), pl.col(vals[0]).is_finite().mean().alias("finite")).sort("date")
            ds = per["date"].to_list()
            rows.append({"file": name, "day": d, "D": f"{len(ds)} slices {ds[0]}..{ds[-1]}" + (" (today in)" if int(d) in [int(x) for x in ds] else " (today NOT in)"),
                         "rows": pm.height, "finite": float(pm[vals[0]].is_finite().mean())})
            if prev is not None:
                j = (pm.select("code", "date", "minute", "V", pl.col(vals[0]).alias("x"))
                     .join(prev.select("code", "date", "minute", "V", pl.col(vals[0]).alias("xp")), on=["code", "date", "minute", "V"])
                     .filter(pl.col("x").is_finite() & pl.col("xp").is_finite()))
                if j.height:
                    eq = j.group_by("date").agg(pl.len().alias("cells"), ((pl.col("x") - pl.col("xp")).abs() <= 1e-9 * pl.col("x").abs()).mean().alias("equal")).sort("date")
                    for r in eq.iter_rows(named=True):
                        frozen.append({"file": name, "day": d, "D": r["date"], "cells": r["cells"], "equal_to_prev_file": r["equal"]})
            prev = pm
    with pl.Config(tbl_rows=300, tbl_width_chars=160, float_precision=3, fmt_str_lengths=60):
        print("== D slices per file")
        print(pl.DataFrame(rows))
        if frozen:
            print("\n== the same D slice in consecutive files: share of cells equal (1 = history frozen, 0 = re-estimated every day)")
            print(pl.DataFrame(frozen))


if __name__ == "__main__":
    main()
