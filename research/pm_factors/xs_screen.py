r"""Which still-open model columns look cross-sectional (a rank / z-score / demeaned value) from PM's training ranges alone.

fingerprint.parquet (pm_models_probe) holds each column's training [lo, hi] and the quantiles of its split thresholds. A rank
in [0, 1] / [-1, 1] / [-0.5, 0.5], or a z-score (symmetric, a few units wide, median ~0), shows in those numbers; a raw
price-like or count-like quantity does not. Prints the open columns of an assignment with a guess and their gain.

    python research/pm_factors/xs_screen.py --assign data/reports/r27_both.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def guess(lo: float, hi: float, med: float) -> str:
    if lo is None or hi is None or hi <= lo:
        return "-"
    w = hi - lo
    if abs(lo) < 0.02 and abs(hi - 1) < 0.02:
        return "rank01"
    if abs(lo + 1) < 0.02 and abs(hi - 1) < 0.02:
        return "rank_pm1"
    if abs(lo + 0.5) < 0.02 and abs(hi - 0.5) < 0.02:
        return "crank"
    if lo < 0 < hi and 0.4 < abs(lo) / hi < 2.5 and 2 <= max(abs(lo), hi) <= 15 and med is not None and abs(med) < 0.1 * w:
        return "z-like"
    if lo < 0 < hi and med is not None and abs(med) < 0.02 * w and 0.3 < abs(lo) / hi < 3:
        return "demeaned?"
    return "raw"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--assign", type=Path, default=Path("data/reports/r27_both.json"))
    a = ap.parse_args()
    fp = pl.read_parquet(a.fingerprint).sort("gain_share", descending=True).unique("feature", maintain_order=True)
    assign = json.loads(a.assign.read_text()) if a.assign.exists() else {}
    rows = []
    for r in fp.iter_rows(named=True):
        rows.append({"feature": r["feature"], "gain": r["gain_share"], "assigned": assign.get(r["feature"], "-"), "lo": r.get("lo"), "hi": r.get("hi"),
                     "q_med": r.get("q_med"), "kind": r.get("kind"), "guess": guess(r.get("lo"), r.get("hi"), r.get("q_med"))})
    t = pl.DataFrame(rows, infer_schema_length=None)
    with pl.Config(tbl_rows=-1, tbl_width_chars=220, fmt_str_lengths=60, float_precision=4):
        print("== guesses over all columns")
        print(t.group_by("guess").agg(pl.len().alias("cols"), pl.col("gain").sum(), (pl.col("assigned") != "-").sum().alias("assigned")).sort("gain", descending=True))
        print("\n== assigned columns whose range looks cross-sectional, with their formula (does the formula agree?)")
        print(t.filter((pl.col("assigned") != "-") & (pl.col("guess") != "raw") & (pl.col("guess") != "-")).select("feature", "guess", "lo", "hi", "q_med", "gain", "assigned"))
        print("\n== OPEN columns whose range looks cross-sectional (scan these with z / rank / dm / dmi / neu first)")
        op = t.filter((pl.col("assigned") == "-") & (pl.col("guess") != "raw") & (pl.col("guess") != "-"))
        print(op.select("feature", "guess", "lo", "hi", "q_med", "kind", "gain"))
    print("XS_OPEN=" + ",".join(op["feature"].to_list()))


if __name__ == "__main__":
    main()
