r"""One line per PM column: the best implied-input match and the best formula, from the A2 csv files.

    python research/pm_factors/stage4_brief.py --reports data/reports/stage4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reports", type=Path, default=Path("data/reports/stage4"))
    a = ap.parse_args()
    im = a.reports / "implied_match.csv"
    if not im.exists():
        print("no implied_match.csv")
        return
    t = pl.read_csv(im)
    best = (t.sort("exact_scaled", "spearman", descending=[True, True], nulls_last=True)
            .group_by("file", "V", maintain_order=True).head(1)
            .select("file", "V", "same", "basic", "a", "ours_day", "shift", "exact_scaled", "ratio", "spearman"))
    fm = a.reports / "implied_formulas.csv"
    if fm.exists():
        f = pl.read_csv(fm).sort("ok3", descending=True).group_by("file", "V", maintain_order=True).head(1)
        best = best.join(f.select("file", "V", pl.col("formula").alias("best_formula"), pl.col("ok3").alias("f_ok3"), pl.col("a").alias("f_a")),
                         on=["file", "V"], how="left")
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=230, float_precision=3, fmt_str_lengths=60):
        print("== A2 brief: per PM column the share of cells carried unchanged from the previous day (same), the best basic for the "
              "implied input y = pm_d - a pm_prev on the changed cells, and the best formula from the structure search")
        print(best.sort("file", "V"))


if __name__ == "__main__":
    main()
