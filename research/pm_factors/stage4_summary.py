r"""Stage 4: one table over every rank_candidates output (data/reports/stage4/<FAMILY>_<date>.parquet).

Per (family, PM value column, V): the candidate variant with the best mean ``exact_scaled`` over the days,
its mean rank correlation and the number of days; then per family the mean over its columns -- a family
whose columns all reach ~1.0 is reproduced up to a unit, one at ~0 needs other candidates.

    python research/pm_factors/stage4_summary.py --reports data/reports/stage4
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
    parts = []
    for f in sorted(a.reports.glob("*.parquet")):
        fam, _, day = f.stem.rpartition("_")
        parts.append(pl.read_parquet(f).with_columns(pl.lit(fam).alias("family"), pl.lit(day).alias("day")))
    if not parts:
        print(f"no rank_candidates output under {a.reports}")
        return
    t = pl.concat(parts, how="diagonal_relaxed")
    best = (t.group_by("family", "value_col", "V", "cand", "VV").agg(
                pl.len().alias("days"), pl.col("exact_scaled").mean(), pl.col("exact").mean(), pl.col("spearman").mean(),
                pl.col("within").mean(), pl.col("ratio").median())
            .sort("family", "value_col", "V", "exact_scaled", "spearman", descending=[False, False, False, True, True])
            .group_by("family", "value_col", "V", maintain_order=True).first())
    fam = (best.group_by("family", "value_col").agg(pl.len().alias("columns"), pl.col("exact_scaled").mean(), pl.col("spearman").mean(),
                                                   pl.col("days").max())
           .sort("exact_scaled", descending=True))
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print("== families: mean over PM columns of the best candidate (exact_scaled 1.0 = reproduced up to a unit)")
        print(fam)
        print("\n== per PM column: best candidate variant")
        print(best.select("family", "value_col", "V", "cand", "VV", "days", "exact_scaled", "exact", "ratio", "spearman", "within"))
    best.write_csv(a.reports / "summary.csv")


if __name__ == "__main__":
    main()
