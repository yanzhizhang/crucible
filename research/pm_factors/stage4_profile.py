r"""Stage 4: what PM's sampler values look like, per file and column, before guessing formulas.

Per export (``data/pm_export/<FAMILY>_<date>.parquet``) and ``V``: slots (and the first / last slot label), share of
nulls, share of integers, min / median / max, and the medians over stocks of the per-stock mean and standard deviation
-- a column with per-stock mean ~0 and sd ~1 is z-scored per stock, one with values in [0, 1] a share or a rank, one
with only integers a count.

    python research/pm_factors/stage4_profile.py --exports data/pm_export --date 20261009
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(a.exports.glob(f"*_{a.date}.parquet")):
        pm = pl.read_parquet(f)
        cols = {c.upper(): c for c in pm.columns}
        s, i, v = cols.get("S"), cols.get("I"), cols.get("V")
        val = next((c for c in pm.columns if c not in (s, i, v, cols.get("D")) and pm.schema[c].is_numeric()), None)
        if v is None or val is None:
            print(f"{f.name}: no V / value column among {pm.columns}")
            continue
        slots = pm[i].unique().sort() if i else pl.Series([None])
        x = pl.col(val)
        per = (pm.group_by(v, s).agg(x.mean().alias("m"), x.std().alias("sd"))
               .group_by(v).agg(pl.col("m").median().alias("stock_mean_med"), pl.col("sd").median().alias("stock_sd_med")))
        t = (pm.group_by(v).agg(x.null_count().truediv(pl.len()).alias("null"), ((x - x.round(0)).abs() < 1e-9).mean().alias("int"),
                                x.min().alias("min"), x.median().alias("med"), x.max().alias("max"), (x == 0).mean().alias("zero"))
             .join(per, on=v).with_columns(pl.lit(f.stem.rsplit("_", 1)[0]).alias("file"), pl.lit(slots.len()).alias("slots"),
                                           pl.lit(str(slots[0])).alias("first_slot"), pl.lit(str(slots[-1])).alias("last_slot"))
             .rename({v: "V"}))
        rows.append(t)
    if not rows:
        print("no exports")
        return
    out = pl.concat(rows).select("file", "V", "slots", "first_slot", "last_slot", "null", "int", "zero", "min", "med", "max",
                                 "stock_mean_med", "stock_sd_med").sort("file", "V")
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(f"== PM sampler columns on {a.date}: slots, null / integer / zero shares, range, per-stock mean and sd (medians)")
        print(out)
    out.write_csv(a.exports.parent / "reports" / "stage4" / f"profile_{a.date}.csv")


if __name__ == "__main__":
    main()
