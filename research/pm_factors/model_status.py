r"""Model line, the worklist: which features carry the model's gain, how they moved at the refit, and which have a formula.

Reads the probe's ``importance.parquet`` (``pm_models_probe.py`` over every month) and ``ic_score``'s ``ic_assign.json``:

* refit diff: features added / dropped between the last two model months, and the biggest gain-share moves;
* worklist: every family of the latest month by gain share, with how many of its features have an assigned formula,
  cumulative gain, and the gain share still without a formula (the order to work in).

    python research/pm_factors/model_status.py --stage5 data/reports/stage5 --assign data/reports/stage6/ic_assign.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage5", type=Path, default=Path("data/reports/stage5"))
    ap.add_argument("--assign", type=Path, default=Path("data/reports/stage6/ic_assign.json"))
    ap.add_argument("--set", default="tk5m", help="model set (first path part under models/)")
    a = ap.parse_args()
    imp = pl.read_parquet(a.stage5 / "importance.parquet")
    if "set" in imp.columns:
        imp = imp.filter(pl.col("set") == a.set) if a.set in imp["set"].unique().to_list() else imp
    months = sorted(imp["month"].unique().to_list())
    per = imp.group_by("month", "feature", "family").agg(pl.col("gain_share").mean())
    assign = json.loads(a.assign.read_text()) if a.assign.exists() else {}
    print(f"months {months}; {len(assign)} features with an assigned formula ({a.assign})")
    with pl.Config(tbl_rows=80, tbl_cols=-1, tbl_width_chars=200, float_precision=4, fmt_str_lengths=50):
        if len(months) >= 2:
            m0, m1 = months[-2], months[-1]
            f0 = set(per.filter(pl.col("month") == m0)["feature"].to_list())
            f1 = set(per.filter(pl.col("month") == m1)["feature"].to_list())
            print(f"\n== refit {m0} -> {m1}: {len(f1 & f0)} kept, added {sorted(f1 - f0) or 'none'}, dropped {sorted(f0 - f1) or 'none'}")
            mv = (per.filter(pl.col("month") == m1).select("feature", "family", pl.col("gain_share").alias("new"))
                  .join(per.filter(pl.col("month") == m0).select("feature", pl.col("gain_share").alias("old")), on="feature", how="full", coalesce=True)
                  .with_columns(pl.col("new").fill_null(0), pl.col("old").fill_null(0)).with_columns((pl.col("new") - pl.col("old")).alias("delta"))
                  .sort(pl.col("delta").abs(), descending=True).head(15))
            print(mv)
        last = per.filter(pl.col("month") == months[-1]).with_columns(pl.col("feature").is_in(list(assign)).alias("has_formula"))
        fam = (last.group_by("family").agg(pl.len().alias("features"), pl.col("gain_share").sum().alias("gain"),
                                            pl.col("has_formula").sum().alias("with_formula"),
                                            pl.col("gain_share").filter(~pl.col("has_formula")).sum().alias("gain_open"))
               .sort("gain", descending=True).with_columns(pl.col("gain").cum_sum().alias("cum_gain")))
        print(f"\n== worklist, month {months[-1]}: families by gain share, features with a formula, gain still open")
        print(fam)
        top = last.filter(~pl.col("has_formula")).sort("gain_share", descending=True).head(30)
        print("\n== top 30 features still without a formula")
        print(top.select("feature", "family", "gain_share"))
        done = last.filter(pl.col("has_formula")).sort("gain_share", descending=True)
        if done.height:
            print("\n== features with a formula")
            print(done.with_columns(pl.col("feature").replace_strict(assign, default="").alias("formula")).select("feature", "gain_share", "formula"))
        print(f"\ncovered gain: {float(last.filter(pl.col('has_formula'))['gain_share'].sum()):.3f} of {float(last['gain_share'].sum()):.3f}")


if __name__ == "__main__":
    main()
