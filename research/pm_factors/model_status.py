r"""Model line, the worklist: which features carry the model's gain, how they moved at the refit, and which have a formula.

Reads the probe's ``models.parquet`` / ``importance.parquet`` (``pm_models_probe.py`` over every month) and ``ic_score``'s
``ic_assign.json`` (+ ``ic_evidence.parquet`` / ``ic_log.parquet`` when present):

* model months: ordered by the files' modification date (a folder without YYYYMM is "undated");
* refit diff: features added / dropped between the last two months, and the biggest gain-share moves;
* worklist: every family of the latest month by gain share; a formula counts as solid when ``ic_score`` graded it strong
  (training-CDF gap <= 0.02 and IC lost without it at t >= 2), and ``gain_open`` is the gain without a solid formula;
* the open features by gain with the closest candidate the last scan saw, and the assigned ones with their grade.

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
    ap.add_argument("--top", type=int, default=25, help="open features listed")
    a = ap.parse_args()
    imp = pl.read_parquet(a.stage5 / "importance.parquet")
    if "set" in imp.columns:
        imp = imp.filter(pl.col("set") == a.set) if a.set in imp["set"].unique().to_list() else imp
    imp = imp.with_columns(pl.col("month").replace("", "undated"))
    mfile = a.stage5 / "models.parquet"
    if mfile.exists() and "mtime" in (m := pl.read_parquet(mfile)).columns:
        m = m.with_columns(pl.col("month").replace("", "undated"))
        if "set" in m.columns and a.set in m["set"].unique().to_list():
            m = m.filter(pl.col("set") == a.set)
        span = (m.group_by("month").agg(pl.len().alias("files"), pl.col("mtime").min().alias("mtime_min"),
                                        pl.col("mtime").max().alias("mtime_max"), pl.col("file").first().alias("example"))
                .sort("mtime_max", "month"))
        months = span["month"].to_list()
        with pl.Config(tbl_rows=20, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=70):
            print("== model months by file date")
            print(span)
    else:
        months = sorted(imp["month"].unique().to_list())
        print(f"months {months} (by name: rerun the probe for file dates)")
    per = imp.group_by("month", "feature", "family").agg(pl.col("gain_share").mean())
    assign = json.loads(a.assign.read_text()) if a.assign.exists() else {}
    efile, lfile = a.assign.parent / "ic_evidence.parquet", a.assign.parent / "ic_log.parquet"
    ev = pl.read_parquet(efile) if efile.exists() else pl.DataFrame(schema={"feature": pl.String, "grade": pl.String,
                                                                            "cdf_gap": pl.Float64, "t_loss": pl.Float64})
    strong = set(ev.filter(pl.col("grade") == "strong")["feature"].to_list())
    print(f"{len(assign)} features with an assigned formula, {len(strong)} strong ({a.assign})")
    with pl.Config(tbl_rows=80, tbl_cols=-1, tbl_width_chars=200, float_precision=4, fmt_str_lengths=60):
        if len(months) >= 2:
            m0, m1 = months[-2], months[-1]
            f0 = set(per.filter(pl.col("month") == m0)["feature"].to_list())
            f1 = set(per.filter(pl.col("month") == m1)["feature"].to_list())
            print(f"\n== refit {m0} -> {m1}: {len(f1 & f0)} kept, added {sorted(f1 - f0) or 'none'}, dropped {sorted(f0 - f1) or 'none'}")
            mv = (per.filter(pl.col("month") == m1).select("feature", "family", pl.col("gain_share").alias("new"))
                  .join(per.filter(pl.col("month") == m0).select("feature", pl.col("gain_share").alias("old")), on="feature", how="full", coalesce=True)
                  .with_columns(pl.col("new").fill_null(0), pl.col("old").fill_null(0)).with_columns((pl.col("new") - pl.col("old")).alias("delta"))
                  .sort(pl.col("delta").abs(), descending=True).head(8))
            print(mv)
        last = per.filter(pl.col("month") == months[-1]).with_columns(pl.col("feature").is_in(list(assign)).alias("has_formula"),
                                                                      pl.col("feature").is_in(list(strong)).alias("solid"))
        fam = (last.group_by("family").agg(pl.len().alias("features"), pl.col("gain_share").sum().alias("gain"),
                                            pl.col("has_formula").sum().alias("formula"), pl.col("solid").sum().alias("strong"),
                                            pl.col("gain_share").filter(~pl.col("solid")).sum().alias("gain_open"))
               .sort("gain_open", descending=True).with_columns(pl.col("gain_open").cum_sum().alias("cum_open")))
        print(f"\n== worklist, month {months[-1]}: families by gain without a strong formula (top 30)")
        print(fam.head(30))
        top = last.filter(~pl.col("solid")).sort("gain_share", descending=True).head(a.top)
        if lfile.exists():  # what the last scan saw for each feature: kept formula and the closest tries
            lg = pl.read_parquet(lfile).group_by("feature").last().select("feature", pl.col("top").str.slice(0, 70).alias("closest"))
            top = top.join(lg, on="feature", how="left")
        print(f"\n== top {a.top} features without a strong formula")
        print(top.with_columns(pl.col("feature").replace_strict(assign, default="").alias("formula"))
              .drop("month", "has_formula", "solid", "family"))
        done = last.filter(pl.col("has_formula")).sort("gain_share", descending=True)
        if done.height:
            print("\n== features with a formula")
            print(done.with_columns(pl.col("feature").replace_strict(assign, default="").alias("formula"))
                  .join(ev.select("feature", "grade", "cdf_gap", "t_loss"), on="feature", how="left")
                  .select("feature", "gain_share", "formula", "grade", "cdf_gap", "t_loss"))
        tot = float(last["gain_share"].sum())
        print(f"\ncovered gain: strong {float(last.filter(pl.col('solid'))['gain_share'].sum()):.3f}, "
              f"any formula {float(last.filter(pl.col('has_formula'))['gain_share'].sum()):.3f} of {tot:.3f}")


if __name__ == "__main__":
    main()
