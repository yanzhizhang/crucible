r"""How does PM switch between its sub-models s1..s4 during the day (hard switch by clock, or blending)? Read-only.

1 ``--meta`` (pm_lgb_meta.csv: month, fit, model, col, lo, hi, splits): per column, the training ranges of s1..s4 in one
  model month and fit. A time-of-day column (seconds since the open, a share of the day's volume ...) trained on disjoint
  or nested day segments shows disjoint / shifted ranges -- the segment boundaries in the column's own units. Columns
  are ranked by how far apart the four ranges sit relative to their union.
2 ``--alpha`` (PM's recorded alpha of one day): cross-sectional mean / std / stock count per local minute, and the
  minute-to-minute jump of the std; a hard switch shows as a step at fixed minutes, blending as a smooth curve.

    python research/pm_factors/seg_probe.py --meta data/reports/pm_lgb_meta.csv --month 202604 \
        --alpha /work/crucible_data/pm_livev1/rec10_p/20260430/alpha.parquet --names /data/prod/models/lgb_fn_v1_209_cte.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, default=Path("data/reports/pm_lgb_meta.csv"))
    ap.add_argument("--month", default="202604", help="model month folder to compare s1..s4 in")
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--names", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--alpha", type=Path)
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    names = {}
    if a.names.exists():
        names = {i: ln.split(",")[0].strip() for i, ln in enumerate(x for x in a.names.read_text(errors="replace").splitlines() if x.strip())}
    if a.meta.exists():
        m = pl.read_csv(a.meta, schema_overrides={"month": pl.String}).filter((pl.col("month") == a.month) & (pl.col("fit") == a.fit))
        print(f"== 1 {a.meta}: {a.month} {a.fit}, sub-models {sorted(m['model'].unique().to_list())}")
        w = m.pivot(on="model", index="col", values=["lo", "hi"])
        subs = sorted(m["model"].unique().to_list())
        rows = []
        for r in w.iter_rows(named=True):
            lo = np.array([r.get(f"lo_{s}") for s in subs], float)
            hi = np.array([r.get(f"hi_{s}") for s in subs], float)
            if not np.all(np.isfinite(lo) & np.isfinite(hi)):
                continue
            union = hi.max() - lo.min()
            if union <= 0:
                continue
            spread = (np.ptp(lo) + np.ptp(hi)) / union  # 0 = the same range in every sub-model
            overlap = max(0.0, hi.min() - lo.max()) / union  # 0 = some two ranges are disjoint
            rows.append({"col": r["col"], "name": names.get(r["col"], ""), "spread": spread, "common_overlap": overlap,
                         **{f"{s}": f"[{lo[i]:.6g}, {hi[i]:.6g}]" for i, s in enumerate(subs)}})
        t = pl.DataFrame(rows).sort("spread", descending=True)
        with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=240, fmt_str_lengths=40, float_precision=3):
            print("  columns whose four training ranges differ most (spread = range of the los + range of the his over their union):")
            print(t.head(a.top))
            print(f"  columns with ranges disjoint between some two sub-models: {t.filter(pl.col('common_overlap') == 0).height}")
    if a.alpha and a.alpha.exists():
        al = pl.read_parquet(a.alpha)
        tc = "rec_epoch" if "rec_epoch" in al.columns else "src_epoch"
        al = al.with_columns(((((pl.col(tc) + 28800) % 86400) // 60).cast(pl.Int64)).alias("minute"),
                             (pl.col(tc).round(0)).alias("sec"))
        snap = al.group_by("minute", "sec").agg(pl.col("alpha").std().alias("xs_std"), pl.col("alpha").mean().alias("xs_mean"),
                                                pl.col("alpha").abs().median().alias("abs_med"), pl.len().alias("n"))
        per = snap.filter(pl.col("n") >= 30).group_by("minute").agg(pl.col("xs_std", "xs_mean", "abs_med", "n").median(), pl.len().alias("seconds")).sort("minute")
        per = per.with_columns((pl.col("xs_std") / pl.col("xs_std").shift(1) - 1).alias("std_jump"),
                               pl.format("{}:{}", pl.col("minute") // 60, (pl.col("minute") % 60).cast(pl.String).str.zfill(2)).alias("hhmm"))
        print(f"\n== 2 {a.alpha}: {al.height:,} rows; per local minute (snapshot seconds with >= 30 stocks):")
        with pl.Config(tbl_rows=-1, float_precision=6):
            print(per.select("hhmm", "seconds", "n", "xs_mean", "xs_std", "abs_med", "std_jump"))
            big = per.filter(pl.col("std_jump").abs() > 0.15)
            print(f"  minutes whose xs std jumps by more than 15 % from the previous minute: {big['hhmm'].to_list()}")


if __name__ == "__main__":
    main()
