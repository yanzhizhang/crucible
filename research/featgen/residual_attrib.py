r"""Which of our quantities does the residual (PM - ours, open columns NaN) line up with, and when?

R126: on 0430's full 3 s grid the residual concentrates in the first 30 minutes and near the day's high. Single fills cannot
tell candidates apart (64 other columns NaN). This asks the residual directly: per mark, the cross-sectional rank
correlation of the residual (PM z-score minus ours) with every base / extra quantity, averaged inside slices -- time of day
(0-5, 5-15, 15-30, 30-120, 120-240 min) and drawdown from the day's high (< 0.2%, 0.2-1%, >= 1%). A quantity the residual
follows (and in the slice it follows it) is a candidate building block of a missing column. Read-only.

    python research/featgen/residual_attrib.py --day data/day_r126 --full data/full_0430 --date 20260430
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--day", type=Path, required=True, help="day.py --out (pred/)")
    ap.add_argument("--full", type=Path, required=True, help="full_day.py --out (pred.parquet, base_<D>.parquet, extra_<D>.parquet)")
    ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    d = a.date
    pm = pl.read_parquet(a.full / "pred.parquet").select(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64), "pm")
    ours = pl.read_parquet(a.day / "pred" / f"pred_{d}.parquet")
    oc = next(c for c in ours.columns if c not in ("date", "sid", "mark"))
    ours = ours.select(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64), pl.col(oc).alias("ours"))
    r = pm.join(ours, on=["sid", "mark"]).filter(pl.col("pm").is_finite() & pl.col("ours").is_finite())
    z = lambda c: (pl.col(c) - pl.col(c).mean().over("mark")) / pl.col(c).std().over("mark")  # noqa: E731
    r = r.with_columns((z("pm") - z("ours")).alias("res"))
    base = pl.read_parquet(a.full / f"base_{d}.parquet").with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64))
    ex = a.full / f"extra_{d}.parquet"
    if ex.exists():
        base = base.join(pl.read_parquet(ex).with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64)), on=["sid", "mark"], how="left")
    cols = [c for c in base.columns if c not in ("sid", "mark") and base.schema[c].is_numeric() and not c.endswith(".wgt")]
    r = r.join(base, on=["sid", "mark"], how="left")
    r = r.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in cols])
    dd = pl.col("mh.dd") if "mh.dd" in r.columns else None
    slices = {"0-5m": pl.col("mark") <= 300, "5-15m": (pl.col("mark") > 300) & (pl.col("mark") <= 900), "15-30m": (pl.col("mark") > 900) & (pl.col("mark") <= 1800),
              "30-120m": (pl.col("mark") > 1800) & (pl.col("mark") <= 7200), "120-240m": pl.col("mark") > 7200}
    if dd is not None:
        slices.update({"dd<0.2%": dd < 0.002, "dd0.2-1%": (dd >= 0.002) & (dd < 0.01), "dd>=1%": dd >= 0.01})
    rk = lambda c: pl.col(c).rank().over("mark")  # noqa: E731
    res = {}
    for name, cond in slices.items():
        g = r.filter(cond)
        if g.height < 200:
            continue
        g = g.with_columns([rk(c).alias(f"_r_{c}") for c in ["res", *cols]])
        per = g.group_by("mark").agg([pl.corr("_r_res", f"_r_{c}").alias(c) for c in cols] + [pl.len().alias("_n")]).filter(pl.col("_n") >= 30)
        res[name] = {c: float(per[c].drop_nans().mean()) if per[c].drop_nans().len() else float("nan") for c in cols}
        top = sorted(((abs(v), c, v) for c, v in res[name].items() if v == v), reverse=True)[: a.top]
        print(f"-- {name}: {g.height:,} rows, {per.height} marks; residual vs quantity, mean per-mark rank corr:")
        print("   " + "  ".join(f"{c} {v:+.3f}" for _, c, v in top))
    allc = sorted(((np.nanmean([abs(res[s].get(c, np.nan)) for s in res]), c) for c in cols), reverse=True)[: a.top * 2]
    print("-- strongest over all slices (mean |corr|): " + "  ".join(f"{c} {v:.3f}" for v, c in allc))


if __name__ == "__main__":
    main()
