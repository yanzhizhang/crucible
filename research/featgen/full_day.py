r"""0430 at the full 3 s grid: day.py's outputs + PM's alpha -> an implied_scan / greedy_fill day directory, plus residual studies.

alpha.parquet has ~4800 PM predictions per stock, but every scan so far used 240 random marks. With every 3 s mark the
single-column test has ~20x the rows. Writes <out>/x.parquet (date, sid, mark, model columns), pred.parquet (date, sid,
mark, pm = PM's latest prediction at or before the mark, stale > --stale s -> NaN), cal_<D>.json (copied), and
base_<D>.parquet (day.py's bases + the engine's zzmh.r / zzeg.c keys at the marks, for --extra).

``--study``: the residual of our prediction (open columns NaN, day.py's pred) against PM's, per mark cross-sectional
z-scores, (a) by time of day (30 min bins), (b) by drawdown from zzmh.hm (sqrt(zzmh.r) / 2; 0 = at the day's high), (c) split
into time scales per stock (the share of residual variance in bands of a running mean: <15 s, 15-60 s, 1-5 min, 5-30 min,
>30 min) -- fast bands point at fast inputs (zzmh.r ~1 s, zzcb.r ~4.7 s), slow ones at profiles / long averages.

    python research/featgen/full_day.py --date 20260430 --day data/day_r126 --target /work/crucible_data/pm_alpha/date=20260430.parquet \
        --cal data/reports/stage6_r70_0430/cal_20260430.json --out data/full_0430 --study
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl


def zs(df: pl.DataFrame, col: str) -> pl.Expr:
    """Cross-sectional z-score of col within each mark."""
    return ((pl.col(col) - pl.col(col).mean().over("mark")) / pl.col(col).std().over("mark"))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--day", type=Path, required=True, help="day.py --out of this day (vec/, base/, engine/, pred/)")
    ap.add_argument("--target", type=Path, required=True, help="PM's predictions (sid, t, prediction), e.g. pm_alpha/date=D.parquet")
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--stale", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--study", action="store_true")
    a = ap.parse_args()
    d = a.date
    vec = pl.read_parquet(a.day / "vec" / f"vec_{d}.parquet").with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64))
    tg = pl.read_parquet(a.target, columns=["sid", "t", "prediction"]).with_columns(pl.col("sid").cast(pl.Int64), pl.col("t").cast(pl.Float64)).sort("t")
    key = vec.select("sid", "mark").sort("mark")
    pm = key.join_asof(tg.rename({"t": "tp"}), left_on="mark", right_on="tp", by="sid", strategy="backward")
    pm = pm.with_columns(pl.when(pl.col("mark") - pl.col("tp") <= a.stale).then(pl.col("prediction")).alias("pm")).select("sid", "mark", "pm")
    a.out.mkdir(parents=True, exist_ok=True)
    vec.with_columns(pl.lit(int(d)).alias("date")).select("date", "sid", "mark", *[c for c in vec.columns if c not in ("sid", "mark")]).write_parquet(a.out / "x.parquet")
    pm.with_columns(pl.lit(int(d)).alias("date")).select("date", "sid", "mark", "pm").write_parquet(a.out / "pred.parquet")
    shutil.copy(a.cal, a.out / f"cal_{d}.json")
    base = pl.read_parquet(a.day / "base" / f"base_{d}.parquet").with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64))
    eng = a.day / "engine" / f"marks_{d}.parquet"
    if eng.exists():
        e = pl.read_parquet(eng).with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64))
        keep = [c for c in e.columns if c.startswith(("zzmh.r", "zzeg.c", "zzmh.hm"))]
        base = base.join(e.select("sid", "mark", *keep), on=["sid", "mark"], how="left")
    base.write_parquet(a.out / f"base_{d}.parquet")
    print(f"{d}: {vec.height:,} rows ({vec['sid'].n_unique()} stocks x {vec['mark'].n_unique()} marks), PM found {pm['pm'].is_not_null().mean():.3f} -> {a.out}")
    if not a.study:
        return
    ours = pl.read_parquet(a.day / "pred" / f"pred_{d}.parquet")
    oc = next(c for c in ours.columns if c not in ("date", "sid", "mark"))
    ours = ours.with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64)).select("sid", "mark", pl.col(oc).alias("ours"))
    r = pm.join(ours, on=["sid", "mark"]).drop_nulls().filter(pl.col("pm").is_finite() & pl.col("ours").is_finite())
    r = r.with_columns(zs(r, "pm").alias("zpm"), zs(r, "ours").alias("zours")).with_columns((pl.col("zpm") - pl.col("zours")).alias("res"))
    corr = r.group_by("mark").agg(pl.corr("pm", "ours", method="spearman").alias("rho"))["rho"].mean()
    print(f"  per-mark rank corr PM vs ours (open columns NaN): {corr:.4f}; residual var {r['res'].var():.4f}")
    print("  (a) by time of day (30 min bins on the trading clock): mark bin, rows, mean |res|, corr")
    for (b,), g in sorted(r.with_columns((pl.col("mark") // 1800).alias("b")).group_by("b"), key=lambda kv: kv[0]):
        c = g.group_by("mark").agg(pl.corr("pm", "ours", method="spearman").alias("rho"))["rho"].mean()
        print(f"    {int(b) * 30:>4}-{int(b) * 30 + 30:<4} min  {g.height:>8,}  {g['res'].abs().mean():.4f}  {c:.4f}")
    if "zzmh.r.sum" in base.columns:
        dd = base.select("sid", "mark", (pl.col("zzmh.r.sum").sqrt() / 2).alias("dd"))
        rd = r.join(dd, on=["sid", "mark"], how="left").drop_nulls("dd")
        cuts = [0, 1e-9, 0.002, 0.005, 0.01, 0.02, 0.04, 1]
        print("  (b) by drawdown from zzmh.hm: bin, rows, mean res (PM above ours = +), mean |res|")
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            g = rd.filter((pl.col("dd") >= lo) & (pl.col("dd") < hi)) if lo else rd.filter(pl.col("dd") < hi)
            if g.height:
                print(f"    [{lo:.3%}, {hi:.3%})  {g.height:>8,}  {g['res'].mean():+.4f}  {g['res'].abs().mean():.4f}")
    print("  (c) residual variance by time scale (per stock, running means of the 3 s residual series): band, share")
    tot = 0.0
    parts = {"<15s": 0.0, "15-60s": 0.0, "1-5min": 0.0, "5-30min": 0.0, ">30min": 0.0}
    for (_,), g in r.sort("mark").group_by("sid"):
        x = g["res"].to_numpy()
        if x.size < 200:
            continue
        x = x - x.mean()
        sm = {w: np.convolve(x, np.ones(w) / w, mode="same") for w in (5, 20, 100, 600)}  # 15 s, 60 s, 5 min, 30 min at 3 s steps
        bands = [x - sm[5], sm[5] - sm[20], sm[20] - sm[100], sm[100] - sm[600], sm[600]]
        for k, bnd in zip(parts, bands, strict=True):
            parts[k] += float((bnd ** 2).sum())
        tot += float((x ** 2).sum())
    for k, v in parts.items():
        print(f"    {k:<8} {v / tot:.3f}")


if __name__ == "__main__":
    main()
