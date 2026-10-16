r"""Stage 3 diagnostic: PM samplerR against our variants by scale and rank, not only exact match.

``compare_pm.py`` says whether cells are equal; when nothing is, this says how they differ, per
column ``{as, ps, ab, pb} x {q50, q90, q95, q98}`` and per variant:

* ``ratio``   -- median of PM / ours (a constant ratio = a unit or a scale difference);
* ``corr_log`` -- Pearson correlation of log values across stocks (same quantity up to scale?);
* ``rank``    -- Spearman rank correlation (same ordering of stocks?);
* ``w10``     -- share of stocks within 10 % of PM.

It also prints a few stocks side by side, how PM's values are shaped (half-integers, round lots),
and the layout of a second PM file (``--pm2``, e.g. ZZAC's 48-row-per-stock samplerR)::

    python research/pm_factors/diag_sampler_r.py --date 20260429 \\
        --pm /data/prod/hstats/sod/HS300/ZZCR/20260430/samplerR.parquet \\
        --pm2 /data/prod/hstats/sod/HS300/ZZAC/20260430/samplerR.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

OUT = Path("/work/crucible_data/store/pm_factors/sampler_r")
ROLES = ("as", "ps", "ab", "pb")
QS = ("q50", "q90", "q95", "q98")
SHOW = (1.0, 2.0, 600000.0, 600519.0)


def shape_of(df: pl.DataFrame, cols: list[str]) -> dict[str, float]:
    """How the values look: share of half-integers, of whole hundreds, of integers."""
    v = pl.concat([df.select(pl.col(c).alias("v")) for c in cols]).drop_nulls()["v"]
    frac = (v - v.floor()).round(6)
    return {"n": float(v.len()), "half": float((frac == 0.5).mean()), "int": float((frac == 0.0).mean()),
            "x100": float(((v % 100) == 0).mean())}


def stats(pm: pl.DataFrame, ours: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per-column scale / rank agreement on the stocks both sides have."""
    j = pm.join(ours, on="sid", how="inner", suffix="_o")
    rows = []
    for c in cols:
        a, b = pl.col(c), pl.col(f"{c}_o")
        ok = j.filter(a.is_not_null() & b.is_not_null() & (a > 0) & (b > 0))
        rows.append(ok.select(
            pl.lit(c).alias("column"), pl.len().alias("n"),
            (a / b).median().alias("ratio"),
            pl.corr(a.log(), b.log()).alias("corr_log"),
            pl.corr(a, b, method="spearman").alias("rank"),
            ((a / b - 1).abs() <= 0.10).mean().alias("w10"),
            ((a - b).abs() <= 0.5).mean().alias("exact")))
    return pl.concat(rows)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pm", type=Path, required=True, help="PM samplerR in our layout (sid + 16 columns)")
    ap.add_argument("--date", required=True, help="our trade day (PM sod/<D> is built from D-1)")
    ap.add_argument("--variants", default="order_all,order_cont,trade_all")
    ap.add_argument("--pm2", type=Path, help="a second PM samplerR whose layout should just be described")
    a = ap.parse_args()
    pm = pl.read_parquet(a.pm)
    cols = [f"{r}.{q}" for r in ROLES for q in QS if f"{r}.{q}" in pm.columns]
    print(f"PM {a.pm}: {pm.shape}, value shape {shape_of(pm, cols)}")
    with pl.Config(tbl_rows=40, tbl_width_chars=220, float_precision=4, tbl_cols=20):
        side = [pm.filter(pl.col("sid").is_in(SHOW)).select("sid", *[f"as.{q}" for q in QS], *[f"pb.{q}" for q in QS])
                .with_columns(pl.lit("PM").alias("src"))]
        for v in filter(None, a.variants.split(",")):
            f = OUT / f"variant={v}" / f"date={a.date}" / "samplerR.parquet"
            if not f.exists():
                print(f"== {v}: {f} missing")
                continue
            ours = pl.read_parquet(f)
            print(f"== {v}: {ours.height} stocks, value shape {shape_of(ours, cols)}")
            print(stats(pm, ours, cols))
            side.append(ours.filter(pl.col("sid").is_in(SHOW)).select(
                "sid", *[f"as.{q}" for q in QS], *[f"pb.{q}" for q in QS]).with_columns(pl.lit(v).alias("src")))
        print("== side by side (as / pb quantiles)")
        print(pl.concat(side, how="diagonal").sort("sid", "src"))
        if a.pm2:
            d = pl.read_parquet(a.pm2)
            print(f"== {a.pm2}: {d.shape}")
            print(d.columns)
            if "rec_epoch" in d.columns:
                t = d.select(pl.from_epoch(pl.col("rec_epoch").cast(pl.Int64), "s").dt.replace_time_zone("UTC")
                             .dt.convert_time_zone("Asia/Shanghai").alias("t")).unique().sort("t")
                print(f"rec_epoch: {t.height} distinct, first {t['t'].head(3).to_list()}, last {t['t'].tail(3).to_list()}")
                print(f"rows per sid: {d.group_by('sid').len()['len'].unique().to_list()}")
            print(d.filter(pl.col("sid") == 1.0).head(6))


if __name__ == "__main__":
    main()
