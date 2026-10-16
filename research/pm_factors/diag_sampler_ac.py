r"""Stage 3 diagnostic: ZZAC's intraday samplerR (48 five-minute rows per stock) and how ZZCR relates to it.

ZZAC's ``sod/<D>`` samplerR has one row per stock and ``rec_epoch`` (09:35, 09:40, ... 15:00 of
day D-1) with ``{as, bs} x {q5, q10, q20, q50, q70, q80, q90, q95, q98}``; ZZCR's has one row per
stock with ``{as, ps, ab, pb} x {q50, q90, q95, q98}``. Two questions, both answerable from
data already on the box:

1. Is ZZCR one of ZZAC's time points? Share of stocks where ZZCR's ``as.qN`` / ``ab.qN`` equals
   ZZAC's ``as.qN`` / ``bs.qN`` at each epoch.
2. Does ZZAC follow our per-order totals of day D-1 accumulated up to each epoch? Variants:
   ``cum_all`` (every trade up to the epoch), ``cum_cont`` (continuous session only, from 09:30)
   and ``slot`` (only the trades of that five-minute slot). Per order = fills summed per
   aggressor order id; ``as`` = active sell, ``bs`` = active buy; quantiles linear (``quantile_cont``).

    python research/pm_factors/diag_sampler_ac.py --date 20260429 \
        --ac /data/prod/hstats/sod/HS300/ZZAC/20260430/samplerR.parquet \
        --cr /data/prod/hstats/sod/HS300/ZZCR/20260430/samplerR.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampler_r as R  # universe + catalog location

LEVELS = (5, 10, 20, 50, 70, 80, 90, 95, 98)
EPOCHS = "09:35,10:00,10:30,11:30,13:05,14:00,15:00"
TZ = dt.timezone(dt.timedelta(hours=8))


def ac_long(path: Path) -> pl.DataFrame:
    """ZZAC samplerR -> long (sid, hhmm, column, pm)."""
    d = pl.read_parquet(path).with_columns(
        pl.from_epoch(pl.col("rec_epoch").cast(pl.Int64), "s").dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Asia/Shanghai").dt.strftime("%H:%M").alias("hhmm"))
    cols = [c for c in d.columns if c[:3] in ("as.", "bs.")]
    return d.unpivot(index=["sid", "hhmm"], on=cols, variable_name="column", value_name="pm")


def cr_vs_ac(ac: pl.DataFrame, cr_path: Path) -> None:
    """At which epoch (if any) does ZZCR equal ZZAC, column by column."""
    cr = pl.read_parquet(cr_path)
    pairs = [(f"as.q{q}", f"as.q{q}") for q in (50, 90, 95, 98)] + [(f"ab.q{q}", f"bs.q{q}") for q in (50, 90, 95, 98)]
    rows = []
    for c_cr, c_ac in pairs:
        if c_cr not in cr.columns:
            continue
        j = ac.filter(pl.col("column") == c_ac).join(cr.select("sid", pl.col(c_cr).alias("cr")), on="sid")
        rows.append(j.group_by("hhmm").agg(((pl.col("pm") - pl.col("cr")).abs() <= 0.5).mean().alias(f"{c_cr}=ac.{c_ac[3:]}")))
    out = rows[0]
    for r in rows[1:]:
        out = out.join(r, on="hhmm")
    print("== share of stocks where ZZCR equals ZZAC at that epoch (top epochs by as.q50)")
    print(out.sort(out.columns[1], descending=True).head(8))


def ours(con: duckdb.DuckDBPyConnection, date: str, hhmm: str, variant: str) -> pl.DataFrame:
    """Per-order active totals up to (or within the slot ending at) ``hhmm`` -> long quantiles."""
    d = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    h, m = int(hhmm[:2]), int(hhmm[3:])
    end = dt.datetime(d.year, d.month, d.day, h, m, tzinfo=TZ)
    ns = lambda t: int(t.timestamp()) * 1_000_000_000  # noqa: E731  local wall time -> epoch ns
    lo = {"cum_all": None, "cum_cont": dt.datetime(d.year, d.month, d.day, 9, 30, tzinfo=TZ),
          "slot": end - dt.timedelta(minutes=5)}[variant]
    if variant == "slot" and hhmm == "13:05":
        lo = dt.datetime(d.year, d.month, d.day, 13, 0, tzinfo=TZ)
    cond = f"r.time <= {ns(end)}" + (f" AND r.time > {ns(lo)}" if lo is not None else "")
    qlist = "[" + ", ".join(str(q / 100) for q in LEVELS) + "]"
    long = con.sql(f"""
      WITH t AS (
        SELECT r.symbol_id, r.dir, r.volume, r.buy_seq_id, r.sell_seq_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {int(date)} AND r.trade_type = 1 AND r.dir IN ({R.BUY}, {R.SELL}) AND {cond}),
      o AS (SELECT symbol_id, CASE dir WHEN {R.BUY} THEN 'bs' ELSE 'as' END AS role,
                   CASE dir WHEN {R.BUY} THEN buy_seq_id ELSE sell_seq_id END AS oid, sum(volume) AS v
            FROM t GROUP BY ALL)
      SELECT symbol_id, role, quantile_cont(v, {qlist}) AS q FROM o GROUP BY ALL""").pl()
    return (long.with_columns(pl.lit(list(LEVELS), dtype=pl.List(pl.Int64)).alias("lvl")).explode("q", "lvl")
            .select(pl.col("symbol_id").cast(pl.Float64).alias("sid"),
                    pl.format("{}.q{}", "role", "lvl").alias("column"), pl.col("q").alias("ours"),
                    pl.lit(hhmm).alias("hhmm"), pl.lit(variant).alias("variant")))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="trade day of the PM file (sod/<D+1> holds day D)")
    ap.add_argument("--ac", type=Path, required=True)
    ap.add_argument("--cr", type=Path)
    ap.add_argument("--epochs", default=EPOCHS)
    ap.add_argument("--variants", default="cum_all,cum_cont,slot")
    ap.add_argument("--index", default="000300")
    a = ap.parse_args()
    ac = ac_long(a.ac)
    print(f"ZZAC: {ac['sid'].n_unique()} stocks, epochs {ac['hhmm'].min()} .. {ac['hhmm'].max()} ({ac['hhmm'].n_unique()})")
    with pl.Config(tbl_rows=60, tbl_width_chars=220, float_precision=4, tbl_cols=20):
        if a.cr:
            cr_vs_ac(ac, a.cr)
        con = duckdb.connect(str(R.CATALOG), read_only=True)
        uni = R.ced_universe(a.index, a.date)
        con.register("uni", uni if uni is not None else R.universe(a.index))
        frames = [ours(con, a.date, e, v) for e in a.epochs.split(",") for v in a.variants.split(",")]
        j = pl.concat(frames).join(ac, on=["sid", "hhmm", "column"], how="inner").filter(
            (pl.col("pm") > 0) & (pl.col("ours") > 0))
        summ = j.group_by("variant", "hhmm").agg(
            pl.len().alias("cells"), ((pl.col("pm") - pl.col("ours")).abs() <= 0.5).mean().alias("exact"),
            ((pl.col("pm") / pl.col("ours") - 1).abs() <= 0.10).mean().alias("w10"),
            (pl.col("pm") / pl.col("ours")).median().alias("ratio"),
            pl.corr(pl.col("pm").log(), pl.col("ours").log()).alias("corr_log"))
        print("== ZZAC vs our per-order totals, by variant and epoch (all 18 columns pooled)")
        print(summ.sort("variant", "hhmm"))
        best = summ.sort("w10", descending=True).row(0, named=True)
        print(f"== per column at the best cell ({best['variant']} {best['hhmm']})")
        print(j.filter((pl.col("variant") == best["variant"]) & (pl.col("hhmm") == best["hhmm"])).group_by("column").agg(
            ((pl.col("pm") - pl.col("ours")).abs() <= 0.5).mean().alias("exact"),
            ((pl.col("pm") / pl.col("ours") - 1).abs() <= 0.10).mean().alias("w10"),
            (pl.col("pm") / pl.col("ours")).median().alias("ratio"),
            pl.corr(pl.col("pm"), pl.col("ours"), method="spearman").alias("rank")).sort("column"))
        print("== 000001 at the best cell")
        print(j.filter((pl.col("sid") == 1.0) & (pl.col("variant") == best["variant"]) & (pl.col("hhmm") == best["hhmm"]))
              .select("column", "pm", "ours").sort("column"))


if __name__ == "__main__":
    main()
