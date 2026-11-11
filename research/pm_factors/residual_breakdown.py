r"""Where does our prediction part from PM's? ``ic_score --dump-pred`` rows broken down by time, board, stock and PM's tail.

Within each (day, mark) both predictions are ranked; the residual is PM's rank minus what our rank explains (OLS). Printed:

* the per-mark rank IC by half hour of the trading clock, per day (missing open / close inputs show up as a dip there);
* by board (SH main, STAR, SZ main, ChiNext): IC and residual variance (limit rules / tick size / ±20 % boards);
* stock persistence: the share of the residual variance a stock-day mean explains, the share a stock mean over all days
  explains, and the day-to-day correlation of the stock means (a persistent stock effect = a daily or static input missing);
* PM's tails: where PM ranks a stock in its top / bottom 10 %, how often ours does too (extremes driven by an input we lack);
* the 15 stocks with the largest mean |residual|, with their mean residual sign.

    python research/pm_factors/residual_breakdown.py --pred data/reports/stage6_r23_implied/pred.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def board(sid: pl.Expr) -> pl.Expr:
    """SH main / STAR / SZ main / ChiNext from the 6-digit code."""
    return (pl.when(sid >= 688000).then(pl.lit("STAR")).when(sid >= 600000).then(pl.lit("SH main"))
            .when(sid >= 300000).then(pl.lit("ChiNext")).otherwise(pl.lit("SZ main")))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    a = ap.parse_args()
    d = pl.read_parquet(a.pred).filter(pl.col("ours").is_finite() & pl.col("pm").is_finite())
    g = ["date", "mark"]
    d = d.with_columns(((pl.col(c).rank() - 0.5) / pl.len() - 0.5).over(g).alias(f"r_{c}") for c in ("ours", "pm"))
    d = d.with_columns((pl.col("r_pm") - (pl.col("r_pm") * pl.col("r_ours")).sum().over(g) / (pl.col("r_ours") ** 2).sum().over(g)
                        * pl.col("r_ours")).alias("res"), board(pl.col("sid")).alias("board"),
                       ((pl.col("mark") // 1800) * 30).cast(pl.Int64).alias("min_from_0930"))
    tot = d["res"].var()
    print(f"{a.pred}: {d.height:,} rows, {d['date'].n_unique()} days, {d.select(g).n_unique()} marks; overall IC "
          f"{d.group_by(g).agg(pl.corr('r_ours', 'r_pm')).select(pl.col('r_ours').mean()).item():.4f}")
    t = d.group_by("date", "min_from_0930", "mark").agg(pl.corr("r_ours", "r_pm").alias("ic")).group_by("date", "min_from_0930").agg(
        pl.col("ic").mean()).sort("min_from_0930").pivot(on="date", index="min_from_0930", values="ic")
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=3):
        print("\n== per-mark IC by half hour of the trading clock (0 = 09:30-10:00, 120 = 13:00-13:30)")
        print(t)
        b = d.group_by("board").agg(pl.len().alias("rows"), pl.corr("r_ours", "r_pm").alias("ic_pooled"),
                                     (pl.col("res").var() / tot).alias("res_var_rel")).sort("board")
        print("\n== by board (IC pooled over the board's rows; residual variance relative to all rows)")
        print(b)
    sd = d.group_by("date", "sid").agg(pl.col("res").mean().alias("m"), pl.len().alias("n"))
    s_all = d.group_by("sid").agg(pl.col("res").mean().alias("m_all"))
    share_sd = d.join(sd, on=["date", "sid"]).select(pl.col("m").var()).item() / tot
    share_s = d.join(s_all, on="sid").select(pl.col("m_all").var()).item() / tot
    days = sorted(d["date"].unique().to_list())
    w = sd.pivot(on="date", index="sid", values="m")
    cors = [w.select(pl.corr(str(x), str(y))).item() for i, x in enumerate(days) for y in days[i + 1:] if str(x) in w.columns and str(y) in w.columns]
    print(f"\n== stock persistence: stock-day mean explains {share_sd:.3f} of the residual variance, stock mean over all days {share_s:.3f}; "
          f"day-to-day corr of stock means " + (", ".join(f"{c:+.2f}" for c in cors) if cors else "n/a (one day)"))
    q = d.with_columns(pl.col("r_pm").rank().over(g).alias("kp"), pl.col("r_ours").rank().over(g).alias("ko"), pl.len().over(g).alias("nn"))
    for name, cond_p, cond_o in (("top 10 %", pl.col("kp") > 0.9 * pl.col("nn"), pl.col("ko") > 0.9 * pl.col("nn")),
                                 ("bottom 10 %", pl.col("kp") <= 0.1 * pl.col("nn"), pl.col("ko") <= 0.1 * pl.col("nn"))):
        sub = q.filter(cond_p)
        hit = sub.select(cond_o.mean()).item()
        print(f"  PM's {name}: ours in the same tail {hit:.3f} of the time (chance 0.100), ours median rank there "
              f"{sub['r_ours'].median():+.3f}")
    top = sd.group_by("sid").agg(pl.col("m").abs().mean().alias("abs_m"), pl.col("m").mean().alias("m")).sort("abs_m", descending=True).head(15)
    print("\n== 15 stocks with the largest mean |residual| (residual > 0: PM ranks it higher than ours explains)")
    print("  " + ", ".join(f"{int(r[0]):06d} {r[2]:+.3f}" for r in top.iter_rows()))


if __name__ == "__main__":
    main()
