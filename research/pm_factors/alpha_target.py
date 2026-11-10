r"""PM's own recorded predictions (``livev1/rec10_p/<D>/alpha.parquet`` on 2.23) -> ``ic_score --target`` input.

The file has one row per scored snapshot: sid, date, src_epoch (exchange second), rec_epoch (receive time), alpha, all
float, epoch seconds. ``ic_score`` reads ``<out>/date=D.parquet`` with ``sid``, ``t`` (trading-clock seconds of the receive
time, as ``pm_signals`` writes it) and ``prediction``; full precision, no log matching.

    python research/pm_factors/alpha_target.py --alpha /work/crucible_data/pm_livev1/rec10_p/20260430/alpha.parquet --out /work/crucible_data/pm_alpha
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def trading_clock(ns: np.ndarray) -> np.ndarray:
    """Epoch ns -> seconds since 09:30 local with the lunch break removed (``pm_signals.trading_clock``)."""
    s = (ns / 1e9 + 28800) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alpha", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/pm_alpha"))
    a = ap.parse_args()
    d = pl.read_parquet(a.alpha)
    days = d["date"].unique().to_list()
    a.out.mkdir(parents=True, exist_ok=True)
    for day in days:
        g = d.filter(pl.col("date") == day).with_columns(
            pl.col("sid").cast(pl.Int64), (pl.col("rec_epoch") * 1e9).round().cast(pl.Int64).alias("md_recv_ts"),
            (pl.col("src_epoch") * 1e9).round().cast(pl.Int64).alias("md_ts"), pl.col("alpha").alias("prediction"))
        g = g.with_columns(pl.Series("t", trading_clock(g["md_recv_ts"].to_numpy())), pl.lit(int(day)).alias("date"),
                           pl.lit("alpha").alias("src")).select("date", "sid", "md_ts", "md_recv_ts", "t", "prediction", "src").sort("sid", "t")
        f = a.out / f"date={int(day)}.parquet"
        g.write_parquet(f)
        per = g.group_by("sid").len()["len"].to_numpy()
        lag = (g["md_recv_ts"] - g["md_ts"]).to_numpy() / 1e6
        print(f"{int(day)}: {g.height:,} predictions, {g['sid'].n_unique()} stocks, per stock p10/p50/p90 "
              f"{np.percentile(per, 10):.0f}/{np.median(per):.0f}/{np.percentile(per, 90):.0f}; t {g['t'].min():.1f} .. {g['t'].max():.1f}; "
              f"receive - exchange ms p1/p50/p99 " + "/".join(f"{np.percentile(lag, q):.0f}" for q in (1, 50, 99)) + f" -> {f}")


if __name__ == "__main__":
    main()
