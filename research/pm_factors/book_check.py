r"""Does our book replay match the exchange's book? Replay totals at a snapshot's instant against that snapshot's own totals.

``book.parquet`` (``book_totals.py``) holds the replayed resting volume at the last snapshot at / strictly before each bar
end (``*_q0`` / ``*_q1``, ``*_q1s`` = events strictly before that instant); the basics ``m5`` holds the feed's total bid /
ask volume of the same snapshots (``btot`` / ``atot`` at the bar end, ``btot_pre`` / ``atot_pre`` strictly before). Equal
volumes mean the replay is the exchange's book there; ZZAL's remaining misses then lie elsewhere. The 15:00 bar is left
out (it reads 14:57:00).

    python research/pm_factors/book_check.py --days 20261020,20261021,20261022
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

BASICS_PM = Path("/work/crucible_data/store/pm_factors/basics_pm")
PAIRS = (("q0", "btot", "btot_q0"), ("q0", "atot", "atot_q0"), ("q1", "btot_pre", "btot_q1"), ("q1", "atot_pre", "atot_q1"),
         ("q1s", "btot_pre", "btot_q1s"), ("q1s", "atot_pre", "atot_q1s"))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days", required=True)
    ap.add_argument("--basics", type=Path, default=BASICS_PM)
    a = ap.parse_args()
    frames = []
    for d in a.days.split(","):
        p = a.basics / f"date={d}"
        if not (p / "book.parquet").exists():
            print(f"{d}: no book.parquet")
            continue
        bk = pl.read_parquet(p / "book.parquet")
        m5 = pl.read_parquet(p / "m5.parquet", columns=["symbol_id", "minute", "btot", "atot", "btot_pre", "atot_pre"])
        j = bk.with_columns(pl.col("symbol_id").cast(pl.Int64), pl.col("minute").cast(pl.Int64)).join(
            m5.with_columns(pl.col("symbol_id").cast(pl.Int64), pl.col("minute").cast(pl.Int64)), on=["symbol_id", "minute"])
        frames.append(j.filter(pl.col("minute") < 900))
    if not frames:
        return
    j = pl.concat(frames, how="diagonal_relaxed").with_columns(
        pl.when(pl.col("symbol_id") >= 600000).then(pl.lit("SH")).otherwise(pl.lit("SZ")).alias("m"))
    for tag, feed, ours in PAIRS:
        if ours not in j.columns:
            print(f"{ours}: missing (rerun book_totals.py)")
            continue
        x = j.filter(pl.col(feed).is_not_null() & pl.col(ours).is_not_null() & (pl.col(feed) > 0)).with_columns(
            (pl.col(ours) - pl.col(feed)).alias("diff"), (pl.col(ours) / pl.col(feed) - 1).alias("rel"))
        g = x.group_by("m").agg(
            (pl.col("diff") == 0).mean().alias("eq"), (pl.col("rel").abs() <= 1e-3).mean().alias("in1e3"),
            pl.col("rel").quantile(0.1).alias("p10"), pl.col("rel").quantile(0.5).alias("p50"), pl.col("rel").quantile(0.9).alias("p90"),
            pl.len().alias("n")).sort("m")
        print(f"{ours:9s} vs feed {feed:9s} " + " | ".join(
            f"{m} eq {eq:.3f} <=0.1% {w:.3f} ours/feed-1 {p10:+.4f}/{p50:+.4f}/{p90:+.4f} n {n}" for m, eq, w, p10, p50, p90, n in g.iter_rows()))


if __name__ == "__main__":
    main()
