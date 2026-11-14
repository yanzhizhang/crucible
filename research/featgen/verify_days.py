r"""The state engine over many days: fold backfill_days.py's verify_<D>.csv files (one row per PM key: share of stocks
equal at 1e-6 on all stocks and on SZSE) into one table -- per key the mean share over days, its worst day, and the days
it is fully equal -- so the keys still off (and whether they are off every day or only on some) show at a glance.

    python research/featgen/verify_days.py data/backfill_3s
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def main() -> None:
    """CLI entry point."""
    root = Path(sys.argv[1])
    fs = sorted(root.glob("verify_*.csv"))
    if not fs:
        raise SystemExit(f"no verify_*.csv under {root}")
    t = pl.concat([pl.read_csv(f).with_columns(pl.lit(f.stem[7:]).alias("day")) for f in fs], how="diagonal_relaxed")
    agg = (t.group_by("key").agg(pl.len().alias("days"), pl.col("SZSE").mean().alias("szse_mean"), pl.col("SZSE").min().alias("szse_worst"),
                                 pl.col("all").mean().alias("all_mean"), (pl.col("SZSE") >= 1).sum().alias("szse_full_days"),
                                 pl.col("day").sort_by("SZSE").first().alias("worst_day"), pl.col("mre").median().alias("mre"))
           .sort("szse_mean", "key"))
    print(f"{len(fs)} days ({fs[0].stem[7:]} .. {fs[-1].stem[7:]}), {agg.height} keys; SZSE fully equal on every day: "
          f"{int((agg['szse_full_days'] == agg['days']).sum())}; mean SZSE share >= 0.999: {int((agg['szse_mean'] >= 0.999).sum())}")
    with pl.Config(tbl_rows=200, tbl_width_chars=170, float_precision=4):
        print(agg.filter(pl.col("szse_mean") < 0.999))
    per_day = t.group_by("day").agg((pl.col("SZSE") >= 1).sum().alias("keys_szse_full"), pl.col("all").median().alias("median_all")).sort("day")
    print("per day: " + " | ".join(f"{d} {k} {m:.4f}" for d, k, m in per_day.iter_rows()))


if __name__ == "__main__":
    main()
