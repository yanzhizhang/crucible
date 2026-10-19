r"""Stage 3 diagnostic: how ZZAC's intraday samplerR is updated from one five-minute slot to the next.

The grid (``sampler_r_grid.py``) found that pooling more days fits PM better and better (w10 0.31
at 1 day -> 0.46 at 10), with no quantile method standing out -- a long-memory statistic. ZZAC
records it every five minutes, so the update rule can be read off directly. For an exponentially
weighted average, every step satisfies::

    pm[t] - pm[t-1] = alpha * (slot[t] - pm[t-1])

with ``slot[t]`` the quantile of the orders of slot ``(t-1, t]``. This fits ``alpha`` (least
squares through the origin) per column, on raw and on log values, and reports R^2 and how often
PM did not change at all. A clear positive alpha with a useful R^2 = EWMA, and alpha gives the
memory (half-life in slots = ln 0.5 / ln(1 - alpha); 48 slots per day). Near-zero R^2 = not an
EWMA of slot quantiles (e.g. a rolling pool, or a different per-slot measure).

    python research/pm_factors/diag_sampler_update.py --date 20260429 \
        --ac /data/prod/hstats/sod/HS300/ZZAC/20260430/samplerR.parquet
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag_sampler_ac as A  # ZZAC reader and per-slot quantiles

R = A.R


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="trade day of the ZZAC rows (sod/<D+1> holds day D)")
    ap.add_argument("--ac", type=Path, required=True)
    ap.add_argument("--index", default="000300")
    a = ap.parse_args()
    ac = A.ac_long(a.ac)
    epochs = sorted(ac["hhmm"].unique().to_list())
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    uni = R.ced_universe(a.index, a.date)
    con.register("uni", uni if uni is not None else R.universe(a.index))
    slot = pl.concat([A.ours(con, a.date, e, "slot") for e in epochs]).select("sid", "column", "hhmm", pl.col("ours").alias("slot"))
    prev = dict(zip(epochs[1:], epochs[:-1], strict=True))
    pm_prev = ac.filter(pl.col("hhmm").is_in(list(prev.values()))).with_columns(
        pl.col("hhmm").replace_strict({v: k for k, v in prev.items()}).alias("hhmm")).rename({"pm": "pm_prev"})
    d = (ac.filter(pl.col("hhmm").is_in(list(prev))).join(pm_prev, on=["sid", "hhmm", "column"])
         .join(slot, on=["sid", "hhmm", "column"]).filter((pl.col("pm") > 0) & (pl.col("pm_prev") > 0) & (pl.col("slot") > 0)))
    print(f"steps: {d.height} (stock x column x {len(prev)} transitions)")
    half = lambda a_: math.log(0.5) / math.log(1 - a_) if 0 < a_ < 1 else float("nan")  # noqa: E731  slots
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=4):
        for name, y, x in (("raw", pl.col("pm") - pl.col("pm_prev"), pl.col("slot") - pl.col("pm_prev")),
                           ("log", (pl.col("pm") / pl.col("pm_prev")).log(), (pl.col("slot") / pl.col("pm_prev")).log())):
            dd = d.with_columns(y.alias("y"), x.alias("x")).filter(pl.col("y").is_finite() & pl.col("x").is_finite())
            res = dd.group_by("column").agg(
                pl.len().alias("n"),
                ((pl.col("x") * pl.col("y")).sum() / (pl.col("x") ** 2).sum()).alias("alpha"),
                pl.corr("x", "y").alias("corr"),
                (pl.col("y") == 0).mean().alias("unchanged"))
            res = res.with_columns((pl.col("corr") ** 2).alias("r2_approx"),
                                   pl.col("alpha").map_elements(half, return_dtype=pl.Float64).alias("half_life_slots"))
            print(f"== {name}: pm[t]-pm[t-1] = alpha * (slot[t] - pm[t-1])")
            print(res.sort("column"))
        print("== 000001 as.q50 path: PM vs this slot's own quantile")
        print(d.filter((pl.col("sid") == 1.0) & (pl.col("column") == "as.q50")).sort("hhmm")
              .select("hhmm", "pm_prev", "pm", "slot").head(20))


if __name__ == "__main__":
    main()
