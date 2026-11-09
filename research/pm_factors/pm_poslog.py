r"""PM's ``pos`` (runS decision lines) against chrono's own position log on 224, and both against PM's prediction.

chrono writes ``position_log_csv`` every 30 s (``time,ts,trading_day,strategy,account,symbol,qty,yesterday_qty,today_qty,
sold_qty,frozen_qty,sellable_qty,avg_cost,base_qty``; ts epoch ns, symbol the numeric code). Each decision row (sid,
src_epoch) takes the latest log row of its stock at or before it; printed, per day:

* how often pos equals qty, yesterday_qty, sellable_qty, base_qty, base_qty - qty, qty - base_qty (exactly, and within
  100 shares), and the correlation of pos with each;
* the same for executp, and executp against pos clipped by sellable_qty (a sell cannot exceed what is sellable);
* qty / base_qty over the day (does the holding move at all, is it at its target);
* PM's prediction against the holding gap: corr(fa, base_qty - qty) and the sign agreement of executp with base_qty - qty.

    python research/pm_factors/pm_poslog.py --poslog /work/crucible_data/chrono_poslog --state /work/crucible_data/pm_signals/state --dates 20261106,20261109
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message="Sortedness of columns")
LOG = ["qty", "yesterday_qty", "today_qty", "sold_qty", "frozen_qty", "sellable_qty", "base_qty"]


def corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 and a[m].std() > 0 and b[m].std() > 0 else float("nan")


def day(poslog: Path, state: Path, d: str) -> None:
    files = sorted(poslog.glob(f"{d}/position_log/*.csv"))  # only the position log, not the base_position csv next to it
    if not files:
        print(f"\n== {d}: no position log under {poslog}/{d}")
        return
    lg = pl.concat([pl.read_csv(f, infer_schema_length=10000) for f in files], how="diagonal_relaxed")
    lg = lg.with_columns(pl.col("symbol").cast(pl.Int64).alias("sid"), pl.col("ts").cast(pl.Int64),
                         *[pl.col(c).cast(pl.Float64) for c in LOG if c in lg.columns]).sort("sid", "ts")
    print(f"\n== {d}: {lg.height:,} position-log rows from {len(files)} file(s), {lg['sid'].n_unique()} stocks, "
          f"{lg['ts'].n_unique()} dumps; qty != 0 {np.mean(lg['qty'].to_numpy() != 0):.3f}, base_qty != 0 {np.mean(lg['base_qty'].to_numpy() != 0):.3f}")
    per = lg.group_by("sid").agg(pl.col("qty").n_unique().alias("nq"), (pl.col("qty") == pl.col("base_qty")).mean().alias("at_base"))
    print(f"  stocks whose qty changes during the day {np.mean(per['nq'].to_numpy() > 1):.3f}; share of rows with qty == base_qty "
          f"{lg.select((pl.col('qty') == pl.col('base_qty')).mean()).item():.3f}")
    f = state / f"date={d}.parquet"
    if not f.exists():
        print(f"  no {f}")
        return
    dec = pl.read_parquet(f).filter(pl.col("kind") == "decision").select("sid", "src_epoch", "fa", "pos", "pp", "executp", "asq")
    dec = dec.with_columns(pl.col("sid").cast(pl.Int64), pl.col("src_epoch").cast(pl.Int64)).drop_nulls("src_epoch").sort("sid", "src_epoch")
    j = dec.join_asof(lg.select("sid", "ts", *[c for c in LOG if c in lg.columns]), left_on="src_epoch", right_on="ts", by="sid",
                      strategy="backward").drop_nulls("qty")
    print(f"  decisions with a log row at or before them: {j.height:,} of {dec.height:,}")
    if j.is_empty():
        return
    j = j.with_columns((pl.col("base_qty") - pl.col("qty")).alias("gap"), (pl.col("qty") - pl.col("base_qty")).alias("ngap"))
    for y in ("pos", "executp"):
        v = j[y].to_numpy()
        res = []
        for c in ("qty", "yesterday_qty", "sellable_qty", "base_qty", "gap", "ngap"):
            x = j[c].to_numpy()
            res.append(f"{c} {np.mean(v == x):.3f}/{np.mean(np.abs(v - x) <= 100):.3f}/r {corr(v, x):+.3f}")
        print(f"  {y} == (exact / within 100 / corr): " + "; ".join(res))
    ex, pos, sell, gap, fa = (j[c].to_numpy() for c in ("executp", "pos", "sellable_qty", "gap", "fa"))
    capped = np.where(pos < 0, np.maximum(pos, -sell), pos)
    nz = ex != 0
    print(f"  executp == pos with sells capped at sellable_qty {np.mean(ex == capped):.3f}; on executp != 0: sign(executp) == sign(base - qty) "
          f"{np.mean(np.sign(ex[nz]) == np.sign(gap[nz])):.3f}, == sign(fa) {np.mean(np.sign(ex[nz]) == np.sign(fa[nz])):.3f}")
    print(f"  prediction vs holding gap: corr(fa, base - qty) {corr(fa, gap):+.3f}; corr(fa, pos) {corr(fa, pos):+.3f}; "
          f"rows with base - qty != 0 {np.mean(gap != 0):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poslog", type=Path, default=Path("/work/crucible_data/chrono_poslog"), help="<dir>/<YYYYMMDD>/... position log csvs")
    ap.add_argument("--state", type=Path, default=Path("/work/crucible_data/pm_signals/state"))
    ap.add_argument("--dates", required=True)
    a = ap.parse_args()
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        day(a.poslog, a.state, d)


if __name__ == "__main__":
    main()
