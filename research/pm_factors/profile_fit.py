r"""Stage 4: the cross-day memory of a samplerS column -- half-life scan of an EMA over our daily basics.

For the columns ``basic_match`` tied to a basic quantity with a high within-stock correlation but a ratio away from 1
(ZZMS n / q, ZZAL an / bn, ZZQO q, ZZXC q / n, ZZTS q, ZZHL n), the PM value of day d is modelled as
``k x sum_j w_j ours[d - j]`` with ``w_j = 0.5 ** (j / h)`` over the earlier days (``incl``: j from 0, the day itself
counted; ``excl``: j from 1). Per (pair, h, incl, bar shift): rank correlation, within-stock correlation, the median
ratio ``k`` and the IQR of log(PM / ours) across cells (0 = the same shape up to k) -- the half-life PM uses is where the
IQR bottoms out. Test days need at least ``--min-prev`` earlier basics days.

    python research/pm_factors/profile_fit.py --dates all --pairs ZZMS:n:cum_amt,ZZAL:an:atot_amt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from basic_match import BASICS
from rank_candidates import pm_frame

PAIRS = ("ZZMS:n:cum_amt", "ZZMS:q:cum_vol", "ZZAL:an:atot_amt", "ZZAL:bn:btot_amt", "ZZQO:q:vol", "ZZXC:q:vol", "ZZXC:n:amt",
         "ZZTS:q:amt", "ZZHL:n:bno", "ZZQO:q:ordvol_sum")
HALF = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="all", help="PM days to test, or all with enough history")
    ap.add_argument("--pairs", default=",".join(PAIRS), help="file:V:basic, comma-separated")
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--min-prev", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4/profile_fit.csv"))
    a = ap.parse_args()
    days = sorted(p.name.split("=")[1] for p in a.basics.glob("date=*"))
    pairs = [tuple(x.split(":")) for x in a.pairs.split(",") if x]
    need = {(f, b) for f, _, b in pairs}
    cache: dict[tuple[str, str], pl.DataFrame] = {}

    def table(day: str, kind: str) -> pl.DataFrame | None:
        if (day, kind) not in cache:
            g = a.basics / f"date={day}" / f"{kind}.parquet"
            cols = ["symbol_id", "minute", *sorted({b for f, b in need})]
            cache[(day, kind)] = (pl.read_parquet(g, columns=[c for c in cols if c in pl.read_parquet_schema(g)])
                                  .rename({"symbol_id": "code"}).with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
                                  ) if g.exists() else None
        return cache[(day, kind)]

    pm_days = sorted({f.stem.rsplit("_", 1)[1] for f in a.exports.glob("*.parquet")})
    dates = [d for d in pm_days if len([x for x in days if x < d]) >= a.min_prev] if a.dates == "all" else a.dates.split(",")
    print(f"test days {dates} (basics days {days[0]}..{days[-1]}, {len(days)})")
    rows = []
    for fam, V, basic in pairs:
        for d in dates:
            f = a.exports / f"{fam}_{d}.parquet"
            if not f.exists():
                continue
            pm, vals, kind = pm_frame(f, None, None, None, None, None)
            pm = pm.filter(pl.col("V") == V).select("code", "minute", pl.col(vals[0]).alias("x"))
            if kind != "time":
                continue
            step = int(pm.select(pl.col("minute").unique().sort().diff().drop_nulls().min()).item() or 1)
            tab = "m5" if step >= 5 else "m1"
            prev = [x for x in days if x < d][-40:]
            hist = [(x, table(x, tab)) for x in [*prev, d]]
            hist = [(x, t) for x, t in hist if t is not None and basic in t.columns]
            if len(hist) < a.min_prev:
                continue
            for incl in (True, False):
                for h in HALF:
                    parts = []
                    for x, t in hist:
                        if x == d and not incl:
                            continue
                        j = len([y for y, _ in hist if y > x])  # days back from d
                        parts.append(t.select("code", "minute", pl.col(basic).alias("v")).with_columns(pl.lit(0.5 ** (j / h)).alias("w")))
                    ours = (pl.concat(parts).filter(pl.col("v").is_finite())
                            .group_by("code", "minute").agg((pl.col("v") * pl.col("w")).sum() / pl.col("w").sum()))
                    for sh in (0, step, -step):
                        jn = pm.with_columns(pl.col("minute") + sh).join(ours, on=["code", "minute"]).filter(pl.col("x").is_finite() & (pl.col("v") > 0) & (pl.col("x") > 0))
                        if jn.height < 100:
                            continue
                        lr = np.log(jn["x"].to_numpy() / jn["v"].to_numpy())
                        w = jn.group_by("code").agg(pl.corr("x", "v").alias("c"))["c"]
                        rows.append({"file": fam, "V": V, "basic": basic, "day": d, "h": h, "incl": incl, "shift": sh, "n": jn.height,
                                     "rho": float(jn.select(pl.corr("x", "v", method="spearman")).item()), "within": float(w.drop_nans().drop_nulls().mean()),
                                     "k": float(np.exp(np.median(lr))), "lr_iqr": float(np.quantile(lr, 0.75) - np.quantile(lr, 0.25))})
            print(f"{fam} {V} ~ {basic} on {d}: {len(hist)} days of history", flush=True)
    if not rows:
        print("nothing fitted")
        return
    t = pl.DataFrame(rows)
    agg = (t.group_by("file", "V", "basic", "h", "incl", "shift").agg(pl.len().alias("days"), pl.col("rho").mean(), pl.col("within").mean(),
                                                                   pl.col("k").median(), pl.col("lr_iqr").mean())
           .sort("file", "V", "basic", "lr_iqr"))
    agg.write_csv(a.out)
    with pl.Config(tbl_rows=300, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print("\n== half-life scan: best 6 per column by log-ratio IQR (0 = same shape up to k); incl = the day itself in the EMA")
        print(agg.group_by("file", "V", "basic", maintain_order=True).head(6))


if __name__ == "__main__":
    main()
