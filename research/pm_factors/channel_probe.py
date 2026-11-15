r"""What window PM's zzeg.c (and zzxu.upper / lower) channel spans: N-day high / low against the .jz values.

The .jz holds per stock zzeg.c.high / .low (price units, e.g. 600519 1480.6 / 1352.98 on 1104, unchanged to 1106, 1449.99
on 1112) and zzeg.c.span = (high + low) / 2 exactly -- a multi-day price channel. Per day of the --jz files: our daily
high / low per stock (max / min of the snapshot high / low fields; DataYes where it has the day, else feitu_raw), the
rolling max / min over the last N trading days ending at D (N = 1..--max-n), and for each N the share of stocks whose
channel equals PM's (|diff| <= 0.005). Also N-day rolling max / min of the closes, in case the channel is on closes.
Read-only.

    python research/pm_factors/channel_probe.py --days 20261104,20261105,20261106,20261112,20261113
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/work/crucible_data")
JZ = {"20261104": "/data/prod/h/HS300/p1_intrading_dcp", "20261105": "/data/prod/h/HS300/p1_intrading_dcp",
      "20261106": "/data/prod/h/HS300/p1_intrading_dcp"}


def daily(d: str, sids: list[int]) -> pl.DataFrame | None:
    """One row per stock: the day's high, low (snapshot fields) and last price."""
    for src in ("dysl2_raw", "feitu_raw"):
        parts = list((DATA / src).glob(f"kind=quotation/date={d}/part-*.parquet"))
        if parts:
            q = pl.scan_parquet(parts).select("symbol_id", "time", "high", "low", "price").filter(pl.col("symbol_id").is_in(sids) & (pl.col("price") > 0))
            return q.group_by("symbol_id").agg(pl.col("high").max().alias("h"), pl.col("low").filter(pl.col("low") > 0).min().alias("l"),
                                               pl.col("price").sort_by("time").last().alias("c")).with_columns(pl.lit(d).alias("date")).collect()
    return None


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--max-n", type=int, default=30)
    a = ap.parse_args()
    ds = a.days.split(",")
    states = {}
    for d in ds:
        p = Path(JZ.get(d, DATA / "seed_probe" / "HS300")) / f"{d}.jz"
        if p.exists():
            states[d] = {int(s): kv for s, kv in json.loads(gzip.open(p).read()).items()}
    sids = sorted({s for st in states.values() for s in st})
    have = sorted({p.name[5:] for src in ("dysl2_raw", "feitu_raw") for p in (DATA / src / "kind=quotation").glob("date=*")})
    hist = [x for x in (daily(d, sids) for d in have if d <= max(ds)) if x is not None]
    h = pl.concat(hist).sort("symbol_id", "date")
    print(f"daily bars: {h['date'].n_unique()} days {h['date'].min()}..{h['date'].max()}, {h['symbol_id'].n_unique()} stocks")
    for d, st in states.items():
        days = sorted(x for x in h["date"].unique().to_list() if x <= d)
        print(f"== {d}: {len(days)} days of history up to it")
        for key_h, key_l in (("zzeg.c.high", "zzeg.c.low"), ("zzxu.upper", "zzxu.lower")):
            pm = pl.DataFrame({"symbol_id": list(st), "ph": [kv.get(key_h) for kv in st.values()], "pl": [kv.get(key_l) for kv in st.values()]},
                              schema={"symbol_id": pl.Int64, "ph": pl.Float64, "pl": pl.Float64})
            res = []
            for n in range(1, min(a.max_n, len(days)) + 1):
                w = h.filter(pl.col("date").is_in(days[-n:])).group_by("symbol_id").agg(
                    pl.col("h").max(), pl.col("l").min(), pl.col("c").max().alias("cmax"), pl.col("c").min().alias("cmin"))
                j = pm.join(w.with_columns(pl.col("symbol_id").cast(pl.Int64)), on="symbol_id")
                eh = float(((j["ph"] - j["h"]).abs() <= 0.005).mean())
                el = float(((j["pl"] - j["l"]).abs() <= 0.005).mean())
                ec = float((((j["ph"] - j["cmax"]).abs() <= 0.005) & ((j["pl"] - j["cmin"]).abs() <= 0.005)).mean())
                res.append(f"N{n}: hi {eh:.2f} lo {el:.2f} close-channel {ec:.2f}")
            print(f"  {key_h}/{key_l}: " + " | ".join(res))


if __name__ == "__main__":
    main()
