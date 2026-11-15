r"""Candidates built the way PM's settled state columns are: ratios of PM's engine state keys and the price.

The settled columns that read the state (baseline_r41c) are plain ratios of keys and the last price: zzup = zzup.pf /
zzup.ps, zzvx.a = zzvx.trDEmaD / px, zzxu.a = zzxu.a / zzxu.low, zzxu.b = (px - zzxu.a) / zzxu.last. Open columns of the
same families (zzba with zzba.p / zzba.r, zzmh with zzmh.hm, zzab2 with zzab2.r, zzxo with zzxo.last, zzgh ...) likely
follow the same grammar. Per key K: K / px, (px - K) / K, (px - K) / px; per pair of keys of one family: K1 / K2,
(K1 - K2) / K2, (px - K1) / K2. px = the last snapshot price at the mark by local arrival time. Output
<out>/state_<D>.parquet (sid, mark, ...), for implied_scan --extra with raw transforms.

    python research/featgen/state_formulas.py --date 20261112 --engine data/engine/marks_20261112.parquet --out data/side_r107
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ratio_extra import asof, mark_ns  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--engine", type=Path, required=True, help="the engine's marks parquet (sid, mark, keys, <ema>.sum / .wgt)")
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/dysl2_raw"))
    ap.add_argument("--out", type=Path, default=Path("data/side_state"))
    a = ap.parse_args()
    d = a.date
    e = pl.read_parquet(a.engine).with_columns(pl.col("sid").cast(pl.Int64))
    if "date" in e.columns:
        e = e.filter(pl.col("date") == int(d)).drop("date")
    keys: dict[str, pl.Expr] = {}
    for c in e.columns:
        if c in ("sid", "mark") or c.endswith(".wgt") or not e.schema[c].is_numeric():
            continue
        if c.endswith(".sum") and f"{c[:-4]}.wgt" in e.columns:  # an EMA: sum / wgt
            keys[c[:-4]] = pl.when(pl.col(f"{c[:-4]}.wgt") != 0).then(pl.col(c) / pl.col(f"{c[:-4]}.wgt"))
        elif not c.endswith(".sum"):
            keys[c] = pl.col(c)
    print(f"{d}: {len(keys)} state keys: {sorted(keys)}")
    k = e.select("sid", "mark", *[x.alias(n) for n, x in keys.items()])
    marks = np.sort(k["mark"].unique().to_numpy())
    mt = pl.DataFrame({"mark": marks, "t": mark_ns(d, marks)})
    q = pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")), columns=["symbol_id", "spider_ts", "price"])
    q = q.filter(pl.col("symbol_id").is_in(k["sid"].unique().cast(pl.Int32).implode()) & (pl.col("price") > 0)).select(
        pl.col("symbol_id").cast(pl.Int64).alias("sid"), pl.col("spider_ts").alias("time"), "price")
    px = asof(q, ["sid"], k.select("sid", "mark").unique().join(mt, on="mark")).select("sid", "mark", pl.col("price").alias("px"))
    k = k.join(px, on=["sid", "mark"], how="left")
    P = pl.col("px")
    out = []
    for n in keys:
        K = pl.col(n)
        out += [(K / P).alias(f"{n}/px"), ((P - K) / K).alias(f"(px-{n})/{n}"), ((P - K) / P).alias(f"(px-{n})/px")]
    fams: dict[str, list[str]] = {}
    for n in keys:
        fams.setdefault(n.split(".")[0], []).append(n)
    for ns in fams.values():
        for k1, k2 in itertools.permutations(ns, 2):
            K1, K2 = pl.col(k1), pl.col(k2)
            out += [(K1 / K2).alias(f"{k1}/{k2}"), ((K1 - K2) / K2).alias(f"({k1}-{k2})/{k2}"), ((P - K1) / K2).alias(f"(px-{k1})/{k2}")]
    res = k.select("sid", "mark", *out)
    res = res.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).alias(c) for c in res.columns[2:]])
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"state_{d}.parquet"
    res.write_parquet(dst)
    print(f"{d}: {res.height:,} rows, {len(res.columns) - 2} candidates, px found {k['px'].is_not_null().mean():.3f} -> {dst}")


if __name__ == "__main__":
    main()
