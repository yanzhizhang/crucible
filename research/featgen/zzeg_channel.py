r"""zzeg candidates from PM's own channel: zzeg.c = the 4-day high / low of the snapshot high / low (R109: N = 4 matches
PM's .jz on ~100% of stocks, every day), zzeg.c.span = (high + low) / 2; zzeg.c.rEma ~ an EMA of the span.

At each scoring row: H = max(the previous 3 trading days' highs, today's running high), L likewise, span = (H + L) / 2,
px = the last snapshot price (by exchange time, as ratio_extra). Candidates: px / span, span / px, (px - span) / span,
(px - L) / (H - L), px / H, px / L, H / L, (H - L) / span. Daily highs / lows of earlier days come from the quotation
stream, or from the trades when a day has no quotation (the April NonConvex days). Output <out>/zzeg_<D>.parquet.

    python research/featgen/zzeg_channel.py --date 20260430 --x data/reports/stage6_r99_0430/x.parquet --raw /work/crucible_data/feitu_raw --out data/side_r110
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ratio_extra import asof, mark_ns  # noqa: E402

DATA = Path(__import__("os").environ.get("CRUCIBLE_DATA", "/work/crucible_data"))


CACHE = DATA / "cache" / "daily_hl"  # one parquet per day and source: computed once, reused by every later round


def day_hl(d: str, sids: list[int], raws: list[Path]) -> pl.DataFrame | None:
    """The day's high / low per stock (all stocks of the stream, cached), filtered to sids."""
    for r in raws:
        f = CACHE / f"date={d}" / f"{r.name}.parquet"
        if f.exists():
            return pl.read_parquet(f).filter(pl.col("symbol_id").is_in(sids))
        hl = _day_hl(d, r)
        if hl is not None:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                hl.write_parquet(f)
            except OSError:  # read-only data root: just do without the cache
                pass
            return hl.filter(pl.col("symbol_id").is_in(sids))
    return None


def _day_hl(d: str, r: Path) -> pl.DataFrame | None:
    """One source's day high / low per stock: quotation high / low fields, else traded prices."""
    q = list((r / f"kind=quotation/date={d}").glob("part-*.parquet"))
    if q:
        return (pl.scan_parquet(q).select("symbol_id", "high", "low", "price").filter(pl.col("price") > 0)
                .group_by("symbol_id").agg(pl.col("high").max().alias("h"), pl.col("low").filter(pl.col("low") > 0).min().alias("l")).collect())
    t = list((r / f"kind=transaction/date={d}").glob("part-*.parquet"))
    if t:
        return (pl.scan_parquet(t).select("symbol_id", "price", "trade_type").filter((pl.col("price") > 0) & (pl.col("trade_type") == 1))
                .group_by("symbol_id").agg(pl.col("price").max().alias("h"), pl.col("price").min().alias("l")).collect())
    return None


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True, help="today's quotation stream")
    ap.add_argument("--hist", default="dysl2_raw,feitu_raw", help="where earlier days' highs / lows are looked for, in order")
    ap.add_argument("--n", type=int, default=4, help="channel length in trading days, today included")
    ap.add_argument("--out", type=Path, default=Path("data/side_zzeg"))
    a = ap.parse_args()
    d = a.date
    key = pl.read_parquet(a.x, columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64))
    sids = key["sid"].unique().cast(pl.Int32).to_list()
    raws = [DATA / s for s in a.hist.split(",")]
    days = sorted({p.name[5:] for r in raws for k in ("quotation", "transaction") for p in (r / f"kind={k}").glob("date=*") if p.name[5:] < d})
    prev = []
    for dd in reversed(days):
        hl = day_hl(dd, sids, raws)
        if hl is not None and hl.height > 0.5 * len(sids):
            prev.append((dd, hl))
        if len(prev) == a.n - 1:
            break
    print(f"{d}: previous days used {[x for x, _ in prev]}")
    ph = pl.concat([hl for _, hl in prev]).group_by("symbol_id").agg(pl.col("h").max().alias("ph"), pl.col("l").min().alias("pl"))
    marks = np.sort(key["mark"].unique().to_numpy())
    mt = pl.DataFrame({"mark": marks, "t": mark_ns(d, marks)})
    q = pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")), columns=["symbol_id", "time", "high", "low", "price"])
    q = q.filter(pl.col("symbol_id").is_in(sids) & (pl.col("price") > 0)).with_columns(pl.col("symbol_id").cast(pl.Int64).alias("sid")).drop("symbol_id")
    s = asof(q, ["sid"], key.join(mt, on="mark")).join(ph.with_columns(pl.col("symbol_id").cast(pl.Int64).alias("sid")).drop("symbol_id"), on="sid", how="left")
    H = pl.max_horizontal("high", "ph")
    L = pl.min_horizontal(pl.when(pl.col("low") > 0).then(pl.col("low")), "pl")
    S = (H + L) / 2
    P = pl.col("price")
    s = s.select("sid", "mark", (P / S).alias("eg.px/span"), (S / P).alias("eg.span/px"), ((P - S) / S).alias("eg.(px-span)/span"),
                 ((P - L) / (H - L)).alias("eg.(px-L)/(H-L)"), (P / H).alias("eg.px/H"), (P / L).alias("eg.px/L"), (H / L).alias("eg.H/L"),
                 ((H - L) / S).alias("eg.(H-L)/span"))
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"zzeg_{d}.parquet"
    s.write_parquet(dst)
    v = s["eg.px/span"]
    print(f"{d}: {s.height:,} rows, px/span found {v.is_not_null().mean():.3f}, quantiles " + " ".join(f"{v.quantile(x):.4f}" for x in (0.01, 0.1, 0.5, 0.9, 0.99)) + f" -> {dst}")


if __name__ == "__main__":
    main()
