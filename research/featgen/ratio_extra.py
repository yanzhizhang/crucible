r"""Price-ratio candidates at the scoring rows (zzeg's training range is [0.867, 1.097], thresholds 0.985-1.030 around 1).

R98 found zzeg never NaN in training, all its thresholds a ratio near 1, and none of 397 candidates (all rank / calibrated)
helping on both days, while a cross-sectionally constant fill helped 1112. So: plain ratios, uncalibrated, of
* each index (CSI 300 / 500 / 1000, SSE 50, SSE composite, SZSE component, ChiNext): last / pre_close, last / open,
  last / high, last / low, last / last 1, 5, 15, 30 minutes before -- the same value for every stock at a mark;
* each stock: last / pre_close, mid / pre_close, last / open, last / vwap, mid / vwap, last / high, last / low, and the
  stock's last / pre_close over CSI 300's (relative strength).
Values are as of the mark (the last snapshot with exchange time <= the mark). Output <out>/ratio_<D>.parquet (sid, mark, ...).

    python research/featgen/ratio_extra.py --date 20261112 --x data/reports/stage6_r99_1112/x.parquet --out data/side_r100
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from decode_tdf_day import CST_NS, SSE, SZSE  # noqa: E402

INDEX = {"i300": (SSE, 300), "i905": (SSE, 905), "i852": (SSE, 852), "i016": (SSE, 16), "ish": (SSE, 1),
         "isz": (SZSE, 399001), "icy": (SZSE, 399006)}


def mark_ns(d: str, marks: np.ndarray) -> np.ndarray:
    """UTC epoch ns of each trading-clock mark (seconds since 09:30, 7200 = 11:30 = 13:00)."""
    day0 = int(dt.datetime.strptime(d, "%Y%m%d").replace(tzinfo=dt.timezone.utc).timestamp()) * 10**9
    sec = np.where(marks <= 7200, 34200 + marks, 46800 + marks - 7200)
    return day0 + sec.astype(np.int64) * 10**9 - CST_NS


def asof(df: pl.DataFrame, by: list[str], grid: pl.DataFrame) -> pl.DataFrame:
    """The last row with time <= each grid time, per key."""
    return grid.sort("t").join_asof(df.sort("time"), left_on="t", right_on="time", by=by or None, strategy="backward")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/dysl2_raw"))
    ap.add_argument("--index-raw", type=Path, default=None, help="where the index stream is, if not under --raw (e.g. pmmp_raw: PM's own mp)")
    ap.add_argument("--out", type=Path, default=Path("data/side_ratio"))
    a = ap.parse_args()
    d = a.date
    key = pl.read_parquet(a.x, columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64))
    marks = np.sort(key["mark"].unique().to_numpy())
    mt = pl.DataFrame({"mark": marks, "t": mark_ns(d, marks)})

    parts = list((a.index_raw or a.raw).glob(f"kind=index/date={d}/part-*.parquet"))
    if not parts:
        sys.exit(f"{d}: no index parts under {a.index_raw or a.raw}")
    ix = pl.read_parquet(parts, columns=["symbol_id", "market_id", "time", "last_price", "open_price", "high_price", "low_price", "pre_close_price"])
    ix = ix.filter(pl.col("last_price") > 0).unique(["symbol_id", "market_id", "time"])  # PM's mp repeats an index across pool files
    print(f"  index stream: {len(parts)} parts, {ix.height:,} rows, codes {sorted(ix['symbol_id'].unique().to_list())[:12]}")
    per_mark = mt
    for nm, (mk, code) in INDEX.items():
        g = ix.filter((pl.col("market_id") == mk) & (pl.col("symbol_id") == code)).drop("symbol_id", "market_id")
        if g.is_empty():
            print(f"  {nm}: no rows")
            continue
        cols = []
        for lag in (0, 60, 300, 900, 1800):
            gg = asof(g, [], mt.with_columns((pl.col("t") - lag * 10**9).alias("t")).rename({"t": "t"}))
            cols.append(gg.select("mark", *[pl.col(c).alias(f"{c}@{lag}") for c in ("last_price", "open_price", "high_price", "low_price", "pre_close_price")]))
        w = cols[0]
        for c in cols[1:]:
            w = w.join(c, on="mark", how="left")
        last = pl.col("last_price@0")
        exprs = [(last / pl.col("pre_close_price@0")).alias(f"{nm}.last/pre"), (last / pl.col("open_price@0")).alias(f"{nm}.last/open"),
                 (last / pl.col("high_price@0")).alias(f"{nm}.last/high"), (last / pl.col("low_price@0")).alias(f"{nm}.last/low")]
        exprs += [(last / pl.col(f"last_price@{lag}")).alias(f"{nm}.last/last{lag // 60}m") for lag in (60, 300, 900, 1800)]
        per_mark = per_mark.join(w.select("mark", *exprs), on="mark", how="left")
    rows = key.join(per_mark.drop("t"), on="mark", how="left")

    q = pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")),
                        columns=["symbol_id", "time", "pre_close", "open", "high", "low", "price", "bid_px", "ask_px", "total_volume", "total_amount"])
    q = q.filter(pl.col("symbol_id").is_in(key["sid"].unique().cast(pl.Int32).implode()) & (pl.col("price") > 0)).with_columns(
        pl.col("symbol_id").cast(pl.Int64).alias("sid"), pl.col("bid_px").cast(pl.List(pl.Float64)).list.first().alias("b1"),  # Array(10) on 97, List here
        pl.col("ask_px").cast(pl.List(pl.Float64)).list.first().alias("a1")).drop("bid_px", "ask_px")
    grid = key.join(mt, on="mark")
    s = asof(q, ["sid"], grid)
    mid = pl.when((pl.col("b1") > 0) & (pl.col("a1") > 0)).then((pl.col("b1") + pl.col("a1")) / 2).otherwise(pl.col("price"))
    vwap = pl.when(pl.col("total_volume") > 0).then(pl.col("total_amount") / pl.col("total_volume")).otherwise(None)
    s = s.select("sid", "mark", (pl.col("price") / pl.col("pre_close")).alias("s.last/pre"), (mid / pl.col("pre_close")).alias("s.mid/pre"),
                 (pl.col("price") / pl.col("open")).alias("s.last/open"), (pl.col("price") / vwap).alias("s.last/vwap"), (mid / vwap).alias("s.mid/vwap"),
                 (pl.col("price") / pl.col("high")).alias("s.last/high"), (pl.col("price") / pl.col("low")).alias("s.last/low"))
    rows = rows.join(s, on=["sid", "mark"], how="left")
    if "i300.last/pre" in rows.columns:
        rows = rows.with_columns((pl.col("s.last/pre") / pl.col("i300.last/pre")).alias("s.rel300"))
    rows = rows.with_columns([pl.lit(c).alias(f"const{c}") for c in (0.99, 0.995, 0.998, 1.0, 1.002, 1.005, 1.01)])  # yardsticks: one value for all
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"ratio_{d}.parquet"
    rows.write_parquet(dst)
    found = rows["s.last/pre"].is_not_null().mean()
    print(f"{d}: {rows.height:,} rows, {len(rows.columns) - 2} ratios, stock rows found {found:.3f}; "
          + " ".join(f"{c} {rows[c].quantile(.05):.4f}..{rows[c].quantile(.95):.4f}" for c in ("i300.last/pre", "s.last/pre", "s.mid/vwap") if c in rows.columns)
          + f" -> {dst}")


if __name__ == "__main__":
    main()
