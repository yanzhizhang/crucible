"""Convert one source-day of tickaudit's lake (``/work/tickaudit_lake/tick``) into crucible's feitu raw Parquet layout.

tickaudit decodes every vendor feed into the exchanges' own records (fixed-point int64, exchange field names); this is the
thin adapter that turns those into exactly the columns, dtypes and column order ``decode_tdf_day.py`` writes, so
``catalog.py`` (``CRUCIBLE_SOURCE=ta_<source>``) and everything built on it read them unchanged::

    in : <lake>/tick/<table>/day=YYYYMMDD/source=<src>/data.parquet     sse_tick sse_snap sse_index szse_order szse_trade szse_snap szse_index
    out: <out>/kind=<order|transaction|quotation|index>/date=YYYYMMDD/part-<SH|SZ>-000.parquet  (+ _manifest, _quality)

Mapping (tickaudit ``cache/Columns.h`` + ``sse|szse/Records.h|Snapshot.h``; conventions as ``decode_tdf_day``):

* rows: ``stream`` ``*_stock`` only, A-share codes (6[08]xxxx SH, 00/30xxxx SZ); index tables keep ``INDEXES``.
* clocks: ``time`` = tickTimeNs / transactTimeNs / timeNs; ``spider_ts`` = ``nano_ts`` = ``server_ts`` = localNs, or
  ``time`` when the source has none (localNs 0).
* SSE UA5803: type A / D -> ``order`` (update_type 1 / 2, order_id = the side's own order number), T -> ``transaction``
  (buy/sell_seq_id = buyOrderNo / sellOrderNo, trade_type 1), S rows dropped; flag B / S / N -> dir 1063 / 1550 / 0;
  seq_id = index_id = bizIndex. price yuan x1e3, qty x1e3.
* SZSE 300192 / 300191: applSeqNum -> order_id = seq_id = index_id; side 1 / 2 -> dir; ordType 1 / U / else -> order_type
  1 / 3 / 2; execType 4 -> trade_type 2 with the side of the set leg, else 1 with dir = the larger ApplSeqNum leg.
  price yuan x1e4, qty x1e2.
* snapshots: SSE prices x1e3, qty x1e3, value x1e5; SZSE preClose x1e4, other prices x1e6, qty x1e2, value x1e4;
  status = first character's ASCII code; INT64_MIN (absent) -> 0; SZSE's no-limit sentinel -> 0 (= no band data).
* index: SSE x1e5 levels, volume x1e5 in hundreds; SZSE x1e6 levels, volume x1e2 (volume unit not yet checked vs feitu).

Prices become ``int / 10**k`` (the nearest double to the exchange's decimal); a vendor float that is not the nearest
double (6.4399999999999995) shows up as a last-bit difference in ``--compare``.

    python research/from_tickaudit.py --source datayes --date 20260915                      # -> $DATA/ta_datayes_raw
    python research/from_tickaudit.py --source datayes --date 20260915 --compare dysl2_raw  # field-by-field vs our decode
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decode_tdf_day import BUY, LEVELS, SELL, SSE, SZSE, ROW_GROUP, SCHEMAS, quality  # noqa: E402

SH_INDEX = [1, 16, 300, 852, 905]  # decode_tdf_day.INDEXES as codes: 000001 000016 000300 000852 000905 .SH
SZ_INDEX = [399001, 399006, 399300]

DATA = Path("/work/crucible_data")
LAKE = Path("/work/tickaudit_lake")
ABSENT = -(2**63)
SZ_NO_LIMIT = 999999999_9999 * 100  # 999999999.9999 yuan x1e6
TABLES = {("order", SSE): "sse_tick", ("transaction", SSE): "sse_tick", ("quotation", SSE): "sse_snap",
          ("index", SSE): "sse_index", ("order", SZSE): "szse_order", ("transaction", SZSE): "szse_trade",
          ("quotation", SZSE): "szse_snap", ("index", SZSE): "szse_index"}


def v(col: str, scale: float | None = None) -> pl.Expr:
    """An int64 lake column with INT64_MIN as 0, optionally divided to a float."""
    e = pl.when(pl.col(col) == ABSENT).then(0).otherwise(pl.col(col))
    return (e / scale).cast(pl.Float64) if scale else e.cast(pl.Int64)


def clocks(tcol: str) -> list[pl.Expr]:
    t = pl.col(tcol).cast(pl.Int64)
    loc = pl.when(pl.col("localNs") > 0).then(pl.col("localNs")).otherwise(t).cast(pl.Int64)
    return [t.alias("time"), loc.alias("spider_ts"), loc.alias("nano_ts"), loc.alias("server_ts")]


def head(market: int) -> list[pl.Expr]:
    return [pl.lit(0, pl.Int32).alias("shm_key"), pl.col("securityId").cast(pl.Int32).alias("symbol_id"),
            pl.lit(market, pl.Int16).alias("market_id"), pl.lit(1, pl.Int32).alias("symbol_type")]


def stocks(lf: pl.LazyFrame, market: int) -> pl.LazyFrame:
    c = pl.col("securityId")
    keep = (c // 10000).is_in([60, 68]) if market == SSE else ((c // 10000).is_in([0, 30]) & (c > 0))
    return lf.filter(pl.col("stream").str.ends_with("_stock") & keep)


def side(e: pl.Expr, buy: str, sell: str) -> pl.Expr:
    return pl.when(e == buy).then(BUY).when(e == sell).then(SELL).otherwise(0).cast(pl.Int16)


def convert(kind: str, market: int, lf: pl.LazyFrame) -> pl.LazyFrame:
    """One lake table -> feitu columns of ``kind`` (unordered; the writer casts and orders)."""
    if kind == "index":
        sc = 1e5 if market == SSE else 1e6
        vol = v("totalVolume", 1e5 / 100) if market == SSE else v("totalVolume", 1e2)
        return lf.filter(pl.col("securityId").is_in(SH_INDEX if market == SSE else SZ_INDEX)).select(
            *head(market)[:3], *clocks("timeNs"), v("lastIdx", sc).alias("last_price"), v("openIdx", sc).alias("open_price"),
            v("highIdx", sc).alias("high_price"), v("lowIdx", sc).alias("low_price"), v("closeIdx", sc).alias("close_price"),
            v("preCloseIdx", sc).alias("pre_close_price"), vol.round().cast(pl.Int64).alias("total_volume"),
            v("totalValue", 1e5 if market == SSE else 1e4).alias("total_amount"),
            pl.col("securityId").cast(pl.String).str.zfill(6).alias("symbol_str"))
    lf = stocks(lf, market)
    if market == SSE and kind in ("order", "transaction"):
        px, qty = v("price", 1e3), (v("qty") // 1000).cast(pl.Int64)
        seq = pl.col("bizIndex").cast(pl.Int64)
        flag = side(pl.col("flag"), "B", "S")
        if kind == "order":
            return lf.filter(pl.col("type").is_in(["A", "D"])).select(
                *head(market), *clocks("tickTimeNs"), px.alias("price"), qty.alias("volume"), flag.alias("dir"),
                pl.when(pl.col("flag") == "B").then(pl.col("buyOrderNo")).otherwise(pl.col("sellOrderNo")).cast(pl.Int64).alias("order_id"),
                pl.col("channel").cast(pl.Int32).alias("channel_id"), seq.alias("seq_id"), seq.alias("index_id"),
                pl.lit(0, pl.Int8).alias("order_type"),
                pl.when(pl.col("type") == "A").then(1).otherwise(2).cast(pl.Int8).alias("update_type"))
        return lf.filter(pl.col("type") == "T").select(
            *head(market), *clocks("tickTimeNs"), px.alias("price"), qty.alias("volume"), flag.alias("dir"),
            pl.col("channel").cast(pl.Int32).alias("channel_id"), seq.alias("index_id"), seq.alias("seq_id"),
            pl.col("buyOrderNo").cast(pl.Int64).alias("buy_seq_id"), pl.col("sellOrderNo").cast(pl.Int64).alias("sell_seq_id"),
            pl.lit(1, pl.Int8).alias("trade_type"))
    if kind == "order":
        seq = pl.col("applSeqNum").cast(pl.Int64)
        return lf.select(
            *head(market), *clocks("transactTimeNs"), v("price", 1e4).alias("price"), (v("orderQty") // 100).alias("volume"),
            side(pl.col("side"), "1", "2").alias("dir"), seq.alias("order_id"), pl.col("channel").cast(pl.Int32).alias("channel_id"),
            seq.alias("seq_id"), seq.alias("index_id"),
            pl.when(pl.col("ordType") == "1").then(1).when(pl.col("ordType") == "U").then(3).otherwise(2).cast(pl.Int8).alias("order_type"),
            pl.lit(1, pl.Int8).alias("update_type"))
    if kind == "transaction":
        seq = pl.col("applSeqNum").cast(pl.Int64)
        bid, ask = pl.col("bidApplSeqNum").cast(pl.Int64), pl.col("offerApplSeqNum").cast(pl.Int64)
        cancel = pl.col("execType") == "4"
        return lf.select(
            *head(market), *clocks("transactTimeNs"), v("lastPx", 1e4).alias("price"), (v("lastQty") // 100).alias("volume"),
            pl.when(cancel).then(pl.when(bid != 0).then(BUY).otherwise(SELL))
            .otherwise(pl.when(bid > ask).then(BUY).otherwise(SELL)).cast(pl.Int16).alias("dir"),
            pl.col("channel").cast(pl.Int32).alias("channel_id"), seq.alias("index_id"), seq.alias("seq_id"),
            bid.alias("buy_seq_id"), ask.alias("sell_seq_id"), pl.when(cancel).then(2).otherwise(1).cast(pl.Int8).alias("trade_type"))
    # quotation
    if market == SSE:
        ps = pre = 1e3
        qd, val = 1000, 1e5
    else:
        ps, pre, qd, val = 1e6, 1e4, 100, 1e4
    q = lambda c: (v(c) // qd).cast(pl.Int64)  # noqa: E731 -- a quantity column in shares
    lvl = lambda s, f, cast: pl.concat_list([cast(f"{s}{i}_{f}") for i in range(1, LEVELS + 1)]).list.to_array(LEVELS)  # noqa: E731
    st = pl.col("status" if market == SSE else "phase")
    has = lambda c: c in lf.collect_schema().names()  # noqa: E731
    zero_f, zero_i = pl.lit(0.0), pl.lit(0, pl.Int64)
    close = pl.when(v("closePx") > 0).then(v("closePx", ps)).otherwise(v("lastPx", ps)) if has("closePx") else v("lastPx", ps)
    lim = lambda c: pl.when(pl.col(c) == SZ_NO_LIMIT).then(0.0).otherwise(v(c, ps)) if has(c) else zero_f  # noqa: E731
    n_lv = lambda s, c: v(c).cast(pl.Int32) if has(c) else pl.sum_horizontal([(v(f"{s}{i}_price") > 0).cast(pl.Int32) for i in range(1, LEVELS + 1)])  # noqa: E731
    return lf.select(
        *head(market), *clocks("timeNs"),
        pl.when(st.str.len_chars() > 0).then(st.str.slice(0, 1).cast(pl.Binary).bin.encode("hex").str.to_integer(base=16, strict=False))
        .otherwise(0).cast(pl.Int16).alias("status"),
        v("preClosePx", pre).alias("pre_close"), v("openPx", ps).alias("open"), v("highPx", ps).alias("high"),
        v("lowPx", ps).alias("low"), close.alias("close"), v("lastPx", ps).alias("price"),
        lvl("bid", "price", lambda c: v(c, ps)).alias("bid_px"), lvl("bid", "qty", q).alias("bid_vol"),
        lvl("bid", "orders", lambda c: v(c).cast(pl.Int32)).alias("bid_no"),
        lvl("ask", "price", lambda c: v(c, ps)).alias("ask_px"), lvl("ask", "qty", q).alias("ask_vol"),
        lvl("ask", "orders", lambda c: v(c).cast(pl.Int32)).alias("ask_no"),
        v("numTrades").alias("total_no"), (v("totalBidNumber") if has("totalBidNumber") else zero_i).alias("total_buy_no"),
        (v("totalOfferNumber") if has("totalOfferNumber") else zero_i).alias("total_sell_no"), q("totalVolume").alias("total_volume"),
        v("totalValue", val).alias("total_amount"), q("totalBidQty").alias("total_bid_volume"), q("totalOfferQty").alias("total_ask_volume"),
        v("wavgBidPx", ps).alias("weighted_avg_bid_price"), v("wavgOfferPx", ps).alias("weighted_avg_ask_price"),
        lim("highLimitPx").alias("high_limited"), lim("lowLimitPx").alias("low_limited"),
        *[pl.lit(0, pl.Int64).alias(c) for c in ("buy_cancel_no", "buy_cancel_volume")], pl.lit(0.0).alias("buy_cancel_amount"),
        *[pl.lit(0, pl.Int64).alias(c) for c in ("sell_cancel_no", "sell_cancel_volume")], pl.lit(0.0).alias("sell_cancel_amount"),
        n_lv("bid", "numBidLevels").alias("num_buy_levels"), n_lv("ask", "numOfferLevels").alias("num_sell_levels"),
        pl.lit(0, pl.Int32).alias("buy_level_queue_no01"), pl.lit(0, pl.Int32).alias("sell_level_queue_no01"),
        v("iopv", 1e5 if market == SSE else ps).alias("iopv"), pl.lit(0.0).alias("match_last_px"),
        pl.lit(0, pl.Int64).alias("auction_volume_trade"), pl.lit(0.0).alias("auction_value_trade"))


def one(task: dict) -> dict:
    """Worker: one (kind, exchange) part of the day."""
    kind, market, src, dst = task["kind"], task["market"], Path(task["src"]), Path(task["dst"])
    t0 = _time.perf_counter()
    tag = "SH" if market == SSE else "SZ"
    row = {"file": f"{tag}:{src.parent.parent.parent.name}", "minute": tag, "bytes": src.stat().st_size, "rows": 0,
           "n_truncated_books": 0, "exchange_ts_min": None, "exchange_ts_max": None, "arrival_ts_min": None,
           "arrival_ts_max": None, "skipped": False, "source": f"tickaudit:{task['source']}", "n_symbols": 0, "seconds": 0.0}
    df = convert(kind, market, pl.scan_parquet(src)).collect()
    sort = ["symbol_id", "time"] + (["seq_id"] if "seq_id" in df.columns else [])
    df = df.sort(sort)
    schema = pa.schema(SCHEMAS[kind])
    t = df.select([f.name for f in schema]).to_arrow().cast(schema)
    tmp = dst.with_suffix(".parquet.tmp")
    pq.write_table(t, tmp, compression="zstd", compression_level=3, row_group_size=ROW_GROUP)
    tmp.replace(dst)
    if df.height:
        row.update(rows=df.height, n_symbols=df["symbol_id"].n_unique(), exchange_ts_min=int(df["time"].min()),
                   exchange_ts_max=int(df["time"].max()), arrival_ts_min=int(df["spider_ts"].min()), arrival_ts_max=int(df["spider_ts"].max()))
    row["seconds"] = round(_time.perf_counter() - t0, 1)
    print(f"[{kind}] {tag}: {row['rows']:,} rows, {row['n_symbols']} symbols, {row['seconds']:.0f}s", flush=True)
    return row


def compare(out: Path, ref: Path, date: str, kinds: tuple[str, ...]) -> None:
    """Row counts, key coverage and per-field equality against another raw tree of the same day (e.g. dysl2_raw)."""
    keys = {"order": ["market_id", "channel_id", "seq_id"], "transaction": ["market_id", "channel_id", "seq_id"],
            "quotation": ["market_id", "symbol_id", "time"], "index": ["market_id", "symbol_id", "time"]}
    for kind in kinds:
        a, b = out / f"kind={kind}" / f"date={date}", ref / f"kind={kind}" / f"date={date}"
        if not list(b.glob("part-*.parquet")):
            print(f"== {kind}: no {b}")
            continue
        A = pl.scan_parquet(str(a / "part-*.parquet"))
        B = pl.scan_parquet(str(b / "part-*.parquet"))
        if kind == "quotation":  # vendor feeds repeat or skip snapshots: compare the last row per (symbol, time)
            A, B = (x.unique(keys[kind], keep="last") for x in (A, B))
        j = A.join(B, on=keys[kind], how="full", suffix="_r", coalesce=True)
        cols = [c for c in pl.scan_parquet(str(a / "part-*.parquet")).collect_schema().names()
                if c not in keys[kind] and c not in ("shm_key", "nano_ts", "server_ts", "spider_ts")]
        agg = [pl.len().alias("rows"), pl.col(f"{cols[0]}").is_null().sum().alias("only_ref"),
               pl.col(f"{cols[0]}_r").is_null().sum().alias("only_ta")]
        both = pl.col(cols[0]).is_not_null() & pl.col(f"{cols[0]}_r").is_not_null()
        agg += [(both & (pl.col(c) != pl.col(f"{c}_r"))).sum().alias(c) for c in cols]  # level arrays compare whole rows
        r = j.select(agg).collect()
        d = r.row(0, named=True)
        print(f"== {kind} {date}: joined rows {d['rows']:,}, only in reference {d['only_ref']:,}, only in tickaudit {d['only_ta']:,}")
        print("   fields that differ on matched rows:", {k: x for k, x in d.items() if k not in ("rows", "only_ref", "only_ta") and x})
        if "price" in cols:
            e = j.filter(both & (pl.col("price") != pl.col("price_r"))).select(
                (pl.col("price") - pl.col("price_r")).abs().max().alias("max_abs"),
                ((pl.col("price") - pl.col("price_r")).abs() < 1e-9).mean().alias("frac_last_bit")).collect()
            print("   price differences:", e.row(0, named=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="tickaudit source name: datayes, xtp, remd, ksq, feitu, dy224_shm")
    ap.add_argument("--date", required=True, help="YYYYMMDD[,YYYYMMDD...]")
    ap.add_argument("--lake", type=Path, default=LAKE)
    ap.add_argument("--out", type=Path, help="default $DATA/ta_<source>_raw")
    ap.add_argument("--kinds", default="order,transaction,quotation,index")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", help="a raw tree under $DATA (e.g. dysl2_raw) to check field by field afterwards")
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    out = a.out or DATA / f"ta_{a.source}_raw"
    kinds = tuple(a.kinds.split(","))
    for date in a.date.split(","):
        if not a.compare_only:
            tasks = []
            for kind in kinds:
                for market in (SSE, SZSE):
                    src = a.lake / "tick" / TABLES[(kind, market)] / f"day={date}" / f"source={a.source}" / "data.parquet"
                    if not src.exists():
                        print(f"[{kind}] no {src}")
                        continue
                    d = out / f"kind={kind}" / f"date={date}"
                    d.mkdir(parents=True, exist_ok=True)
                    tasks.append({"kind": kind, "market": market, "source": a.source, "src": str(src),
                                  "dst": str(d / f"part-{'SH' if market == SSE else 'SZ'}-000.parquet")})
            if not tasks:
                sys.exit(f"nothing for source={a.source} day={date} under {a.lake}/tick: decode it in tickaudit first")
            with ProcessPoolExecutor(min(a.workers, len(tasks))) as ex:
                rows = list(ex.map(one, tasks))
            for kind in kinds:
                rs = [r for t, r in zip(tasks, rows) if t["kind"] == kind]
                if rs:
                    pq.write_table(pa.Table.from_pylist(rs), out / f"kind={kind}" / f"date={date}" / "_manifest.parquet")
            quality(out, date, kinds)
        if a.compare:
            ref = Path(a.compare)
            compare(out, ref if ref.is_absolute() else DATA / ref, date, kinds)


if __name__ == "__main__":
    main()
