"""Decode the PM's ``.mp`` market-data files into the feitu raw Parquet layout (intranet).

``/data/prod/mp/<date>/{SH,SZ}_{HS300,ZZ500,ZZ1000}_{order,trade,snap,index}.mp`` are zstd-
compressed msgpack streams, one array per record, timestamps as float epoch seconds (UTC).
Each file holds the constituents of that index on that exchange. They are written exactly
like ``research/decode_feitu_day.py`` output, so every downstream tool (quality gate,
1-minute bars, samplerR, factor candidates, catalog views) runs on them unchanged::

    <out>/kind=<order|transaction|quotation|index>/date=<D>/part-<EX>_<IDX>[-<k>].parquet

Record layouts (read from the files, 2026-09-30):

======================  =======================================================================
file                    fields
======================  =======================================================================
SZ order (10)           code, local ts, exchange ts, appl seq, price, qty, ord type (ASCII
                        '1' market / '2' limit / 'U' own best), side ('B' / 'S'), channel,
                        seq (= appl seq)
SH order (10)           code, local ts, exchange ts, order no, price, qty, type ('A' add /
                        'D' delete), side ('B' / 'S'), channel, biz index
SZ / SH trade (9)       code, local ts, exchange ts, buy order, sell order, price, qty,
                        channel, seq / biz index. SZ price 0 = a cancel. **No BS flag.**
snap (51)               code, local ts, exchange ts, cum volume, cum amount, ask px 1-10,
                        bid px 1-10, ask vol 1-10, bid vol 1-10, pre close, last, n trades,
                        open, high, low
index (7)               10000000 + code, local ts, exchange ts, pre close, last, volume, amount
======================  =======================================================================

Choices (logged in ``_manifest.parquet`` and checked by ``--check``):

* trade direction: none in the files; the aggressor is the leg with the larger order number
  (it arrived later); call-auction trades (before 09:30 or from 14:57 exchange time) get 0;
* SZ trades with price 0 are cancels (``trade_type`` 2, direction 0);
* snapshot volume order is assumed to follow the prices (asks 25-34, bids 35-44) --
  ``--check`` reports which volume block is empty where ask 1 is empty (limit-up books);
* fields the files do not carry (per-level order counts, whole-book totals, price limits)
  are null, never 0.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SSE, SZSE = 3553, 3554
BUY, SELL = 1063, 1550
KIND_OUT = {"order": "order", "trade": "transaction", "snap": "quotation", "index": "index"}
NAME_RE = re.compile(r"^(SH|SZ)_([A-Z0-9]+)_(order|trade|snap|index)\.mp$")
CHUNK = 2_000_000
LEVELS = 10
_AUCTION_OPEN_END = 9 * 3600 + 30 * 60  # 09:30:00 local, seconds of day
_AUCTION_CLOSE_START = 14 * 3600 + 57 * 60  # 14:57:00


def _ns(x: object) -> np.ndarray:
    return np.rint(np.asarray(x, dtype=np.float64) * 1e9).astype(np.int64)


def _auction(exch_ns: np.ndarray) -> np.ndarray:
    sod = ((exch_ns // 1_000_000_000) + 8 * 3600) % 86400
    return (sod < _AUCTION_OPEN_END) | (sod >= _AUCTION_CLOSE_START)


def _aggressor(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    return np.where(buy > sell, BUY, SELL).astype(np.int16)


def _common(c: list, market: int, n: int) -> dict:
    local, exch = _ns(c[1]), _ns(c[2])
    return {"shm_key": np.zeros(n, np.int32), "nano_ts": local, "symbol_id": np.asarray(c[0], np.int64).astype(np.int32),
            "market_id": np.full(n, market, np.int16), "time": exch, "symbol_type": np.ones(n, np.int32),
            "spider_ts": local, "server_ts": local}


def _order(c: list, market: int) -> dict:
    n = len(c[0])
    out = _common(c, market, n)
    side = np.asarray(c[7], np.int64)
    out["price"] = np.asarray(c[4], np.float64)
    out["volume"] = np.rint(np.asarray(c[5], np.float64)).astype(np.int64)
    out["dir"] = np.where(side == 66, BUY, np.where(side == 83, SELL, 0)).astype(np.int16)
    out["channel_id"] = np.asarray(c[8], np.int64).astype(np.int32)
    typ = np.asarray(c[6], np.int64)
    if market == SZSE:
        out["seq_id"] = np.asarray(c[3], np.int64)
        out["order_id"] = np.asarray(c[9], np.int64)
        out["order_type"] = np.select([typ == 49, typ == 50, typ == 85], [1, 2, 3], 0).astype(np.int8)
        out["update_type"] = np.ones(n, np.int8)
    else:
        out["order_id"] = np.asarray(c[3], np.int64)
        out["seq_id"] = np.asarray(c[9], np.int64)
        out["order_type"] = np.full(n, 2, np.int8)
        out["update_type"] = np.select([typ == 65, typ == 68], [1, 2], 0).astype(np.int8)
    out["index_id"] = out["seq_id"]
    return out


def _trade(c: list, market: int) -> dict:
    n = len(c[0])
    out = _common(c, market, n)
    buy, sell = np.asarray(c[3], np.int64), np.asarray(c[4], np.int64)
    price = np.asarray(c[5], np.float64)
    cancel = (price == 0) if market == SZSE else np.zeros(n, bool)
    out.update({"price": price, "volume": np.rint(np.asarray(c[6], np.float64)).astype(np.int64),
                "channel_id": np.asarray(c[7], np.int64).astype(np.int32), "seq_id": np.asarray(c[8], np.int64),
                "buy_seq_id": buy, "sell_seq_id": sell,
                "trade_type": np.where(cancel, 2, 1).astype(np.int8)})
    out["index_id"] = out["seq_id"]
    out["dir"] = np.where(cancel | _auction(out["time"]), 0, _aggressor(buy, sell)).astype(np.int16)
    return out


def _snap(c: list, market: int) -> dict:
    n = len(c[0])
    out = _common(c, market, n)

    def block(first: int, dtype: type) -> np.ndarray:
        return np.column_stack([np.asarray(c[first + k], np.float64) for k in range(LEVELS)]).astype(dtype)

    out.update({
        "status": np.zeros(n, np.int16),
        "pre_close": np.asarray(c[45], np.float64), "open": np.asarray(c[48], np.float64),
        "high": np.asarray(c[49], np.float64), "low": np.asarray(c[50], np.float64),
        "close": np.asarray(c[46], np.float64), "price": np.asarray(c[46], np.float64),
        "ask_px": block(5, np.float64), "bid_px": block(15, np.float64),
        "ask_vol": np.rint(block(25, np.float64)).astype(np.int64), "bid_vol": np.rint(block(35, np.float64)).astype(np.int64),
        "total_no": np.rint(np.asarray(c[47], np.float64)).astype(np.int64),
        "total_volume": np.rint(np.asarray(c[3], np.float64)).astype(np.int64),
        "total_amount": np.asarray(c[4], np.float64),
    })
    return out


def _index(c: list, market: int) -> dict:
    n = len(c[0])
    out = _common(c, market, n)
    out["symbol_id"] = (np.asarray(c[0], np.int64) % 1_000_000).astype(np.int32)
    nan = np.full(n, np.nan)
    out.update({"last_price": np.asarray(c[4], np.float64), "pre_close_price": np.asarray(c[3], np.float64),
                "open_price": nan, "high_price": nan, "low_price": nan, "close_price": nan,
                "total_volume": np.rint(np.asarray(c[5], np.float64)).astype(np.int64),
                "total_amount": np.asarray(c[6], np.float64),
                "symbol_str": np.array([f"{s:06d}" for s in out["symbol_id"]], dtype=object)})
    out.pop("symbol_type")  # the feitu index stream has shm_key but no symbol_type
    return out


BUILD = {"order": _order, "trade": _trade, "snap": _snap, "index": _index}
# columns the feitu decoder writes that .mp cannot fill: written as typed nulls
NULLS = {
    "quotation": {"bid_no": pa.list_(pa.int32(), LEVELS), "ask_no": pa.list_(pa.int32(), LEVELS),
                  "total_buy_no": pa.int64(), "total_sell_no": pa.int64(), "total_bid_volume": pa.int64(),
                  "total_ask_volume": pa.int64(), "weighted_avg_bid_price": pa.float64(),
                  "weighted_avg_ask_price": pa.float64(), "high_limited": pa.float64(), "low_limited": pa.float64(),
                  "buy_cancel_no": pa.int64(), "buy_cancel_volume": pa.int64(), "buy_cancel_amount": pa.float64(),
                  "sell_cancel_no": pa.int64(), "sell_cancel_volume": pa.int64(), "sell_cancel_amount": pa.float64(),
                  "num_buy_levels": pa.int32(), "num_sell_levels": pa.int32(), "buy_level_queue_no01": pa.int32(),
                  "sell_level_queue_no01": pa.int32(), "iopv": pa.float64(), "match_last_px": pa.float64(),
                  "auction_volume_trade": pa.int64(), "auction_value_trade": pa.float64()},
}


def _table(cols: dict, kind_out: str) -> pa.Table:
    arrays = {}
    for k, v in cols.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            arrays[k] = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(v).reshape(-1)), v.shape[1])
        else:
            arrays[k] = pa.array(v)
    n = len(next(iter(arrays.values())))
    for k, t in NULLS.get(kind_out, {}).items():
        arrays[k] = pa.nulls(n, type=t)
    return pa.table(arrays)


def decode_file(src: Path, dst_dir: Path, overwrite: bool) -> dict:
    """One .mp file -> Parquet part(s); returns a manifest row."""
    import msgpack
    import zstandard as zstd

    m = NAME_RE.match(src.name)
    if m is None:
        raise ValueError(f"unexpected file name {src.name!r}")
    ex, idx, kind = m.groups()
    market = SSE if ex == "SH" else SZSE
    stem = f"part-{ex}_{idx}"
    row = {"file": src.name, "minute": "", "bytes": src.stat().st_size, "rows": 0, "n_truncated_books": 0,
           "exchange_ts_min": None, "exchange_ts_max": None, "arrival_ts_min": None, "arrival_ts_max": None}
    if not overwrite and any(dst_dir.glob(stem + "*.parquet")):
        row["skipped"] = True
        return row
    for old in dst_dir.glob(stem + "*.parquet"):
        old.unlink()
    t0 = time.perf_counter()
    with src.open("rb") as f:
        unpacker = msgpack.Unpacker(zstd.ZstdDecompressor().stream_reader(f), raw=True,
                                    strict_map_key=False, max_buffer_size=2**31 - 1)
        for k in itertools.count():
            chunk = list(itertools.islice(unpacker, CHUNK))
            if not chunk:
                break
            cols = BUILD[kind](list(zip(*chunk, strict=True)), market)
            tbl = _table(cols, KIND_OUT[kind])
            tmp = dst_dir / f"{stem}-{k:03d}.parquet.tmp"
            pq.write_table(tbl, tmp, compression="zstd", compression_level=3, row_group_size=1 << 20)
            tmp.rename(tmp.with_suffix(""))
            row["rows"] += tbl.num_rows
            for key, col in (("exchange_ts", "time"), ("arrival_ts", "spider_ts")):
                v = cols[col]
                row[f"{key}_min"] = int(v.min()) if row[f"{key}_min"] is None else min(row[f"{key}_min"], int(v.min()))
                row[f"{key}_max"] = int(v.max()) if row[f"{key}_max"] is None else max(row[f"{key}_max"], int(v.max()))
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


def check(out: Path, date: str) -> None:
    """Assumption checks on the decoded day (printed; nothing written)."""
    import polars as pl

    q = pl.scan_parquet(out / "kind=quotation" / f"date={date}" / "part-*.parquet")
    b = q.select(pl.col("ask_px").arr.get(0).alias("ap1"), pl.col("bid_px").arr.get(0).alias("bp1"),
                 pl.col("ask_vol").arr.get(0).alias("av1"), pl.col("bid_vol").arr.get(0).alias("bv1"))
    s = b.select(
        ((pl.col("ap1") == 0) & (pl.col("bp1") > 0)).sum().alias("rows_ask_empty"),
        ((pl.col("ap1") == 0) & (pl.col("bp1") > 0) & (pl.col("av1") == 0)).sum().alias("ask_vol_zero_there"),
        ((pl.col("ap1") == 0) & (pl.col("bp1") > 0) & (pl.col("bv1") == 0)).sum().alias("bid_vol_zero_there"),
        ((pl.col("ap1") > 0) & (pl.col("bp1") > 0) & (pl.col("ap1") <= pl.col("bp1"))).sum().alias("crossed_books"),
    ).collect()
    print("snapshot volume order check (ask1 empty -> its volume must be 0):", s.to_dicts()[0])
    t = pl.scan_parquet(out / "kind=transaction" / f"date={date}" / "part-*.parquet")
    print("trades by market / trade_type / dir:")
    print(t.group_by("market_id", "trade_type", "dir").agg(pl.len()).sort("market_id", "trade_type", "dir").collect())
    o = pl.scan_parquet(out / "kind=order" / f"date={date}" / "part-*.parquet")
    print("orders by market / update_type / order_type / dir:")
    print(o.group_by("market_id", "update_type", "order_type", "dir").agg(pl.len())
          .sort("market_id", "update_type", "order_type", "dir").collect())


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("/data/prod/mp"), help="directory holding <date>/*.mp")
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="print the assumption checks after decoding")
    a = ap.parse_args()
    files = sorted(p for p in (a.root / a.date).glob("*.mp") if NAME_RE.match(p.name))
    if not files:
        sys.exit(f"no .mp files under {a.root / a.date}")
    by_kind: dict[str, list[dict]] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        for f in files:
            kind = KIND_OUT[NAME_RE.match(f.name).group(3)]
            dst = a.out / f"kind={kind}" / f"date={a.date}"
            dst.mkdir(parents=True, exist_ok=True)
            futs[ex.submit(decode_file, f, dst, a.overwrite)] = (kind, f.name)
        for fut in as_completed(futs):
            kind, name = futs[fut]
            row = fut.result()
            by_kind.setdefault(kind, []).append(row)
            print(f"  {name}: {row['rows']:,} rows{' (skipped)' if row.get('skipped') else ''} "
                  f"{row.get('seconds', 0)}s")
    for kind, rows in by_kind.items():
        man = pa.Table.from_pylist([{**r, "source": "pm_mp"} for r in rows])
        pq.write_table(man, a.out / f"kind={kind}" / f"date={a.date}" / "_manifest.parquet")
    print(f"{a.date}: {sum(r['rows'] for rs in by_kind.values() for r in rs):,} rows -> {a.out}")
    if a.check:
        check(a.out, a.date)


if __name__ == "__main__":
    main()
