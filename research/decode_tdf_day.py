"""Convert one day of Wind-TDF-layout CSV dumps (the XTP feed on 2.23) into the feitu raw Parquet layout.

Input layout (one zip or csv per symbol, sub-folder and day; ``0`` = snapshots, ``2`` = orders, ``3`` = trades,
``1`` = index, which is skipped)::

    <root>/600000.SH/0/20261026.zip     LocalTime,WindCode,Code,ActionDay,Time,Status,PreClose,...,AskPrice1..,BidVol1..
    <root>/600000.SH/2/20261026.zip     LocalTime,WindCode,Code,ActionDay,Time,Order,OrderKind,FunctionCode,Price,Volume,...
    <root>/600000.SH/3/20261026.zip     LocalTime,WindCode,Code,ActionDay,Time,Index,FunctionCode,OrderKind,BSFlag,Price,...

Output: exactly the columns, dtypes and column order of ``decode_feitu_day.py`` (``crucible_kernels.feitu``), so
``catalog.py``'s ``raw_order / raw_transaction / raw_quotation`` views and everything built on them read these days
unchanged::

    <out>/kind=<order|transaction|quotation>/date=YYYYMMDD/part-<SH|SZ>-NNN.parquet   NNN = symbol chunk
    <out>/kind=<...>/date=YYYYMMDD/_manifest.parquet   one row per part, ``source = "xtp_tdf"``
    <out>/kind=<...>/date=YYYYMMDD/_quality.parquet    per (exchange, channel) sequence coverage, see ``quality``

Field mapping (tickaudit ``TdfSource.h`` is the TDF reference, ``research/bt/events.py`` the feitu one):

* clocks: ``time`` = ActionDay + Time (HHMMSSmmm, CST) as UTC epoch ns; ``spider_ts`` = ``nano_ts`` = ``server_ts`` =
  LocalTime (the receiver's clock) as UTC epoch ns.
* ids: ``symbol_id`` = int(code), ``market_id`` 3553 SH / 3554 SZ, ``symbol_type`` 1, ``shm_key`` 0.
* orders SSE: OrderKind A / D / S -> ``update_type`` 1 / 2 / 3, FunctionCode B / S -> ``dir`` 1063 / 1550,
  ``order_id`` = OrderOriNo (else Order), ``seq_id`` = ``index_id`` = BizIndex (else Order), ``order_type`` 2.
* orders SZSE: Order -> ``order_id`` = ``seq_id`` = ``index_id`` (ApplSeqNum), OrderKind 1 / 0|2 / U ->
  ``order_type`` 1 market / 2 limit / 3 own-side best, ``update_type`` 1.
* trades SSE: Index (BizIndex when set) -> ``seq_id`` = ``index_id``, BSFlag B / S / other -> ``dir``,
  BidOrder / AskOrder -> ``buy_seq_id`` / ``sell_seq_id``, ``trade_type`` 1.
* trades SZSE: Index -> ``seq_id``, FunctionCode 0|F -> ``trade_type`` 1, C|4 -> 2 (cancel), ``dir`` = the leg with the
  larger ApplSeqNum (the aggressor rule events.py uses), a cancel's side = whichever order id is set.
* snapshots: named columns mapped by the table in ``QUOTE_MAP``; fields TDF does not carry stay 0; ``status`` is the
  TDF status character's ASCII code.

Usage (97)::

    python research/decode_tdf_day.py --root /data/xtp_tdf --date 20261026 --probe       # headers + mapping only
    python research/decode_tdf_day.py --root /data/xtp_tdf --date 20261026              # convert + quality
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import time as _time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LEVELS = 10
SSE, SZSE = 3553, 3554
BUY, SELL = 1063, 1550
STOCKS = re.compile(r"^(60|68|00|30)\d{4}\.(SH|SZ)$")  # A-share stocks: SH main / STAR, SZ main / ChiNext
SUBS = {"0": "quotation", "2": "order", "3": "transaction"}
CST_NS = 8 * 3600 * 10**9

#: feitu column order and dtype per kind (``kernels/src/bindings.cpp``); level columns are FixedSizeList[LEVELS].
ORDER_SCHEMA = [("shm_key", pa.int32()), ("nano_ts", pa.int64()), ("symbol_id", pa.int32()), ("market_id", pa.int16()),
                ("time", pa.int64()), ("symbol_type", pa.int32()), ("price", pa.float64()), ("volume", pa.int64()),
                ("dir", pa.int16()), ("order_id", pa.int64()), ("channel_id", pa.int32()), ("seq_id", pa.int64()),
                ("index_id", pa.int64()), ("order_type", pa.int8()), ("update_type", pa.int8()),
                ("spider_ts", pa.int64()), ("server_ts", pa.int64())]
TRADE_SCHEMA = [("shm_key", pa.int32()), ("nano_ts", pa.int64()), ("symbol_id", pa.int32()), ("market_id", pa.int16()),
                ("time", pa.int64()), ("symbol_type", pa.int32()), ("price", pa.float64()), ("volume", pa.int64()),
                ("dir", pa.int16()), ("channel_id", pa.int32()), ("index_id", pa.int64()), ("seq_id", pa.int64()),
                ("buy_seq_id", pa.int64()), ("sell_seq_id", pa.int64()), ("trade_type", pa.int8()),
                ("spider_ts", pa.int64()), ("server_ts", pa.int64())]
QUOTE_SCHEMA = [("shm_key", pa.int32()), ("nano_ts", pa.int64()), ("symbol_id", pa.int32()), ("market_id", pa.int16()),
                ("time", pa.int64()), ("symbol_type", pa.int32()), ("status", pa.int16()), ("pre_close", pa.float64()),
                ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()),
                ("price", pa.float64()),
                ("bid_px", pa.list_(pa.float64(), LEVELS)), ("bid_vol", pa.list_(pa.int64(), LEVELS)),
                ("bid_no", pa.list_(pa.int32(), LEVELS)), ("ask_px", pa.list_(pa.float64(), LEVELS)),
                ("ask_vol", pa.list_(pa.int64(), LEVELS)), ("ask_no", pa.list_(pa.int32(), LEVELS)),
                ("total_no", pa.int64()), ("total_buy_no", pa.int64()), ("total_sell_no", pa.int64()),
                ("total_volume", pa.int64()), ("total_amount", pa.float64()), ("total_bid_volume", pa.int64()),
                ("total_ask_volume", pa.int64()), ("weighted_avg_bid_price", pa.float64()),
                ("weighted_avg_ask_price", pa.float64()), ("high_limited", pa.float64()), ("low_limited", pa.float64()),
                ("buy_cancel_no", pa.int64()), ("buy_cancel_volume", pa.int64()), ("buy_cancel_amount", pa.float64()),
                ("sell_cancel_no", pa.int64()), ("sell_cancel_volume", pa.int64()), ("sell_cancel_amount", pa.float64()),
                ("num_buy_levels", pa.int32()), ("num_sell_levels", pa.int32()), ("buy_level_queue_no01", pa.int32()),
                ("sell_level_queue_no01", pa.int32()), ("iopv", pa.float64()), ("match_last_px", pa.float64()),
                ("auction_volume_trade", pa.int64()), ("auction_value_trade", pa.float64()),
                ("spider_ts", pa.int64()), ("server_ts", pa.int64())]
SCHEMAS = {"order": ORDER_SCHEMA, "transaction": TRADE_SCHEMA, "quotation": QUOTE_SCHEMA}

#: feitu snapshot field -> candidate TDF header names (lower-case, first present wins).
QUOTE_MAP = {
    "pre_close": ("preclose", "pre_close", "precloseprice"), "open": ("open", "openprice"), "high": ("high", "highprice"),
    "low": ("low", "lowprice"), "price": ("match", "last", "lastprice", "lastpx", "price"),
    "total_no": ("numtrades", "num_trades", "tradesnum"), "total_volume": ("ivolume", "volume", "totalvolume"),
    "total_amount": ("iturnover", "turnover", "totalturnover", "amount"),
    "total_bid_volume": ("totalbidvol", "totalbidvolume"), "total_ask_volume": ("totalaskvol", "totalaskvolume"),
    "weighted_avg_bid_price": ("weightedavgbidprice", "wavgbidprice"),
    "weighted_avg_ask_price": ("weightedavgaskprice", "wavgaskprice"),
    "high_limited": ("highlimited", "highlimit", "upperlimit"), "low_limited": ("lowlimited", "lowlimit", "lowerlimit"),
    "iopv": ("iopv",), "total_buy_no": ("totalbidno", "bidnum"), "total_sell_no": ("totalaskno", "asknum"),
}
QUOTE_LEVEL_MAP = {"bid_px": "bidprice{}", "bid_vol": "bidvol{}", "ask_px": "askprice{}", "ask_vol": "askvol{}"}


def _read_csv(path: Path) -> pl.DataFrame:
    """One TDF file (zip holding one csv, or a bare csv) as strings, header names lower-cased."""
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            data = z.read(names[0]) if names else b""
    else:
        data = path.read_bytes()
    if not data.strip():
        return pl.DataFrame()
    df = pl.read_csv(io.BytesIO(data), infer_schema_length=0, truncate_ragged_lines=True)
    return df.rename({c: c.strip().lower() for c in df.columns})


def _code_char(col: str, df: pl.DataFrame) -> pl.Expr:
    """Read a TDF code column as its character: TDF writes either the character ('A') or its ASCII code ('65')."""
    if col not in df.columns:
        return pl.lit("")
    return pl.col(col).str.strip_chars().replace(_ASCII).fill_null("")


_ASCII = {str(i): chr(i) for i in range(33, 127)}


def _int(col: str, df: pl.DataFrame, default: int = 0) -> pl.Expr:
    if col not in df.columns:
        return pl.lit(default, pl.Int64)
    return pl.col(col).str.strip_chars().cast(pl.Float64, strict=False).fill_null(default).cast(pl.Int64)


def _flt(col: str, df: pl.DataFrame) -> pl.Expr:
    if col not in df.columns:
        return pl.lit(0.0, pl.Float64)
    return pl.col(col).str.strip_chars().cast(pl.Float64, strict=False).fill_null(0.0)


def _clocks(df: pl.DataFrame, day: int) -> list[pl.Expr]:
    """``time`` from ActionDay + Time (HHMMSSmmm, CST) and ``spider_ts`` from LocalTime, both UTC epoch ns."""
    action = _int("actionday", df, day).cast(pl.String)
    t = _int("time", df)
    hms = (t // 1000).cast(pl.String).str.zfill(6)
    ms = (t % 1000).cast(pl.String).str.zfill(3)
    exch = (action + hms + ms).str.to_datetime("%Y%m%d%H%M%S%3f", time_unit="ns", strict=False)
    exch_ns = exch.cast(pl.Int64) - CST_NS
    if "localtime" in df.columns:
        loc = pl.col("localtime").str.strip_chars().str.to_datetime("%Y%m%d-%H:%M:%S%.f", time_unit="ns", strict=False)
        loc_ns = loc.cast(pl.Int64) - CST_NS
    else:
        loc_ns = exch_ns
    return [exch_ns.fill_null(0).alias("time"), loc_ns.fill_null(exch_ns).fill_null(0).alias("spider_ts")]


def _side(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr == "B").then(BUY).when(expr == "S").then(SELL).otherwise(0).cast(pl.Int16)


def convert_orders(df: pl.DataFrame, code: int, market: int, day: int) -> pl.DataFrame:
    """Sub-folder 2 -> feitu ``order`` columns."""
    kind = _code_char("orderkind", df)
    func = _code_char("functioncode", df)
    order = _int("order", df)
    biz = _int("bizindex", df)
    seq = pl.when(biz != 0).then(biz).otherwise(order)
    if market == SSE:
        ori = _int("orderorino", df)
        order_id = pl.when(ori != 0).then(ori).otherwise(order)
        update = pl.when(kind == "A").then(1).when(kind == "D").then(2).otherwise(3).cast(pl.Int8)
        otype = pl.lit(2, pl.Int8)
    else:
        order_id = order
        update = pl.lit(1, pl.Int8)
        otype = pl.when(kind == "1").then(1).when(kind == "U").then(3).otherwise(2).cast(pl.Int8)
    return df.select(
        pl.lit(0, pl.Int32).alias("shm_key"), pl.lit(0, pl.Int64).alias("nano_ts"),
        pl.lit(code, pl.Int32).alias("symbol_id"), pl.lit(market, pl.Int16).alias("market_id"),
        *_clocks(df, day), pl.lit(1, pl.Int32).alias("symbol_type"),
        _flt("price", df).alias("price"), _int("volume", df).alias("volume"), _side(func).alias("dir"),
        order_id.alias("order_id"), _int("channel", df).cast(pl.Int32).alias("channel_id"),
        seq.alias("seq_id"), seq.alias("index_id"), otype.alias("order_type"), update.alias("update_type"),
    ).with_columns(pl.col("spider_ts").alias("nano_ts"), pl.col("spider_ts").alias("server_ts"))


def convert_trades(df: pl.DataFrame, code: int, market: int, day: int) -> pl.DataFrame:
    """Sub-folder 3 -> feitu ``transaction`` columns."""
    func = _code_char("functioncode", df)
    bs = _code_char("bsflag", df)
    idx = _int("index", df)
    biz = _int("bizindex", df)
    seq = pl.when(biz != 0).then(biz).otherwise(idx)
    bid, ask = _int("bidorder", df), _int("askorder", df)
    if market == SSE:
        ttype = pl.lit(1, pl.Int8)
        side = _side(bs)
    else:
        cancel = func.is_in(["C", "4"])
        ttype = pl.when(cancel).then(2).otherwise(1).cast(pl.Int8)
        side = (pl.when(cancel).then(pl.when(bid != 0).then(BUY).otherwise(SELL))
                .otherwise(pl.when(bid > ask).then(BUY).otherwise(SELL)).cast(pl.Int16))
    return df.select(
        pl.lit(0, pl.Int32).alias("shm_key"), pl.lit(0, pl.Int64).alias("nano_ts"),
        pl.lit(code, pl.Int32).alias("symbol_id"), pl.lit(market, pl.Int16).alias("market_id"),
        *_clocks(df, day), pl.lit(1, pl.Int32).alias("symbol_type"),
        _flt("price", df).alias("price"), _int("volume", df).alias("volume"), side.alias("dir"),
        _int("channel", df).cast(pl.Int32).alias("channel_id"), seq.alias("index_id"), seq.alias("seq_id"),
        bid.alias("buy_seq_id"), ask.alias("sell_seq_id"), ttype.alias("trade_type"),
    ).with_columns(pl.col("spider_ts").alias("nano_ts"), pl.col("spider_ts").alias("server_ts"))


def quote_mapping(columns: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map feitu field -> TDF header actually present, and list the TDF headers nothing consumes."""
    used = {"localtime", "windcode", "code", "actionday", "time", "status", "vendor"}
    m: dict[str, str] = {}
    for k, cands in QUOTE_MAP.items():
        for c in cands:
            if c in columns:
                m[k] = c
                used.add(c)
                break
    for k, pat in QUOTE_LEVEL_MAP.items():
        for i in range(1, LEVELS + 1):
            c = pat.format(i)
            if c in columns:
                m[f"{k}{i}"] = c
                used.add(c)
    return m, [c for c in columns if c not in used]


def convert_quotes(df: pl.DataFrame, code: int, market: int, day: int) -> pl.DataFrame:
    """Sub-folder 0 -> feitu ``quotation`` columns (levels as fixed-size lists)."""
    m, _ = quote_mapping(df.columns)
    status = _code_char("status", df)
    cols = [
        pl.lit(0, pl.Int32).alias("shm_key"), pl.lit(0, pl.Int64).alias("nano_ts"),
        pl.lit(code, pl.Int32).alias("symbol_id"), pl.lit(market, pl.Int16).alias("market_id"),
        *_clocks(df, day), pl.lit(1, pl.Int32).alias("symbol_type"),
        pl.when(status.str.len_chars() > 0).then(status.str.slice(0, 1).cast(pl.Binary).bin.encode("hex")
                                                 .str.to_integer(base=16, strict=False)).otherwise(0).cast(pl.Int16).alias("status"),
    ]
    for name, typ in QUOTE_SCHEMA[7:]:
        if name in ("spider_ts", "nano_ts", "time", "status"):
            continue
        if isinstance(typ, pa.FixedSizeListType):
            inner = pl.Float64 if pa.types.is_floating(typ.value_type) else pl.Int64
            parts = [(_flt(m[f"{name}{i}"], df) if inner is pl.Float64 else _int(m[f"{name}{i}"], df)).cast(inner)
                     if f"{name}{i}" in m else pl.lit(0, inner) for i in range(1, LEVELS + 1)]
            cols.append(pl.concat_list(parts).list.to_array(LEVELS).alias(name))
        elif name in m:
            cols.append((_flt(m[name], df) if pa.types.is_floating(typ) else _int(m[name], df)).alias(name))
        elif name == "close":
            cols.append(_flt(m["price"], df).alias(name) if "price" in m else pl.lit(0.0).alias(name))
        elif name == "num_buy_levels":
            cols.append(pl.sum_horizontal([(_flt(m[f"bid_px{i}"], df) > 0).cast(pl.Int32) for i in range(1, LEVELS + 1)
                                           if f"bid_px{i}" in m] or [pl.lit(0, pl.Int32)]).alias(name))
        elif name == "num_sell_levels":
            cols.append(pl.sum_horizontal([(_flt(m[f"ask_px{i}"], df) > 0).cast(pl.Int32) for i in range(1, LEVELS + 1)
                                           if f"ask_px{i}" in m] or [pl.lit(0, pl.Int32)]).alias(name))
        else:
            cols.append(pl.lit(0.0 if pa.types.is_floating(typ) else 0).alias(name))
    return df.select(cols).with_columns(pl.col("spider_ts").alias("nano_ts"), pl.col("spider_ts").alias("server_ts"))


CONVERT = {"order": convert_orders, "transaction": convert_trades, "quotation": convert_quotes}


def _to_table(df: pl.DataFrame, kind: str) -> pa.Table:
    """Cast to the feitu schema, in its column order."""
    schema = pa.schema(SCHEMAS[kind])
    t = df.select([n for n, _ in SCHEMAS[kind]]).to_arrow()
    return t.cast(schema)


def _one(kind: str, path: Path, code: int, market: int, day: int) -> pl.DataFrame | None:
    df = _read_csv(path)
    if df.is_empty():
        return None
    out = CONVERT[kind](df, code, market, day)
    return out.filter(pl.col("time") > 0)


def symbols(root: Path, pattern: re.Pattern[str]) -> list[tuple[str, int, int]]:
    """(dir name, code, market) of every symbol folder matching ``pattern``."""
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and pattern.match(p.name):
            code, ex = p.name.split(".")
            out.append((p.name, int(code), SSE if ex == "SH" else SZSE))
    return out


def _file(root: Path, sym: str, sub: str, date: str) -> Path | None:
    for ext in (".zip", ".csv"):
        f = root / sym / sub / f"{date}{ext}"
        if f.is_file():
            return f
    return None


def probe(root: Path, date: str, syms: list[tuple[str, int, int]]) -> None:
    """Print the header, two rows and the snapshot mapping of one SH and one SZ file per kind, then stop."""
    for market in (SSE, SZSE):
        pick = next((s for s in syms if s[2] == market), None)
        if pick is None:
            print(f"no symbol of market {market}")
            continue
        for sub, kind in SUBS.items():
            f = _file(root, pick[0], sub, date)
            print(f"\n== {kind} {pick[0]} {f}")
            if f is None:
                continue
            df = _read_csv(f)
            print(f"rows {df.height}; columns: {df.columns}")
            print(df.head(2))
            if kind == "quotation":
                m, unused = quote_mapping(df.columns)
                print("mapped:", m)
                print("UNMAPPED TDF columns:", unused)
                missing = [k for k in QUOTE_MAP if k not in m]
                print("feitu fields left at 0:", missing)
            out = CONVERT[kind](df, pick[1], market, int(date))
            with pl.Config(tbl_cols=20, tbl_width_chars=200):
                print(out.head(3))


def convert_day(root: Path, date: str, out: Path, syms: list[tuple[str, int, int]], *, kinds: tuple[str, ...],
                threads: int, chunk: int, overwrite: bool) -> None:
    """Write every kind as per-exchange symbol-chunk parts plus the manifest; then the quality table."""
    day = int(date)
    for kind in kinds:
        sub = next(s for s, k in SUBS.items() if k == kind)
        dst_dir = out / f"kind={kind}" / f"date={date}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        foreign = [f.name for f in dst_dir.glob("part-*.parquet") if not re.match(r"part-(SH|SZ)-\d{3}\.parquet$", f.name)]
        if foreign:  # a feitu day already lives here; two sources in one partition would double-count in the views
            sys.exit(f"{dst_dir} already holds feitu parts {foreign[:3]}: convert this day with another --out (e.g. xtp_raw)")
        rows: list[dict[str, Any]] = []
        t0 = _time.perf_counter()
        total = 0
        for market, tag in ((SSE, "SH"), (SZSE, "SZ")):
            todo = [(s, c, _file(root, s, sub, date)) for s, c, m in syms if m == market]
            todo = [(s, c, f) for s, c, f in todo if f is not None]
            if not todo:
                print(f"[{kind}] no {tag} files for {date} under {root}")
                continue
            for ci in range(0, len(todo), chunk):
                part = todo[ci:ci + chunk]
                dst = dst_dir / f"part-{tag}-{ci // chunk:03d}.parquet"
                row: dict[str, Any] = {"file": f"{tag}:{part[0][0]}..{part[-1][0]}", "minute": tag, "bytes": sum(f.stat().st_size for _, _, f in part),
                                       "rows": 0, "n_truncated_books": 0, "exchange_ts_min": None, "exchange_ts_max": None,
                                       "arrival_ts_min": None, "arrival_ts_max": None, "skipped": False,
                                       "source": "xtp_tdf", "n_symbols": len(part)}
                if dst.exists() and not overwrite:
                    row["rows"], row["skipped"] = pq.read_metadata(dst).num_rows, True
                    rows.append(row)
                    total += row["rows"]
                    continue
                with ThreadPoolExecutor(threads) as ex:
                    frames = [d for d in ex.map(lambda x, k=kind, m=market: _one(k, x[2], x[1], m, day), part) if d is not None]
                if frames:
                    df = pl.concat(frames).sort("spider_ts", "symbol_id")
                    row["rows"] = df.height
                    row["exchange_ts_min"], row["exchange_ts_max"] = int(df["time"].min()), int(df["time"].max())
                    row["arrival_ts_min"], row["arrival_ts_max"] = int(df["spider_ts"].min()), int(df["spider_ts"].max())
                    tmp = dst.with_suffix(".parquet.tmp")
                    pq.write_table(_to_table(df, kind), tmp, compression="zstd", compression_level=3, row_group_size=1 << 20)
                    tmp.replace(dst)
                rows.append(row)
                total += row["rows"]
                print(f"[{kind}] {dst.name}: {len(part)} symbols, {row['rows']:,} rows ({_time.perf_counter() - t0:.0f}s)")
        pq.write_table(pa.Table.from_pylist(rows), dst_dir / "_manifest.parquet")
        print(f"[{kind}] {date}: {total:,} rows in {_time.perf_counter() - t0:.0f}s -> {dst_dir}")
    quality(out, date, kinds)


def quality(out: Path, date: str, kinds: tuple[str, ...]) -> None:
    """Print and write the day's coverage: sequence gaps per (exchange, channel) and snapshots per symbol.

    SZSE ApplSeqNum and SSE BizIndex run through orders AND trades, so a gap in their union is a lost record.
    """
    lf = []
    for kind in kinds:
        if kind == "quotation":
            continue
        d = out / f"kind={kind}" / f"date={date}"
        if list(d.glob("part-*.parquet")):
            lf.append(pl.scan_parquet(str(d / "part-*.parquet")).select("market_id", "channel_id", "seq_id"))
    rows = []
    if lf:
        seq = pl.concat(lf).filter(pl.col("seq_id") > 0)
        q = (seq.group_by("market_id", "channel_id")
             .agg(pl.col("seq_id").min().alias("seq_min"), pl.col("seq_id").max().alias("seq_max"),
                  pl.col("seq_id").n_unique().alias("seq_distinct"), pl.len().alias("records"))
             .with_columns((pl.col("seq_max") - pl.col("seq_min") + 1).alias("seq_span"))
             .with_columns((1 - pl.col("seq_distinct") / pl.col("seq_span")).alias("missing_frac"))
             .sort("market_id", "channel_id").collect())
        q = q.with_columns(pl.lit("seq").alias("check"))
        rows.append(q)
        with pl.Config(tbl_rows=40, tbl_width_chars=160):
            print(f"\n== {date} sequence coverage per (market, channel): missing_frac > 0 = lost records (SSE channel-wide "
                  "BizIndex covers only the symbols converted, so a small positive value there is the universe filter, not loss)")
            print(q.select("market_id", "channel_id", "records", "seq_span", "seq_distinct", "missing_frac"))
            print(q.group_by("market_id").agg(pl.col("records").sum(), pl.col("seq_span").sum(), pl.col("seq_distinct").sum())
                  .with_columns((1 - pl.col("seq_distinct") / pl.col("seq_span")).alias("missing_frac")).sort("market_id"))
    if "quotation" in kinds:
        d = out / "kind=quotation" / f"date={date}"
        if list(d.glob("part-*.parquet")):
            s = (pl.scan_parquet(str(d / "part-*.parquet")).group_by("market_id", "symbol_id").agg(pl.len().alias("snapshots"))
                 .group_by("market_id").agg(pl.col("snapshots").median().alias("snap_median"), pl.col("snapshots").min().alias("snap_min"),
                                            pl.col("snapshots").quantile(0.05).alias("snap_p05"), pl.len().alias("symbols"))
                 .sort("market_id").collect())
            print("\n== snapshots per symbol (feitu reference: SZSE ~4700-4800 at 3 s, SSE similar):")
            print(s)
    if rows:
        pq.write_table(rows[0].to_arrow(), out / "kind=order" / f"date={date}" / "_quality.parquet")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="directory holding the <symbol>.<SH|SZ>/<sub>/<date>.zip tree")
    ap.add_argument("--date", required=True, help="trading day YYYYMMDD")
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--kinds", default="quotation,order,transaction")
    ap.add_argument("--symbols", default=STOCKS.pattern, help="regex on the symbol folder name")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=250, help="symbols per Parquet part")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--probe", action="store_true", help="print headers, mapping and three converted rows; write nothing")
    ap.add_argument("--quality-only", action="store_true", help="recompute the coverage table from existing parts")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    if args.quality_only:
        quality(args.out, args.date, kinds)
        return
    syms = symbols(args.root, re.compile(args.symbols))
    print(f"{len(syms)} symbols under {args.root} match {args.symbols!r}")
    if args.probe:
        probe(args.root, args.date, syms)
        return
    convert_day(args.root, args.date, args.out, syms, kinds=kinds, threads=args.threads, chunk=args.chunk, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
