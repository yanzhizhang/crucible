"""Convert DataYes (通联) XTP L2 day packages into the feitu raw Parquet layout, several packages through one pool.

Input, one zip per trading day (``/mnt/TickDataRepo/XTPSharesL2`` on 97, ~11 GB each), or the same unzipped::

    <root>/20261026.zip   holding  20261026/<sub>/600000.SH[.csv]     sub 0 snapshots / 1 index / 2 orders / 3 trades
    <root>/20261026/<sub>/600000.SH[.csv]

Headers follow the DataYes landing spec (sht-demo ``prototype/tdf/datayes_row.hpp``) and are renamed onto the TDF
ones, after which ``decode_tdf_day.py``'s converters, part writer and pool run unchanged, so the output, manifest
(``source = "datayes"``) and quality table are the same as for a TDF day:

* clocks: LocalTime ``YYYY-mm-dd HH:MM:SS.fffffffff`` -> TDF ``YYYYmmdd-HH:MM:SS.f``, Time ``hh:mm:ss.mmm`` -> HHMMSSmmm.
* orders: Type -> OrderKind (SSE A / D / S, SZSE 1 / 2 / U), TickBSFlag -> FunctionCode, BizIndex -> Order and
  BizIndex, the non-zero of Buy/SellOrderNO -> OrderOriNo, Qty -> Volume.
* trades: Type -> FunctionCode (F / 4, or their ASCII codes 70 / 52), TickBSFlag -> BSFlag, BizIndex -> Index,
  BuyOrderNO / SellOrderNO -> BidOrder / AskOrder, Qty -> Volume.
* snapshots: TradeNumber -> NumTrades, High/LowLimitPrice -> HighLimited / LowLimited, Ask/BidVolumeN -> Ask/BidVolN.

Every (day, kind, exchange, symbol chunk) part is an independent task, so all requested packages decode at once,
biggest parts first; each worker keeps the packages it touches open and reads only its own entries.

    python research/decode_datayes_day.py --date 20261102 --probe                      # headers + mapping only
    python research/decode_datayes_day.py --date 20260914-20261102 --workers 96        # every package in the range
    python research/decode_datayes_day.py --date all     # nightly: only days feitu lacks, only ones not yet converted

A day whose manifests already exist for every kind is skipped without opening its package, so ``--date all`` only
converts new packages; a package still being written (no zip directory yet) is skipped and picked up next run.
Days feitu already holds in ``--out`` are skipped too (feitu is the primary source; one source per partition).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_tdf_day import (
    CONVERT,
    LEVELS,
    QUOTE_MAP,
    SSE,
    STOCKS,
    SUBS,
    SZSE,
    check_partition,
    convert,
    parse_csv,
    part_row,
    quote_mapping,
    run_parts,
    write_part,
)

ROOT = Path("/mnt/TickDataRepo/XTPSharesL2")
ENTRY = re.compile(r"(?:^|/)(\d)/(\d{6}\.(?:SH|SZ))(?:\.csv)?$")  # <sub>/<WindCode>[.csv] inside a day

#: DataYes header (lower-cased) -> TDF header, per kind; orders also get Order / OrderOriNo in ``from_datayes``.
RENAME = {
    "order": {"type": "orderkind", "tickbsflag": "functioncode", "qty": "volume"},
    "transaction": {"type": "functioncode", "tickbsflag": "bsflag", "bizindex": "index", "buyorderno": "bidorder",
                    "sellorderno": "askorder", "qty": "volume"},
    # totalbuynum / totalsellnum = XTP total_buy_count / total_sell_count (SSE whole-book order counts, feitu totalBuyNo);
    # bidnum / asknum are XTP num_bid_orders / num_ask_orders, not that -- compare_xtp_feitu.py checks it
    "quotation": {"tradenumber": "numtrades", "highlimitprice": "highlimited", "lowlimitprice": "lowlimited",
                  "totalbuynum": "totalbidno", "totalsellnum": "totalaskno",
                  **{f"{s}volume{i}": f"{s}vol{i}" for s in ("ask", "bid") for i in range(1, LEVELS + 1)}},
}


def from_datayes(df: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Rename a DataYes frame onto TDF headers and rewrite its two clocks in the TDF formats."""
    df = df.rename(plain_names(df.columns))
    codes = [pl.col(c).str.replace(r"\(.*\)$", "").str.strip_chars() for c in ("type", "tickbsflag") if c in df.columns]
    df = df.with_columns(codes) if codes else df  # SZSE writes the side as "B(1)" / "S(2)"
    df = df.rename({k: v for k, v in RENAME[kind].items() if k in df.columns})
    extra = []
    if "time" in df.columns:
        raw = pl.col("time").str.strip_chars()
        t = raw.str.to_time("%H:%M:%S%.f", strict=False)
        hms = (t.dt.hour().cast(pl.Int64) * 10_000_000 + t.dt.minute().cast(pl.Int64) * 100_000
               + t.dt.second().cast(pl.Int64) * 1000 + t.dt.millisecond().cast(pl.Int64)).fill_null(0).cast(pl.String)
        extra.append(pl.when(raw.str.contains(":")).then(hms).otherwise(raw).alias("time"))  # else already HHMMSSmmm
    if "localtime" in df.columns:
        lt = pl.col("localtime").str.strip_chars()
        date, clock = lt.str.slice(0, 10).str.replace_all("-", ""), lt.str.slice(11)
        extra.append(pl.when(lt.str.slice(4, 1) == "-").then(date + "-" + clock).otherwise(lt).alias("localtime"))
    if kind == "order" and "bizindex" in df.columns:
        extra.append(pl.col("bizindex").alias("order"))
        if "buyorderno" in df.columns and "sellorderno" in df.columns:
            b = pl.col("buyorderno").str.strip_chars()
            extra.append(pl.when(b.cast(pl.Float64, strict=False).fill_null(0) != 0).then(b)
                         .otherwise(pl.col("sellorderno")).alias("orderorino"))
    return df.with_columns(extra) if extra else df


def package(root: Path, date: str) -> Path | None:
    """Find the day's source: ``<date>.zip``, else the unzipped ``<date>/`` folder."""
    for p in (root / f"{date}.zip", root / date):
        if p.exists():
            return p
    return None


def index(pkg: Path) -> dict[tuple[str, str], tuple[str, int]]:
    """(WindCode, sub) -> (entry name or file path, compressed bytes), from the zip directory (nothing decompressed)."""
    out: dict[tuple[str, str], tuple[str, int]] = {}
    if pkg.is_file():
        with zipfile.ZipFile(pkg) as z:
            for info in z.infolist():
                m = ENTRY.search(info.filename)
                if m:
                    out[(m[2], m[1])] = (info.filename, info.compress_size)
    else:
        for f in pkg.glob("*/*"):
            m = ENTRY.search(f"{f.parent.name}/{f.name}")
            if m and f.is_file():
                out[(m[2], m[1])] = (str(f), f.stat().st_size)
    return out


_OPEN: dict[str, zipfile.ZipFile] = {}  # packages a worker has open: each part reads hundreds of entries


def read(source: str) -> pl.DataFrame:
    """One entry, ``<package.zip>!<entry>`` or a plain file path."""
    if "!" not in source:
        return parse_csv(Path(source).read_bytes())
    pkg, entry = source.split("!", 1)
    if pkg not in _OPEN:
        _OPEN[pkg] = zipfile.ZipFile(pkg)
    return parse_csv(_OPEN[pkg].read(entry))


def plain_names(columns: list[str]) -> dict[str, str]:
    """Drop the alias DataYesSharesL2 appends to a header (``code(securityid)`` -> ``code``) unless that name is taken."""
    out, taken = {}, set(columns)
    for c in columns:
        base = re.sub(r"\(.*\)$", "", c)
        if base != c and base not in taken:
            out[c] = base
            taken.add(base)
    return out


def load(kind: str, source: str, code: int, market: int, day: int) -> pl.DataFrame | None:
    """One DataYes entry -> feitu columns."""
    df = read(source)
    if df.is_empty():
        return None
    df = from_datayes(df, kind)
    if kind == "transaction" and market == SSE and "functioncode" in df.columns:
        # DataYesSharesL2 has a few SSE trades typed Z (meaning unknown): kept, but as trade_type 0 so no fill reader counts them
        z = pl.col("functioncode") == "Z"
        parts = [convert(df.filter(~z), kind, code, market, day),
                 convert(df.filter(z), kind, code, market, day)]
        if parts[1] is not None:
            parts[1] = parts[1].with_columns(pl.lit(0, pl.Int8).alias("trade_type"))
        parts = [x for x in parts if x is not None]
        return pl.concat(parts).sort("seq_id") if parts else None
    return convert(df, kind, code, market, day)


def _write_part(task: dict[str, Any]) -> dict[str, Any]:
    """Worker process: one part of DataYes entries."""
    return write_part(task, load)


def resolve_dates(root: Path, spec: str) -> list[str]:
    """``all``, ``FROM-TO`` (packages present in the range) or a comma list."""
    have = sorted({p.name[:8] for p in root.iterdir() if re.fullmatch(r"\d{8}(\.zip)?", p.name)})
    if spec == "all":
        return have
    if re.fullmatch(r"\d{8}-\d{8}", spec):
        a, b = spec.split("-")
        return [d for d in have if a <= d <= b]
    return [d.strip() for d in spec.split(",") if d.strip()]


def feitu_day(out: Path, date: str) -> bool:
    """Tell whether ``out`` already holds this day from feitu (minute-named parts, not part-SH-NNN)."""
    return any(not re.fullmatch(r"part-(SH|SZ)-\d{3}\.parquet", f.name) for f in out.glob(f"kind=*/date={date}/part-*.parquet"))


def plan(root: Path, dates: list[str], out: Path, pattern: re.Pattern[str], kinds: tuple[str, ...], chunk: int,
         overwrite: bool) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]], list[str]]:
    """Every (day, kind, exchange, symbol chunk) part to write, and the manifest rows of parts already on disk."""
    tasks: list[dict[str, Any]] = []
    done: dict[tuple[str, str], list[dict[str, Any]]] = {}
    found = []
    for date in dates:
        if feitu_day(out, date):
            continue  # feitu has this day
        if not overwrite and all((out / f"kind={k}" / f"date={date}" / "_manifest.parquet").exists() for k in kinds):
            continue  # converted on an earlier run
        pkg = package(root, date)
        if pkg is None:
            print(f"{date}: no package under {root}")
            continue
        try:
            idx = index(pkg)
        except zipfile.BadZipFile as e:  # still being copied in: no central directory yet
            print(f"{date}: {pkg.name} not readable yet ({e}), skipped")
            continue
        print(f"{date}: {pkg.name}, {len(idx)} entries, {sum(b for _, b in idx.values()) / 1e9:.1f} GB compressed")
        found.append(date)
        prefix = f"{pkg}!" if pkg.is_file() else ""
        for kind in kinds:
            sub = next(s for s, k in SUBS.items() if k == kind)
            dst_dir = out / f"kind={kind}" / f"date={date}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            check_partition(dst_dir)
            done[(date, kind)] = []
            for market, tag in ((SSE, "SH"), (SZSE, "SZ")):
                todo = sorted((sym, int(sym[:6]), prefix + ref, b) for (sym, s), (ref, b) in idx.items()
                              if s == sub and sym.endswith(tag) and pattern.match(sym))
                if not todo:
                    print(f"[{kind}] no {tag} entries in {pkg.name}")
                    continue
                for ci in range(0, len(todo), chunk):
                    rows = todo[ci:ci + chunk]
                    part = [(sym, code, ref) for sym, code, ref, _ in rows]
                    dst = dst_dir / f"part-{tag}-{ci // chunk:03d}.parquet"
                    row = part_row(tag, part, sum(b for *_, b in rows), f"datayes:{root.name}")
                    if dst.exists() and not overwrite:
                        row["rows"], row["skipped"] = pq.read_metadata(dst).num_rows, True
                        done[(date, kind)].append(row)
                        continue
                    tasks.append({"kind": kind, "market": market, "day": int(date), "date": date, "dst": str(dst),
                                  "part": part, "row": row})
    return tasks, done, found


def show_codes(df: pl.DataFrame) -> None:
    """One line per code column (value counts) and per id column pair (share equal), for a raw order / trade file."""
    for c in ("type", "oritype(type)", "tickbsflag"):
        if c in df.columns:
            vc = df[c].value_counts(sort=True).head(8)
            print(f"  {c}: " + "  ".join(f"{v!r}={n}" for v, n in vc.iter_rows()))
    pairs = [("bizindex", "seq"), ("bizindex", "orderno"), ("seq", "orderno"), ("type", "oritype(type)")]
    eq = [f"{a}=={b} {(df[a] == df[b]).mean():.3f}" for a, b in pairs if a in df.columns and b in df.columns]
    nz = [f"{c}!=0 {(df[c].cast(pl.Float64, strict=False).fill_null(0) != 0).mean():.3f}"
          for c in ("buyorderno", "sellorderno", "orderno", "seq") if c in df.columns]
    print("  " + "  ".join(eq + nz))
    ids = [c for c in ("time", "bizindex", "seq", "orderno", "buyorderno", "sellorderno", "type", "tickbsflag", "price", "qty")
           if c in df.columns]
    mid = df.height // 2
    with pl.Config(tbl_cols=12, tbl_width_chars=160, tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print("\n".join("  " + line for line in str(df.select(ids).slice(mid, 4)).splitlines()))
    print("  time " + " .. ".join(df["time"].sort().gather([0, df.height - 1]).to_list()) if "time" in df.columns else "")


def probe(root: Path, date: str, pattern: re.Pattern[str]) -> None:
    """Print the header, two raw rows and three converted rows of one liquid SH and SZ name per kind; write nothing."""
    pkg = package(root, date)
    if pkg is None:
        sys.exit(f"no package for {date} under {root}")
    idx = index(pkg)
    prefix = f"{pkg}!" if pkg.is_file() else ""
    print(f"{pkg}: {len(idx)} entries, subs {sorted({s for _, s in idx})}")
    for market, tag, liquid in ((SSE, "SH", "600000.SH"), (SZSE, "SZ", "000001.SZ")):
        syms = sorted({sym for sym, _ in idx if sym.endswith(tag) and pattern.match(sym)})
        pick = liquid if liquid in syms else (syms[0] if syms else None)
        if pick is None:
            print(f"no {tag} symbol")
            continue
        for sub, kind in SUBS.items():
            hit = idx.get((pick, sub))
            print(f"\n== {kind} {pick} {hit[0] if hit else None}")
            if hit is None:
                continue
            df = read(prefix + hit[0])
            print(f"rows {df.height}; columns: {df.columns}")
            print(df.head(2))
            if kind != "quotation":
                show_codes(df)
            df = from_datayes(df, kind)
            if kind == "quotation":
                m, unused = quote_mapping(df.columns)
                print("UNMAPPED columns:", unused)
                print("feitu fields left at 0:", [k for k in QUOTE_MAP if k not in m])
            with pl.Config(tbl_cols=20, tbl_width_chars=200):
                print(CONVERT[kind](df, int(pick[:6]), market, int(date)).head(3))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--date", required=True, help="YYYYMMDD, comma list, FROM-TO, or all")
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--kinds", default="quotation,order,transaction")
    ap.add_argument("--symbols", default=STOCKS.pattern, help="regex on the WindCode")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 8, 96), help="worker processes = cores used")
    ap.add_argument("--chunk", type=int, default=250, help="symbols per Parquet part")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--probe", action="store_true", help="print headers, mapping and three converted rows; write nothing")
    args = ap.parse_args()
    kinds = tuple(k.strip() for k in args.kinds.split(",") if k.strip())
    pattern = re.compile(args.symbols)
    dates = resolve_dates(args.root, args.date)
    if not dates:
        sys.exit(f"no packages for {args.date} under {args.root}")
    if args.probe:
        probe(args.root, dates[0], pattern)
        return
    tasks, done, found = plan(args.root, dates, args.out, pattern, kinds, args.chunk, args.overwrite)
    print(f"{len(found)} of {len(dates)} day(s) to convert: {', '.join(found) or '-'} (the rest are already on disk)")
    if not found:
        return
    run_parts(tasks, done, args.out, found, kinds, args.workers, _write_part)  # manifests + quality per day too


if __name__ == "__main__":
    main()
