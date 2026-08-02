"""Probe real feitu capnp dumps and verify them against the exchange spec.

Runs **on the dev box**, next to the data. Deliberately depends only on
``pycapnp`` (+ ``zstandard`` for ``.zst``) and the standard library, so it does
not need the crucible stack installed there.

What it checks, and why each matters
------------------------------------
1. **Timestamp unit.** v1/v2 stamp milliseconds, v3 nanoseconds. Decoded
   timestamps must land on a plausible trading date; a 10^6 error puts them in
   1970 or the far future.
2. **Causality.** ``exchange_ts <= arrival_ts <= broker_ts``. A violation is
   clock skew between the exchange stamp and the capture box, and it must be
   surfaced rather than sorted away -- the point-in-time gate keys on arrival.
3. **Wire latency.** ``arrival_ts - exchange_ts`` distribution. This is the
   quantity that decides whether a tick backtest is honest; it is also the
   first thing to change when the network changes.
4. **Float price loss.** The archive stores prices as ``Float64``. Measures how
   many would truncate a tick low, against the ~5% predicted from the spec's
   integer scaling.
5. **Sequence contiguity.** ``(channelId, seqId)`` is the exchange's own replay
   order and is contiguous within a channel. Gaps are packet loss, not quiet
   periods.
6. **Cancel split by venue.** SSE cancels ride the order stream
   (``updateType==2``); SZSE cancels ride the trade stream (``tradeType==2``).
   Confirms the vendor preserves the divergence rather than flattening it.

Usage::

    python3 probe_feitu.py /media/zzz/Stocks --limit 200000
"""

from __future__ import annotations

import argparse
import datetime as dt
import lzma
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

TYPES = {
    "v2_quotation": ("quotation_v2_capnp", "QuotationV2"),
    "v3_quotation": ("quotation_v3_capnp", "QuotationV3"),
    "index": ("index_capnp", "Index"),
    "v3_index": ("index_v3_capnp", "IndexV3"),
    "order": ("order_capnp", "Order"),
    "v2_order": ("order_v2_capnp", "OrderV2"),
    "v3_order": ("order_v3_capnp", "OrderV3"),
    "transaction": ("transaction_capnp", "Transaction"),
    "v2_transaction": ("transaction_v2_capnp", "TransactionV2"),
    "v3_transaction": ("transaction_v3_capnp", "TransactionV3"),
}
UNIT_NS = {k: (1 if k.startswith("v3") else 1_000_000) for k in TYPES}
MARKET = {3553: "XSHG", 3554: "XSHE"}
PRICE_SCALE = {"XSHG": 10**5, "XSHE": 10**4}


def detect_type(name: str) -> str | None:
    low = name.lower()
    for key in sorted(TYPES, key=len, reverse=True):
        if key in low:
            return key
    return None


def read_blob(path: Path) -> bytes:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as fh:
            return fh.read()
    if path.suffix == ".zst":
        import zstandard

        with path.open("rb") as fh:
            return zstandard.ZstdDecompressor().stream_reader(fh).read()
    raise ValueError(f"unsupported extension {path.suffix!r}")


_SCHEMA_DIR: Path | None = None


def _load_schemas(key: str):
    """Return ``(ExportedData, inner_struct)`` for a type key.

    pycapnp 2.x removed the implicit ``import foo_capnp`` hook the vendor demo
    relies on, so load the schema files explicitly. Falls back to the hook when
    running against pycapnp 1.x.
    """
    import capnp  # noqa: F401

    mod_name, struct_name = TYPES[key]
    file_stem = mod_name.removesuffix("_capnp")

    if _SCHEMA_DIR is not None:
        env = capnp.load(str(_SCHEMA_DIR / "export.capnp"))
        inner = capnp.load(str(_SCHEMA_DIR / f"{file_stem}.capnp"))
        return env.ExportedData, getattr(inner, struct_name)

    import importlib

    env = importlib.import_module("export_capnp")
    inner = importlib.import_module(mod_name)
    return env.ExportedData, getattr(inner, struct_name)


def iter_records(path: Path, key: str, limit: int):
    exported_data, struct = _load_schemas(key)
    blob = read_blob(path)
    for i, env in enumerate(exported_data.read_multiple_bytes_packed(blob)):
        if i >= limit:
            return
        yield struct.from_bytes_packed(env.data)


def fmt_ns(ns: int) -> str:
    if ns <= 0:
        return "0"
    try:
        return dt.datetime.fromtimestamp(ns / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (OSError, ValueError, OverflowError):
        return f"<out of range: {ns}>"


def probe(path: Path, key: str, limit: int) -> None:
    print(f"\n{'=' * 78}\n{path.name}   [{key}]\n{'=' * 78}")
    unit = UNIT_NS[key]
    is_index = "index" in key

    n = 0
    venues: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    cancels_by_venue: dict[str, Counter[str]] = defaultdict(Counter)
    latency: list[int] = []
    skew = 0
    ex_lo = ex_hi = None
    seq_by_channel: dict[int, list[int]] = defaultdict(list)
    px_total = px_lossy = 0

    for r in iter_records(path, key, limit):
        n += 1
        ex = int(getattr(r, "time", 0) or 0) * unit
        ar = int(getattr(r, "spiderTs", 0) or 0) * unit
        br = int(getattr(r, "serverTs", 0) or 0) * unit

        if ex > 0:
            ex_lo = ex if ex_lo is None else min(ex_lo, ex)
            ex_hi = ex if ex_hi is None else max(ex_hi, ex)
        if ex > 0 and ar > 0:
            latency.append(ar - ex)
            if ar < ex or (br > 0 and br < ar):
                skew += 1

        venue = MARKET.get(int(getattr(r, "marketId", 0) or 0), "other")
        venues[venue] += 1

        if not is_index:
            ch = int(getattr(r, "channelId", 0) or 0)
            sq = int(getattr(r, "seqId", 0) or 0)
            if sq:
                seq_by_channel[ch].append(sq)

            if "order" in key:
                ut = int(getattr(r, "updateType", 0) or 0)
                label = {1: "add", 2: "cancel"}.get(ut, f"status/{ut}")
            else:
                tt = int(getattr(r, "tradeType", 0) or 0)
                label = {1: "trade", 2: "cancel"}.get(tt, f"status/{tt}")
            kinds[label] += 1
            if label == "cancel":
                cancels_by_venue[venue]["cancel"] += 1

            px = float(getattr(r, "price", 0.0) or 0.0)
            if px > 0 and venue in PRICE_SCALE:
                s = PRICE_SCALE[venue]
                px_total += 1
                if int(px * s) != round(px * s):
                    px_lossy += 1

    if n == 0:
        print("  no records decoded")
        return

    print(f"records            : {n:,}")
    print(f"venues             : {dict(venues)}")

    # 1 + 2: unit sanity and causality
    print(f"exchange_ts range  : {fmt_ns(ex_lo or 0)}  ..  {fmt_ns(ex_hi or 0)}")
    plausible = ex_lo and dt.datetime(2015, 1, 1) <= dt.datetime.fromtimestamp(ex_lo / 1e9) <= dt.datetime(2035, 1, 1)
    print(f"unit check         : {'PASS' if plausible else 'FAIL'}  (assumed {unit} ns/tick)")
    print(f"causality breaches : {skew:,} of {len(latency):,}"
          f"{'  <-- clock skew' if skew else ''}")

    # 3: wire latency -- the number the PIT gate depends on
    if latency:
        latency.sort()
        q = lambda p: latency[min(len(latency) - 1, int(len(latency) * p))]  # noqa: E731
        print("wire latency (exchange -> colo):")
        print(f"  median {q(0.50) / 1e6:9.3f} ms   p90 {q(0.90) / 1e6:9.3f} ms")
        print(f"  p99    {q(0.99) / 1e6:9.3f} ms   max {latency[-1] / 1e6:9.3f} ms")
        print(f"  mean   {statistics.mean(latency) / 1e6:9.3f} ms")

    # 4: float price loss
    if px_total:
        pct = px_lossy / px_total
        print(f"float price loss   : {px_lossy:,}/{px_total:,} ({pct:.2%}) truncate a tick low")
        print(f"                     (spec predicts ~5%; round(), never int())")

    # 5: sequence contiguity
    if seq_by_channel:
        total_gap = 0
        worst = None
        for ch, seqs in seq_by_channel.items():
            u = sorted(set(seqs))
            missing = (u[-1] - u[0] + 1) - len(u)
            total_gap += missing
            if missing and (worst is None or missing > worst[1]):
                worst = (ch, missing)
        print(f"channels           : {len(seq_by_channel)}")
        print(f"sequence gaps      : {total_gap:,} missing"
              f"{f'  (worst channel {worst[0]}: {worst[1]:,})' if worst else '  -- contiguous'}")

    # 6: cancel split
    if kinds:
        print(f"record types       : {dict(kinds)}")
        if cancels_by_venue:
            print(f"cancels by venue   : "
                  f"{ {v: c['cancel'] for v, c in cancels_by_venue.items()} }")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path, help="directory holding the dump subdirs")
    ap.add_argument("--limit", type=int, default=200_000, help="records per file")
    ap.add_argument("--schema-dir", type=Path, default=None, help="dir with *.capnp")
    args = ap.parse_args()

    global _SCHEMA_DIR
    if args.schema_dir:
        _SCHEMA_DIR = args.schema_dir.resolve()
        sys.path.insert(0, str(_SCHEMA_DIR))

    files: list[tuple[Path, str]] = []
    for p in sorted(args.root.rglob("*")):
        if p.suffix not in (".zst", ".xz") or not p.is_file():
            continue
        key = detect_type(p.name) or detect_type(p.parent.name)
        if key:
            files.append((p, key))

    print(f"root  : {args.root}")
    print(f"found : {len(files)} dump files")
    by_kind: Counter[str] = Counter(k for _, k in files)
    print(f"kinds : {dict(by_kind)}")

    # One file per kind keeps the probe quick; the point is verifying the
    # contract, not scanning the archive.
    seen: set[str] = set()
    for p, key in files:
        if key in seen:
            continue
        seen.add(key)
        try:
            probe(p, key, args.limit)
        except Exception as exc:  # noqa: BLE001
            print(f"\n{p.name} [{key}] FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
