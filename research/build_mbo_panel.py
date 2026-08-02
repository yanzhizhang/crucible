"""Build a 3-second order-book panel by MBO replay. Runs on the dev box.

Decodes both tick streams, applies the venue-specific order keys measured by
``probe_mbo_linkage.py``, replays the book per channel in ``(channel, seqId)``
order, and snapshots at each slot boundary.

Two implementation constraints worth stating
--------------------------------------------
**Replay is per channel.** ``replay_events`` advances its slot cursor as events
arrive, so it needs monotonically increasing time. Globally sorting by
``(channel, seqId)`` groups all of channel 1 before channel 2, which is not
time-ordered. Each channel is therefore replayed independently -- which is also
correct, since a symbol lives in exactly one channel, and it parallelises.

**Symbols are filtered before decode.** One minute is ~2.5M events; held as
Python tuples that is over a gigabyte per worker. Restricting to an allowlist
of liquid names first cuts it by more than an order of magnitude.

Usage::

    python build_mbo_panel.py --order-dir ... --trade-dir ... \\
        --symbols top300.txt --schema-dir ... --out mbo.parquet \\
        --from 1000 --to 1100 --workers 6
"""

from __future__ import annotations

import argparse
import io
import lzma
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

_STAMP = re.compile(r"(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})")
MARKET = {3553: "XSHG", 3554: "XSHE"}
PRICE_SCALE = {"XSHG": 10**5, "XSHE": 10**4}
_SCHEMA: Path | None = None


def _key_of(p: Path) -> str | None:
    m = _STAMP.search(p.name)
    return "_".join(m.groups()) if m else None


def _hhmm(k: str) -> int:
    return int(k[-5:-3] + k[-2:])


def _load(stem: str, struct: str):
    import capnp

    assert _SCHEMA is not None
    return getattr(capnp.load(str(_SCHEMA / f"{stem}.capnp")), struct)


def _blob(p: Path) -> bytes:
    if p.suffix == ".xz":
        with lzma.open(p, "rb") as fh:
            return fh.read()
    import zstandard

    with p.open("rb") as fh:
        return zstandard.ZstdDecompressor().stream_reader(fh).read()


def _iter(path: Path, stem: str, struct: str):
    import capnp  # noqa: F401

    env = _load("export", "ExportedData")
    inner = _load(stem, struct)
    for e in env.read_multiple_bytes_packed(_blob(path)):
        yield inner.from_bytes_packed(e.data)


def _px(raw: float, ven: str) -> float:
    s = PRICE_SCALE.get(ven)
    return round(raw * s) / s if s else raw


def collect_events(order_p: Path, trade_p: Path, keep: set[str]) -> dict[int, list[tuple]]:
    """Decode both streams into per-channel event lists.

    Order keys follow the measured venue divergence:

    * SSE  -- ``orderId`` identifies the order; cancels arrive on the order
      stream with ``updateType == 2`` and carry the cancelled order's
      ``orderId``.
    * SZSE -- ``seqId`` identifies the order; the order stream carries adds
      only, and cancels arrive on the **trade** stream with ``tradeType == 2``,
      referencing the order through ``buySeqId`` / ``sellSeqId``.
    """
    by_channel: dict[int, list[tuple]] = defaultdict(list)

    for r in _iter(order_p, "order_v2", "OrderV2"):
        ven = MARKET.get(int(r.marketId or 0))
        if ven is None:
            continue
        sym = f"{int(r.symbolId or 0):06d}"
        if sym not in keep:
            continue
        ut = int(r.updateType or 0)
        if ut not in (1, 2):
            continue  # product-status record
        d = int(r.dir or 0)
        side = "buy" if d == 1063 else "sell" if d == 1550 else None
        key = int(r.orderId or 0) if ven == "XSHG" else int(r.seqId or 0)
        by_channel[int(r.channelId or 0)].append((
            int(r.channelId or 0), int(r.seqId or 0), int(r.spiderTs or 0) * 1_000_000,
            sym, ven, "add" if ut == 1 else "cancel",
            _px(float(r.price or 0.0), ven), int(r.volume or 0), side, key, 0, 0,
        ))

    for r in _iter(trade_p, "transaction_v2", "TransactionV2"):
        ven = MARKET.get(int(r.marketId or 0))
        if ven is None:
            continue
        sym = f"{int(r.symbolId or 0):06d}"
        if sym not in keep:
            continue
        tt = int(r.tradeType or 0)
        if tt not in (1, 2):
            continue
        d = int(r.dir or 0)
        side = "buy" if d == 1063 else "sell" if d == 1550 else None
        b, s = int(r.buySeqId or 0), int(r.sellSeqId or 0)
        ch = int(r.channelId or 0)
        ts = int(r.spiderTs or 0) * 1_000_000
        qty = int(r.volume or 0)
        price = _px(float(r.price or 0.0), ven)
        if tt == 2:
            # SZSE cancel riding the trade stream: one of the two refs names
            # the order being pulled.
            by_channel[ch].append(
                (ch, int(r.seqId or 0), ts, sym, ven, "cancel", price, qty, side, b or s, 0, 0)
            )
        else:
            by_channel[ch].append(
                (ch, int(r.seqId or 0), ts, sym, ven, "trade", price, qty, side, 0, b, s)
            )

    return by_channel


def _one_chunk(args: tuple[list[tuple[str, str]], int, str, tuple[str, ...], int]) -> bytes | None:
    """Replay a contiguous block of minutes, emitting only the non-warmup part.

    A book started cold has no resting orders, so early trades reference orders
    it never saw and depth is understated until the queue refills. Each chunk
    therefore replays ``n_warm`` minutes ahead of its output range purely to
    populate the book, and discards those slots.

    Chunks must be contiguous *and* replayed in ``seqId`` order within each
    channel; that is why the unit of parallelism is a block of minutes rather
    than a single minute.
    """
    pairs, n_warm, schema_dir, keep, slot_ns = args
    global _SCHEMA
    _SCHEMA = Path(schema_dir)
    sys.path.insert(0, str(_SCHEMA))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mbo_book import replay_events

    try:
        keep_set = set(keep)
        merged: dict[int, list[tuple]] = defaultdict(list)
        warm_until = 0
        for i, (op, tp) in enumerate(pairs):
            chans = collect_events(Path(op), Path(tp), keep_set)
            for ch, evs in chans.items():
                merged[ch].extend(evs)
            if i == n_warm - 1:
                warm_until = max(
                    (e[2] for evs in merged.values() for e in evs), default=0
                )

        rows: list[dict] = []
        for evs in merged.values():
            evs.sort(key=lambda e: e[1])
            rows.extend(replay_events(evs, slot_ns))
        if warm_until:
            rows = [r for r in rows if r["slot"] > warm_until]
        if not rows:
            return None
        buf = io.BytesIO()
        pl.DataFrame(rows).write_parquet(buf)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED chunk {pairs[0][0]}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order-dir", type=Path, required=True)
    ap.add_argument("--trade-dir", type=Path, required=True)
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--symbols", type=Path, required=True, help="one 6-digit code per line")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="t_from", type=int, default=0)
    ap.add_argument("--to", dest="t_to", type=int, default=2359)
    ap.add_argument("--slot-seconds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=10, help="output minutes per chunk")
    ap.add_argument("--warmup", type=int, default=8, help="minutes replayed to fill the book")
    a = ap.parse_args()

    keep = tuple(
        s.strip() for s in a.symbols.read_text().split() if s.strip() and not s.startswith("#")
    )
    print(f"symbols: {len(keep)}")

    o_by = {k: p for p in a.order_dir.iterdir() if (k := _key_of(p))}
    t_by = {k: p for p in a.trade_dir.iterdir() if (k := _key_of(p))}
    keys = sorted(set(o_by) & set(t_by))
    keys = [k for k in keys if a.t_from <= _hhmm(k) <= a.t_to]
    if not keys:
        print("no minutes matched")
        return
    print(f"minutes: {len(keys)}  ({keys[0]} .. {keys[-1]})")

    slot_ns = a.slot_seconds * 1_000_000_000
    sd = str(a.schema_dir.resolve())

    # Contiguous chunks, each preceded by `warmup` minutes used only to fill
    # the book. Chunks never straddle a session break, so days are split apart.
    by_day: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_day[k[:10]].append(k)

    jobs = []
    for day_keys in by_day.values():
        for i in range(0, len(day_keys), a.chunk):
            block = day_keys[i : i + a.chunk]
            lo = max(0, i - a.warmup)
            warm = day_keys[lo:i]
            pairs = [(str(o_by[k]), str(t_by[k])) for k in (*warm, *block)]
            jobs.append((pairs, len(warm), sd, keep, slot_ns))
    print(f"chunks: {len(jobs)}  (chunk={a.chunk} min, warmup={a.warmup} min)")

    parts: list[pl.DataFrame] = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for blob in ex.map(_one_chunk, jobs):
            done += 1
            if blob:
                parts.append(pl.read_parquet(io.BytesIO(blob)))
            if done % 2 == 0:
                print(f"  {done}/{len(keys)}  ->  {sum(p.height for p in parts):,} rows", flush=True)

    if not parts:
        print("nothing decoded")
        return

    panel = pl.concat(parts, how="vertical").sort(["slot", "symbol"], maintain_order=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(a.out, compression="zstd")

    unres = int(panel["unresolved"].max() or 0)
    print(f"\nMBO panel: {panel.height:,} rows x {panel.width} cols")
    print(f"symbols {panel['symbol'].n_unique():,}   slots {panel['slot'].n_unique():,}")
    print(f"max unresolved refs on a symbol: {unres:,}")
    if unres:
        print("  (references to orders placed before the window -- warm from the open to clear)")
    have_quote = panel.filter((pl.col("best_bid") > 0) & (pl.col("best_ask") > 0)).height
    print(f"rows with a two-sided book: {have_quote:,} ({have_quote / panel.height:.1%})")
    print(f"written: {a.out} ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
