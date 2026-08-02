"""Determine the order<->trade linkage key before building any book.

An MBO replay needs to know which field on a trade points back at the resting
order it consumed. Get it wrong and the book still "works" -- orders are added,
quantities are decremented, a spread comes out -- but the decrements land on the
wrong orders. Nothing raises. Every depth, queue-position and microprice feature
is then quietly fiction.

The two venues do not agree, and the vendor schema does not say which is which:

* ``OrderV2`` carries **three** candidate identifiers: ``orderId`` (订单编号),
  ``seqId`` (逐笔编号, per-channel) and ``indexId`` (委托编号).
* ``TransactionV2`` carries ``buySeqId`` / ``sellSeqId`` (买/卖方订单编号).

So this measures hit rates for every candidate, per venue, and reports which
one actually resolves. Also profiles the fields a book build depends on:
``orderType`` (market / limit / best-price) and cancel routing.

Run on the box against one minute of data.
"""

from __future__ import annotations

import argparse
import lzma
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

MARKET = {3553: "XSHG", 3554: "XSHE"}
_SCHEMA: Path | None = None


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


def _iter(path: Path, stem: str, struct: str, limit: int):
    import capnp  # noqa: F401

    env = _load("export", "ExportedData")
    inner = _load(stem, struct)
    for i, e in enumerate(env.read_multiple_bytes_packed(_blob(path))):
        if i >= limit:
            return
        yield inner.from_bytes_packed(e.data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("order", type=Path)
    ap.add_argument("trade", type=Path)
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=400_000)
    a = ap.parse_args()

    global _SCHEMA
    _SCHEMA = a.schema_dir.resolve()

    # ---- collect order identifiers per venue -------------------------------
    ids: dict[str, dict[str, set[int]]] = {
        v: {"orderId": set(), "seqId": set(), "indexId": set()} for v in MARKET.values()
    }
    otype: dict[str, Counter[int]] = {v: Counter() for v in MARKET.values()}
    utype: dict[str, Counter[int]] = {v: Counter() for v in MARKET.values()}
    # Does a cancel carry the same id as the add it removes?
    cancel_ids: dict[str, set[int]] = {v: set() for v in MARKET.values()}
    add_ids: dict[str, set[int]] = {v: set() for v in MARKET.values()}

    for r in _iter(a.order, "order_v2", "OrderV2", a.limit):
        v = MARKET.get(int(r.marketId or 0))
        if v is None:
            continue
        ut = int(r.updateType or 0)
        utype[v][ut] += 1
        oid, sq, ix = int(r.orderId or 0), int(r.seqId or 0), int(r.indexId or 0)
        ids[v]["orderId"].add(oid)
        ids[v]["seqId"].add(sq)
        ids[v]["indexId"].add(ix)
        if ut == 1:
            otype[v][int(r.orderType or 0)] += 1
            add_ids[v].add(sq)
        elif ut == 2:
            cancel_ids[v].add(sq)

    # ---- test each candidate against trade references ----------------------
    hits: dict[str, Counter[str]] = {v: Counter() for v in MARKET.values()}
    n_trades: Counter[str] = Counter()
    zero_ref: Counter[str] = Counter()

    for r in _iter(a.trade, "transaction_v2", "TransactionV2", a.limit):
        v = MARKET.get(int(r.marketId or 0))
        if v is None or int(r.tradeType or 0) != 1:
            continue
        n_trades[v] += 1
        b, s = int(r.buySeqId or 0), int(r.sellSeqId or 0)
        if b == 0 and s == 0:
            zero_ref[v] += 1
            continue
        for cand in ("orderId", "seqId", "indexId"):
            pool = ids[v][cand]
            if b in pool:
                hits[v][f"{cand}.buy"] += 1
            if s in pool:
                hits[v][f"{cand}.sell"] += 1

    # ---- report ------------------------------------------------------------
    print("=" * 74)
    print("order<->trade linkage: which id does buySeqId/sellSeqId resolve to?")
    print("=" * 74)
    for v in MARKET.values():
        n = n_trades[v]
        if not n:
            print(f"\n{v}: no trades in sample")
            continue
        print(f"\n{v}   trades={n:,}   both-refs-zero={zero_ref[v]:,} "
              f"({zero_ref[v] / n:.1%})")
        for cand in ("orderId", "seqId", "indexId"):
            hb = hits[v][f"{cand}.buy"] / n
            hs = hits[v][f"{cand}.sell"] / n
            flag = "  <== LINKAGE KEY" if min(hb, hs) > 0.80 else ""
            print(f"   {cand:>9}: buy {hb:6.1%}   sell {hs:6.1%}{flag}")
        print(f"   distinct orderId={len(ids[v]['orderId']):,}  "
              f"seqId={len(ids[v]['seqId']):,}  indexId={len(ids[v]['indexId']):,}")

    print()
    print("=" * 74)
    print("fields a book build depends on")
    print("=" * 74)
    names = {1: "market", 2: "limit", 3: "best-price", 0: "unknown"}
    for v in MARKET.values():
        if not utype[v]:
            continue
        tot_u = sum(utype[v].values())
        tot_o = sum(otype[v].values()) or 1
        print(f"\n{v}")
        print("   updateType: " + "  ".join(
            f"{k}={c:,}({c / tot_u:.1%})" for k, c in sorted(utype[v].items())))
        print("   orderType : " + "  ".join(
            f"{names.get(k, k)}={c:,}({c / tot_o:.1%})" for k, c in sorted(otype[v].items())))
        overlap = len(add_ids[v] & cancel_ids[v])
        print(f"   cancels sharing a seqId with an add: {overlap:,} "
              f"of {len(cancel_ids[v]):,} cancels")
        if cancel_ids[v] and overlap == 0:
            print("     -> cancels carry their OWN seqId; the removed order must be")
            print("        found via a different field (orderId / indexId).")


if __name__ == "__main__":
    main()
