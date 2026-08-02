"""Verify two hypotheses raised by the first real-data probe.

**H1 -- the timestamp chain is exchange -> serverTs -> spiderTs, not the
reverse.** The first probe flagged 100% "causality breaches", which is not
plausible as clock skew. Re-reading the vendor comments: ``spiderTs`` is
接收时间 (our capture) and ``serverTs`` is 券商服务器收到数据的时间 (the broker
server's receipt). The broker sits *upstream* of the capture box, so the true
order is exchange -> broker -> spider. Measures each leg separately instead of
assuming either direction.

**H2 -- order and transaction share one sequence space per channel.** The probe
reported ~189k "missing" sequence numbers in the order stream alone. If the two
tick streams share a per-channel ``ApplSeqNum`` space (as the exchange specs
imply), then each stream read alone looks full of holes while the *merged*
stream is contiguous. Merges both and re-measures.

Run on the box, next to the data.
"""

from __future__ import annotations

import argparse
import lzma
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

MARKET = {3553: "XSHG", 3554: "XSHE"}


def load(schema_dir: Path, stem: str, struct: str):
    import capnp

    return getattr(capnp.load(str(schema_dir / f"{stem}.capnp")), struct)


def read_blob(p: Path) -> bytes:
    if p.suffix == ".xz":
        with lzma.open(p, "rb") as fh:
            return fh.read()
    import zstandard

    with p.open("rb") as fh:
        return zstandard.ZstdDecompressor().stream_reader(fh).read()


def iter_recs(path: Path, schema_dir: Path, stem: str, struct: str, limit: int):
    import capnp  # noqa: F401

    env = load(schema_dir, "export", "ExportedData")
    inner = load(schema_dir, stem, struct)
    for i, e in enumerate(env.read_multiple_bytes_packed(read_blob(path))):
        if i >= limit:
            return
        yield inner.from_bytes_packed(e.data)


def pct(vals: list[int], p: float) -> int:
    if not vals:
        return 0
    return sorted(vals)[min(len(vals) - 1, int(len(vals) * p))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("order", type=Path)
    ap.add_argument("transaction", type=Path)
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300_000, help="cap for the H1 latency pass")
    ap.add_argument(
        "--full-limit",
        type=int,
        default=100_000_000,
        help="cap for the H2 sequence pass; keep large so whole files are read",
    )
    a = ap.parse_args()

    # ---------------- H1: which timestamp leads? ----------------
    print("=" * 74)
    print("H1  timestamp chain: exchange -> ? -> ?")
    print("=" * 74)

    for label, path, stem, struct in (
        ("order", a.order, "order_v2", "OrderV2"),
        ("transaction", a.transaction, "transaction_v2", "TransactionV2"),
    ):
        ex_to_srv: list[int] = []
        srv_to_spd: list[int] = []
        ex_to_spd: list[int] = []
        srv_before_spd = spd_before_srv = 0

        for r in iter_recs(path, a.schema_dir, stem, struct, a.limit):
            ex = int(r.time or 0)
            spd = int(r.spiderTs or 0)
            srv = int(r.serverTs or 0)
            if not (ex and spd and srv):
                continue
            ex_to_srv.append(srv - ex)
            srv_to_spd.append(spd - srv)
            ex_to_spd.append(spd - ex)
            if srv <= spd:
                srv_before_spd += 1
            else:
                spd_before_srv += 1

        n = len(ex_to_spd)
        if not n:
            print(f"\n{label}: no complete triples")
            continue
        print(f"\n{label}  (n={n:,})")
        print(f"  serverTs <= spiderTs : {srv_before_spd:,} ({srv_before_spd / n:.1%})")
        print(f"  spiderTs <  serverTs : {spd_before_srv:,} ({spd_before_srv / n:.1%})")
        for name, v in (
            ("exchange -> server", ex_to_srv),
            ("server   -> spider", srv_to_spd),
            ("exchange -> spider", ex_to_spd),
        ):
            print(
                f"  {name}: median {pct(v, 0.5):>7,} ms   "
                f"p90 {pct(v, 0.9):>7,}   p99 {pct(v, 0.99):>8,}   max {max(v):>8,}"
            )

    # ---------------- H2: shared sequence space? ----------------
    print()
    print("=" * 74)
    print("H2  do order and transaction share one (channel, seqId) space?")
    print("=" * 74)

    # Collect (channel, seq, exchange_ts) for the WHOLE file. Truncating each
    # stream at a fixed record count cuts them at different wall-clock times,
    # so the merge would compare misaligned windows and invent gaps that are
    # an artefact of the slice rather than a property of the data.
    raw: dict[str, list[tuple[int, int, int]]] = {"order": [], "transaction": []}
    venue_of_channel: dict[int, str] = {}

    for label, path, stem, struct in (
        ("order", a.order, "order_v2", "OrderV2"),
        ("transaction", a.transaction, "transaction_v2", "TransactionV2"),
    ):
        for r in iter_recs(path, a.schema_dir, stem, struct, a.full_limit):
            ch = int(r.channelId or 0)
            sq = int(r.seqId or 0)
            ts = int(r.time or 0)
            if sq and ts:
                raw[label].append((ch, sq, ts))
                venue_of_channel.setdefault(ch, MARKET.get(int(r.marketId or 0), "?"))
        print(f"  read {label}: {len(raw[label]):,} sequenced records")

    # Restrict both streams to the overlap of their exchange-time ranges.
    lo = max(min(t for _, _, t in v) for v in raw.values() if v)
    hi = min(max(t for _, _, t in v) for v in raw.values() if v)
    print(f"  common exchange-time window: {lo} .. {hi}  ({(hi - lo) / 1000:.2f}s)")

    per_stream: dict[str, dict[int, set[int]]] = {
        "order": defaultdict(set),
        "transaction": defaultdict(set),
    }
    for label, rows in raw.items():
        for ch, sq, ts in rows:
            if lo <= ts <= hi:
                per_stream[label][ch].add(sq)

    def holes(seqs: set[int]) -> tuple[int, int]:
        if not seqs:
            return 0, 0
        return (max(seqs) - min(seqs) + 1) - len(seqs), len(seqs)

    print(f"\n{'channel':>8} {'venue':>6} {'ord n':>9} {'ord gaps':>10} "
          f"{'txn n':>9} {'txn gaps':>10} {'merged n':>9} {'merged gaps':>12}")
    tot_o = tot_t = tot_m = 0
    for ch in sorted(set(per_stream["order"]) | set(per_stream["transaction"]))[:14]:
        o, t = per_stream["order"][ch], per_stream["transaction"][ch]
        go, no = holes(o)
        gt, nt = holes(t)
        gm, nm = holes(o | t)
        tot_o += go
        tot_t += gt
        tot_m += gm
        print(f"{ch:>8} {venue_of_channel.get(ch, '?'):>6} {no:>9,} {go:>10,} "
              f"{nt:>9,} {gt:>10,} {nm:>9,} {gm:>12,}")

    print(f"\n  totals (shown channels): order gaps {tot_o:,}   "
          f"transaction gaps {tot_t:,}   MERGED gaps {tot_m:,}")
    if tot_m < min(tot_o, tot_t) * 0.2:
        print("  => merged stream is far more contiguous: SHARED sequence space CONFIRMED.")
        print("     Per-stream gap counts are an artefact, not packet loss.")
    else:
        print("  => merging did not close the gaps; investigate as real loss or slicing.")


if __name__ == "__main__":
    main()
