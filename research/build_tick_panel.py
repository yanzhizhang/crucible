"""Decode feitu tick dumps into a 3-second factor panel. Runs on the dev box.

Self-contained: needs only ``pycapnp``, ``zstandard`` and ``polars``, so it
does not require the crucible stack installed next to the data. Output is a
single parquet panel small enough to copy back and run the real pipeline
against.

Design
------
One minute of ticks is ~2.5M records, and a day is ~600M. Decoding that in one
process would take hours, so each minute is decoded and **aggregated to slots
independently** -- embarrassingly parallel, and each worker's peak memory is one
minute, not one day.

Prices are converted from the vendor's ``Float64`` back to the exchange's tick
grid by rounding, per venue scale. Truncating instead loses one tick on ~1% of
real prices (measured), and those errors do not cancel: they are all in the
same direction.

Usage::

    python build_tick_panel.py \\
        --order-dir /media/zzz/Stocks/v2_order \\
        --trade-dir /media/zzz/Stocks/v2_transaction \\
        --out ~/crucible_probe/panel.parquet \\
        --from 1000 --to 1100 --workers 8
"""

from __future__ import annotations

import argparse
import lzma
import re
import sys
from collections.abc import Iterator
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
# SSE price is N13(5), SZSE N13(4) -- never a shared constant.
PRICE_SCALE = {"XSHG": 10**5, "XSHE": 10**4}
_SCHEMA_DIR: Path | None = None


def _key(p: Path) -> str | None:
    m = _STAMP.search(p.name)
    return "_".join(m.groups()) if m else None


def _hhmm(k: str) -> int:
    return int(k[-5:-3] + k[-2:])


def _load(stem: str, struct: str):
    import capnp

    assert _SCHEMA_DIR is not None
    return getattr(capnp.load(str(_SCHEMA_DIR / f"{stem}.capnp")), struct)


def _blob(p: Path) -> bytes:
    if p.suffix == ".xz":
        with lzma.open(p, "rb") as fh:
            return fh.read()
    import zstandard

    with p.open("rb") as fh:
        return zstandard.ZstdDecompressor().stream_reader(fh).read()


def _iter(path: Path, stem: str, struct: str) -> Iterator:
    import capnp  # noqa: F401

    env = _load("export", "ExportedData")
    inner = _load(stem, struct)
    for e in env.read_multiple_bytes_packed(_blob(path)):
        yield inner.from_bytes_packed(e.data)


def _tick_price(px: float, venue: str) -> float:
    """Vendor float -> exact tick price. Round, never truncate."""
    s = PRICE_SCALE.get(venue)
    return round(px * s) / s if s else px


def decode_orders(path: Path) -> pl.DataFrame:
    sym, ven, ts, px, vol, side, rtype, otype = [], [], [], [], [], [], [], []
    for r in _iter(path, "order_v2", "OrderV2"):
        v = MARKET.get(int(r.marketId or 0))
        if v is None:
            continue
        ut = int(r.updateType or 0)
        # Anything not add/cancel is a product-status record. The v2 schema
        # comment documents only 0/1/2, but real dumps carry 3..10 as well.
        rt = "order_add" if ut == 1 else "order_cancel" if ut == 2 else "status"
        if rt == "status":
            continue
        d = int(r.dir or 0)
        sym.append(f"{int(r.symbolId or 0):06d}")
        ven.append(v)
        ts.append(int(r.spiderTs or 0) * 1_000_000)  # v2 is ms -> ns
        px.append(_tick_price(float(r.price or 0.0), v))
        vol.append(int(r.volume or 0))
        side.append("buy" if d == 1063 else "sell" if d == 1550 else None)
        rtype.append(rt)
        otype.append(int(r.orderType or 0))
    return pl.DataFrame(
        {
            "symbol": sym, "exchange": ven, "arrival_ts": ts, "price": px,
            "volume": vol, "side": side, "record_type": rtype, "order_type": otype,
        },
        schema_overrides={"volume": pl.Int64, "arrival_ts": pl.Int64, "order_type": pl.Int32},
    )


def decode_trades(path: Path) -> pl.DataFrame:
    sym, ven, ts, px, vol, side, rtype = [], [], [], [], [], [], []
    for r in _iter(path, "transaction_v2", "TransactionV2"):
        v = MARKET.get(int(r.marketId or 0))
        if v is None:
            continue
        tt = int(r.tradeType or 0)
        # SZSE cancels ride the TRADE stream. Counting them as executions
        # inflates volume; dropping them loses the cancel entirely.
        rt = "trade" if tt == 1 else "order_cancel" if tt == 2 else "status"
        if rt == "status":
            continue
        d = int(r.dir or 0)
        sym.append(f"{int(r.symbolId or 0):06d}")
        ven.append(v)
        ts.append(int(r.spiderTs or 0) * 1_000_000)
        px.append(_tick_price(float(r.price or 0.0), v))
        vol.append(int(r.volume or 0))
        side.append("buy" if d == 1063 else "sell" if d == 1550 else None)
        rtype.append(rt)
    return pl.DataFrame(
        {
            "symbol": sym, "exchange": ven, "arrival_ts": ts, "price": px,
            "volume": vol, "side": side, "record_type": rtype,
        },
        schema_overrides={"volume": pl.Int64, "arrival_ts": pl.Int64},
    )


def _one_minute(args: tuple[str, str, str, int]) -> bytes | None:
    """Decode one minute and return its slot aggregate as parquet bytes."""
    order_p, trade_p, schema_dir, slot_ns = args
    global _SCHEMA_DIR
    _SCHEMA_DIR = Path(schema_dir)
    sys.path.insert(0, str(_SCHEMA_DIR))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from micro_factors import aggregate_to_slots

    try:
        o = decode_orders(Path(order_p))
        t = decode_trades(Path(trade_p))
        if o.height == 0 and t.height == 0:
            return None
        # SZSE cancels arrive on the trade stream; move them to the order side
        # so cancel factors see both venues rather than only Shanghai.
        cx = t.filter(pl.col("record_type") == "order_cancel").with_columns(
            order_type=pl.lit(0, dtype=pl.Int32)
        )
        o = pl.concat([o, cx.select(o.columns)], how="vertical") if cx.height else o
        agg = aggregate_to_slots(o, t, slot_ns=slot_ns)
        import io

        buf = io.BytesIO()
        agg.write_parquet(buf)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED {Path(order_p).name}: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--order-dir", type=Path, required=True)
    ap.add_argument("--trade-dir", type=Path, required=True)
    ap.add_argument("--schema-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="t_from", type=int, default=0, help="HHMM inclusive")
    ap.add_argument("--to", dest="t_to", type=int, default=2359, help="HHMM inclusive")
    ap.add_argument("--slot-seconds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    o_by = {k: p for p in a.order_dir.iterdir() if (k := _key(p))}
    t_by = {k: p for p in a.trade_dir.iterdir() if (k := _key(p))}
    keys = sorted(set(o_by) & set(t_by))
    keys = [k for k in keys if a.t_from <= _hhmm(k) <= a.t_to]
    print(f"minutes to process: {len(keys)}  ({keys[0]} .. {keys[-1]})" if keys else "no minutes matched")
    if not keys:
        return

    slot_ns = a.slot_seconds * 1_000_000_000
    jobs = [(str(o_by[k]), str(t_by[k]), str(a.schema_dir.resolve()), slot_ns) for k in keys]

    parts: list[pl.DataFrame] = []
    done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for blob in ex.map(_one_minute, jobs):
            done += 1
            if blob:
                import io

                parts.append(pl.read_parquet(io.BytesIO(blob)))
            if done % 10 == 0:
                rows = sum(p.height for p in parts)
                print(f"  {done}/{len(keys)} minutes  ->  {rows:,} slot rows", flush=True)

    if not parts:
        print("nothing decoded")
        return

    panel = pl.concat(parts, how="vertical").sort(["slot", "symbol"], maintain_order=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(a.out, compression="zstd")
    print(f"\npanel: {panel.height:,} rows x {panel.width} cols")
    print(f"symbols: {panel['symbol'].n_unique():,}   slots: {panel['slot'].n_unique():,}")
    print(f"written: {a.out}  ({a.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
