"""Adapter: NonConvex/feitu capnp SHM dumps -> the crucible raw standard.

This is an **adapter onto** :mod:`quarry.raw`, not a definition of it. The
exchange specification is the authority; this vendor archive is one encoding of
it, and where the two disagree the spec wins and the divergence is recorded
here rather than absorbed silently.

Wire format (per the vendor's own reference implementation)::

    file.zst / file.xz
      -> decompress
      -> stream of PACKED capnp `ExportedData` envelopes {shmKey, nanoTs, data}
      -> each envelope's `data` is itself a PACKED capnp message

Both layers are *packed* encoding, not plain capnp framing. Reading either with
the unpacked reader yields garbage rather than an error.

Three vendor-specific hazards this module handles
-------------------------------------------------

**1. Timestamp units change with the schema generation.** v1 and v2 stamp
milliseconds; v3 stamps nanoseconds. Mixing them is a 10^6 error that lands
timestamps in 1970. Everything here is normalised to **nanoseconds** on the way
in, and the per-generation unit is declared in :data:`TIME_UNIT_NS` rather than
inferred.

**2. Field order differs between streams.** ``Index`` places ``serverTs`` at @11
and ``spiderTs`` at @12; ``OrderV2``/``TransactionV2`` place ``spiderTs`` at @13
and ``serverTs`` at @14 -- i.e. **reversed**. Positional decoding would swap
colo-capture time with broker time, quietly corrupting every latency measurement
and the point-in-time gate that depends on it. Access is by name, always.

**3. Prices are ``Float64``.** The exchange transmits scaled integers; this
archive re-encodes them as binary floats, which cannot represent most 2-decimal
CNY prices exactly. Conversion back goes through
:func:`quarry.raw.from_vendor_float_price`, which rounds -- truncation loses a
tick on ~5% of realistic prices.

The vendor does **not** flatten the venue cancel divergence, which is correct:
SSE cancels arrive as ``OrderV2.updateType == 2`` and SZSE cancels as
``TransactionV2.tradeType == 2``, mirroring the exchanges. :func:`record_type_of`
normalises both into :class:`quarry.raw.RecordType`.
"""

from __future__ import annotations

import lzma
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import polars as pl

from quarry.raw import (
    Exchange,
    RecordType,
    Side,
    from_vendor_float_price,
)

__all__ = [
    "TYPES",
    "TIME_UNIT_NS",
    "detect_type",
    "open_dump",
    "iter_records",
    "records_to_frame",
    "symbol_of",
    "exchange_of",
    "record_type_of",
    "side_of",
]

#: type key -> (capnp module, struct name). Keys mirror the on-disk filename
#: prefix, matching the vendor's own reference reader.
TYPES: Final[dict[str, tuple[str, str]]] = {
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

TIME_UNIT_NS: Final[dict[str, int]] = {
    # v1/v2 stamp milliseconds; v3 stamps nanoseconds. Declared, not sniffed:
    # guessing from magnitude works until a clock is wrong and then fails
    # silently in the direction that looks plausible.
    "index": 1_000_000,
    "order": 1_000_000,
    "transaction": 1_000_000,
    "v2_quotation": 1_000_000,
    "v2_order": 1_000_000,
    "v2_transaction": 1_000_000,
    "v3_quotation": 1,
    "v3_index": 1,
    "v3_order": 1,
    "v3_transaction": 1,
}

_MARKET_SHSE: Final = 3553
_MARKET_SZSE: Final = 3554
_DIR_BUY: Final = 1063
_DIR_SELL: Final = 1550


def detect_type(path: str | Path) -> str:
    """Infer the type key from a filename.

    Longest key first, so ``v3_index`` wins over ``index`` and ``v2_order``
    over ``order``. Raises rather than returning ``None``: decoding a dump with
    the wrong schema produces plausible-looking nonsense, not an error, so a
    failed guess must stop the run.
    """
    name = Path(path).name.lower()
    for key in sorted(TYPES, key=len, reverse=True):
        if key in name:
            return key
    raise ValueError(
        f"cannot infer dump type from {Path(path).name!r}; pass the type explicitly. "
        f"Known: {sorted(TYPES)}"
    )


def open_dump(path: str | Path) -> bytes:
    """Decompress a ``.zst`` or ``.xz`` dump into memory.

    Returns raw bytes because the capnp reader needs the whole stream. These
    files are per-day-per-venue and typically hundreds of MB decompressed --
    check available memory before opening several at once.
    """
    p = Path(path)
    if p.suffix == ".xz":
        return lzma.open(p, "rb").read()
    if p.suffix == ".zst":
        try:
            import zstandard
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "reading .zst dumps needs `zstandard` (pip install zstandard)"
            ) from exc
        with p.open("rb") as fh:
            return zstandard.ZstdDecompressor().stream_reader(fh).read()
    raise ValueError(f"unsupported dump extension {p.suffix!r}; expected .zst or .xz")


def iter_records(
    path: str | Path,
    type_key: str | None = None,
    *,
    schema_dir: str | Path | None = None,
    limit: int | None = None,
) -> Iterator[Any]:
    """Yield decoded capnp records from a dump.

    Parameters
    ----------
    schema_dir:
        Directory holding the ``*.capnp`` schema files. Required because
        ``pycapnp`` resolves schemas from the import path, not from the dump.
    limit:
        Stop after this many records. Use it when probing an unfamiliar dump --
        a full day of tick data is tens of millions of records.

    Notes
    -----
    Both the envelope stream and each payload are **packed** capnp. The vendor's
    reference reader uses ``read_multiple_bytes_packed`` and
    ``from_bytes_packed`` respectively, and this mirrors it exactly.
    """
    import sys

    try:
        import capnp  # noqa: F401  # registers the *_capnp import hook
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "decoding feitu dumps needs `pycapnp`, which links the system libcapnp:\n"
            "  apt-get install -y libcapnp-dev && pip install pycapnp zstandard"
        ) from exc

    key = type_key or detect_type(path)
    if key not in TYPES:
        raise ValueError(f"unknown dump type {key!r}; known: {sorted(TYPES)}")

    if schema_dir is not None:
        d = str(Path(schema_dir).resolve())
        if d not in sys.path:
            sys.path.insert(0, d)

    import importlib

    export_capnp = importlib.import_module("export_capnp")
    module_name, struct_name = TYPES[key]
    struct = getattr(importlib.import_module(module_name), struct_name)

    blob = open_dump(path)
    for i, exported in enumerate(export_capnp.ExportedData.read_multiple_bytes_packed(blob)):
        if limit is not None and i >= limit:
            return
        yield struct.from_bytes_packed(exported.data)


# --------------------------------------------------------------------------
# field normalisation
# --------------------------------------------------------------------------


def symbol_of(symbol_id: int) -> str:
    """Vendor ``Int32`` code -> canonical 6-digit string.

    ``000001`` travels as the integer ``1``. Formatting it back without
    zero-padding yields ``"1"``, which then matches nothing in any join against
    the rest of the stack -- and matches *silently*, producing an empty frame
    rather than an error.
    """
    return f"{int(symbol_id):06d}"


def exchange_of(market_id: int) -> Exchange | None:
    """Vendor ``marketId`` -> :class:`quarry.raw.Exchange`.

    Returns ``None`` for anything else (overseas venues appear in the index
    stream) so the caller can filter rather than mis-attribute.
    """
    if market_id == _MARKET_SHSE:
        return Exchange.XSHG
    if market_id == _MARKET_SZSE:
        return Exchange.XSHE
    return None


def side_of(dir_code: int) -> Side | None:
    """Vendor ``dir`` -> :class:`quarry.raw.Side`. ``0`` means unknown."""
    if dir_code == _DIR_BUY:
        return Side.BUY
    if dir_code == _DIR_SELL:
        return Side.SELL
    return None


def record_type_of(kind: str, record: Any) -> RecordType:
    """Normalise a vendor record into the venue-neutral vocabulary.

    The vendor mirrors the exchanges' split rather than flattening it, so
    cancels must be read from different fields depending on the stream:

    * order stream    -- ``updateType == 2`` is a cancel (SSE)
    * trade stream    -- ``tradeType  == 2`` is a cancel (SZSE)

    Reading only one of them drops half the cancels in a mixed-venue dump.
    """
    if "order" in kind:
        ut = int(getattr(record, "updateType", 0) or 0)
        if ut == 2:
            return RecordType.ORDER_CANCEL
        if ut == 1:
            return RecordType.ORDER_ADD
        return RecordType.STATUS
    if "transaction" in kind:
        tt = int(getattr(record, "tradeType", 0) or 0)
        if tt == 2:
            return RecordType.ORDER_CANCEL
        if tt == 1:
            return RecordType.TRADE
        return RecordType.STATUS
    return RecordType.STATUS


def records_to_frame(
    records: Iterator[Any],
    kind: str,
    *,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Decode records into a long frame in the crucible raw standard.

    Returns columns ``exchange_ts``, ``arrival_ts``, ``broker_ts`` (all
    **nanoseconds**, converted from the generation's native unit), ``symbol``,
    ``exchange``, ``channel_id``, ``seq_id`` and stream-specific fields.

    Prices become scaled integers via
    :func:`quarry.raw.from_vendor_float_price`; ``price_raw`` keeps the vendor
    float alongside so the conversion is auditable rather than assumed.

    Notes
    -----
    Rows whose ``marketId`` is neither SSE nor SZSE are kept with a null
    ``exchange`` rather than dropped -- the index stream legitimately carries
    overseas venues, and silently discarding rows is how a coverage gap becomes
    invisible.
    """
    unit = TIME_UNIT_NS.get(kind)
    if unit is None:
        raise ValueError(f"no timestamp unit declared for kind {kind!r}")

    is_index = "index" in kind
    rows: list[dict[str, Any]] = []

    for n, r in enumerate(records):
        if max_rows is not None and n >= max_rows:
            break
        venue = exchange_of(int(getattr(r, "marketId", 0) or 0))
        row: dict[str, Any] = {
            "exchange_ts": int(getattr(r, "time", 0) or 0) * unit,
            "arrival_ts": int(getattr(r, "spiderTs", 0) or 0) * unit,
            "broker_ts": int(getattr(r, "serverTs", 0) or 0) * unit,
            "symbol": symbol_of(getattr(r, "symbolId", 0) or 0),
            "exchange": venue.value if venue else None,
        }

        if is_index:
            row.update(
                last=float(getattr(r, "lastPrice", 0.0) or 0.0),
                open=float(getattr(r, "openPrice", 0.0) or 0.0),
                high=float(getattr(r, "highPrice", 0.0) or 0.0),
                low=float(getattr(r, "lowPrice", 0.0) or 0.0),
                prev_close=float(getattr(r, "preClosePrice", 0.0) or 0.0),
                # Index volume is in LOTS (手), not shares, unlike the equity
                # streams. Keeping the vendor's unit and naming it prevents a
                # 100x error the moment someone sums it against share volume.
                total_volume_lots=int(getattr(r, "totalVolume", 0) or 0),
                total_amount=float(getattr(r, "totalAmount", 0.0) or 0.0),
            )
        else:
            px = float(getattr(r, "price", 0.0) or 0.0)
            row.update(
                record_type=record_type_of(kind, r).value,
                price_raw=px,
                price=from_vendor_float_price(px, venue) if venue else None,
                volume=int(getattr(r, "volume", 0) or 0),
                side=(s.value if (s := side_of(int(getattr(r, "dir", 0) or 0))) else None),
                channel_id=int(getattr(r, "channelId", 0) or 0),
                seq_id=int(getattr(r, "seqId", 0) or 0),
            )
            if "order" in kind:
                row["order_id"] = int(getattr(r, "orderId", 0) or 0)
                row["order_type"] = int(getattr(r, "orderType", 0) or 0)
            else:
                row["buy_seq_id"] = int(getattr(r, "buySeqId", 0) or 0)
                row["sell_seq_id"] = int(getattr(r, "sellSeqId", 0) or 0)

        rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    # Canonical replay order is (channel, sequence), never timestamp -- see
    # quarry.raw.replay_key. The index stream has no sequence, so it falls back
    # to arrival time, which is the honest ordering for a snapshot feed.
    keys = [k for k in ("channel_id", "seq_id") if k in df.columns]
    return df.sort(keys or ["arrival_ts"], maintain_order=True)
