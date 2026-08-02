"""quarry -- data access.

DuckDB views over a Parquet partition tree, plus the prism ``factor_frame``
loader and its schema-fingerprint gate.

Two rules this package enforces rather than suggests:

* **Nothing is loaded that was not asked for.** Loaders push date and symbol
  predicates into SQL so DuckDB prunes partitions before scanning, and
  :func:`load_ticks` refuses an unbounded full-day read unless you say so.
* **A mismatched producer is refused.** :class:`SchemaFingerprint` covers factor
  names, parameters and order; a dump that disagrees raises rather than loading.
"""

from __future__ import annotations

from quarry.db import DATASETS, DataRoot, hive_date, open_db, register_view, stream
from quarry.loaders import (
    dump_fingerprint,
    iter_ticks,
    load_bars,
    load_daily,
    load_factor_frame,
    load_index,
    load_ticks,
    view_columns,
)
from quarry.raw import (
    PIT_TIMESTAMP,
    SSE_SCALING,
    SZSE_SCALING,
    Exchange,
    PhaseFlags,
    RecordType,
    Scaling,
    Side,
    Timestamps,
    TradingPhase,
    decode_sse_time,
    decode_szse_time,
    find_sequence_gaps,
    from_vendor_float_price,
    merged_sequence_gaps,
    normalize_sse_order,
    normalize_szse_trade,
    parse_sse_phase,
    parse_szse_phase,
    replay_key,
    scaling_for,
    to_amount,
    to_price,
    to_quantity,
)
from quarry.schema import FactorSpec, SchemaFingerprint, validate_fingerprint

__all__ = [
    "DATASETS",
    "PIT_TIMESTAMP",
    "SSE_SCALING",
    "SZSE_SCALING",
    "DataRoot",
    "Exchange",
    "FactorSpec",
    "PhaseFlags",
    "RecordType",
    "Scaling",
    "SchemaFingerprint",
    "Side",
    "Timestamps",
    "TradingPhase",
    "decode_sse_time",
    "decode_szse_time",
    "find_sequence_gaps",
    "from_vendor_float_price",
    "merged_sequence_gaps",
    "normalize_sse_order",
    "normalize_szse_trade",
    "parse_sse_phase",
    "parse_szse_phase",
    "replay_key",
    "scaling_for",
    "to_amount",
    "to_price",
    "to_quantity",
    "dump_fingerprint",
    "hive_date",
    "iter_ticks",
    "load_bars",
    "load_daily",
    "load_factor_frame",
    "load_index",
    "load_ticks",
    "open_db",
    "register_view",
    "stream",
    "validate_fingerprint",
    "view_columns",
]
