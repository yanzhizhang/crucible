"""Step 1 gate: streaming loaders, partition pruning, and producer refusal."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from crucible.errors import SchemaMismatch
from quarry.db import DataRoot, open_db, stream
from quarry.loaders import (
    dump_fingerprint,
    iter_ticks,
    load_bars,
    load_daily,
    load_factor_frame,
    load_index,
    load_ticks,
)
from quarry.schema import FactorSpec, SchemaFingerprint
from quarry.synth import make_store

# --------------------------------------------------------------------------
# store layout
# --------------------------------------------------------------------------


def test_views_registered_for_present_datasets(store, conn) -> None:
    views = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    assert {"daily", "factor_frame", "index", "bars_1min", "ticks"} <= views


def test_hive_keys_detected_per_dataset(store) -> None:
    """Daily partitions by date alone; bars add symbol. Both must open."""
    dr = DataRoot(store.root)
    assert dr.hive_keys("daily") == ("date",)
    assert dr.hive_keys("bars/1min") == ("date", "symbol")


def test_symbol_partition_stays_a_string(store, conn) -> None:
    """``symbol=000001`` must not be inferred as the integer 1.

    Type inference here is the classic silent join failure: the read succeeds,
    the join matches nothing, and every downstream frame is empty.
    """
    bars = load_bars(conn, "1min", dates=[store.dates[0]], symbols=[store.symbols[0]])
    assert bars["symbol"].dtype == pl.Utf8
    assert bars["symbol"][0] == store.symbols[0]
    assert bars.height > 0


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def test_loaders_return_long_format(store, conn) -> None:
    for frame in (
        load_daily(conn, dates=store.dates[:2]),
        load_factor_frame(conn, dates=store.dates[:2]),
        load_bars(conn, "1min", dates=store.dates[:1], symbols=list(store.symbols[:2])),
    ):
        assert frame.columns[:2] == ["ts", "symbol"]
        assert frame.height > 0


def test_frames_are_sorted_canonically(store, conn) -> None:
    """Determinism depends on stable row order across runs."""
    df = load_daily(conn, dates=store.dates[:3])
    assert df.equals(df.sort(["ts", "symbol"], maintain_order=True))


def test_date_predicate_restricts_result(store, conn) -> None:
    one = load_daily(conn, dates=[store.dates[0]])
    assert set(one["ts"].dt.date().to_list()) == {store.dates[0]}
    assert one.height == len(store.symbols)


def test_symbol_predicate_restricts_result(store, conn) -> None:
    picked = list(store.symbols[:3])
    df = load_daily(conn, symbols=picked)
    assert set(df["symbol"].to_list()) == set(picked)


def test_start_end_range(store, conn) -> None:
    df = load_daily(conn, start=store.dates[1], end=store.dates[3])
    got = sorted(set(df["ts"].dt.date().to_list()))
    assert got == list(store.dates[1:4])


def test_factor_subset_reads_only_named_columns(store, conn) -> None:
    df = load_factor_frame(conn, dates=store.dates[:2], factors=["alpha_strong"])
    assert df.columns == ["ts", "symbol", "alpha_strong"]


def test_unknown_factor_raises(store, conn) -> None:
    with pytest.raises(KeyError, match="not in factor_frame"):
        load_factor_frame(conn, dates=store.dates[:1], factors=["no_such_alpha"])


def test_empty_symbol_list_is_an_error(store, conn) -> None:
    """``[]`` almost always means an upstream filter emptied out."""
    with pytest.raises(ValueError, match="symbols was empty"):
        load_daily(conn, symbols=[])


def test_load_index(store, conn) -> None:
    idx = load_index(conn, "000300.SH")
    assert idx.height == len(store.dates)
    assert set(idx["symbol"].to_list()) == {"000300.SH"}


def test_daily_ts_promoted_from_partition(store, conn) -> None:
    df = load_daily(conn, dates=[store.dates[0]])
    assert df["ts"].dtype == pl.Datetime
    assert df["ts"].dt.date()[0] == store.dates[0]


# --------------------------------------------------------------------------
# schema fingerprint -- refusing a mismatched producer
# --------------------------------------------------------------------------


def test_fingerprint_roundtrip() -> None:
    fp = SchemaFingerprint.of(["a", "b"], producer="prism-1.2.3")
    assert SchemaFingerprint.from_json(fp.to_json()).digest() == fp.digest()


def test_fingerprint_is_process_stable() -> None:
    """Must not use salted ``hash()``; the digest is compared across runs."""
    a = SchemaFingerprint.of(["a", "b"], producer="p")
    b = SchemaFingerprint.of(["a", "b"], producer="p")
    assert a.digest() == b.digest()


def test_param_order_does_not_change_the_digest() -> None:
    a = SchemaFingerprint((FactorSpec("mom", {"win": 20, "lag": 1}),))
    b = SchemaFingerprint((FactorSpec("mom", {"lag": 1, "win": 20}),))
    assert a.digest() == b.digest()


def test_changed_lookback_is_a_different_factor() -> None:
    a = SchemaFingerprint((FactorSpec("mom", {"win": 20}),))
    b = SchemaFingerprint((FactorSpec("mom", {"win": 22}),))
    with pytest.raises(SchemaMismatch, match="win=22"):
        a.validate(b)


def test_column_reorder_is_refused() -> None:
    a = SchemaFingerprint.of(["x", "y"])
    b = SchemaFingerprint.of(["y", "x"])
    with pytest.raises(SchemaMismatch, match="column order differs"):
        a.validate(b)


def test_missing_and_extra_columns_are_named() -> None:
    a = SchemaFingerprint.of(["x", "y"])
    b = SchemaFingerprint.of(["x", "z"])
    with pytest.raises(SchemaMismatch, match="missing from dump.*y|unexpected in dump.*z"):
        a.validate(b)


def test_load_factor_frame_refuses_mismatched_producer(store, conn) -> None:
    """The producer/consumer gate, end to end."""
    wrong = SchemaFingerprint.of(["totally_different_factor"], producer="prism-9.9.9")
    with pytest.raises(SchemaMismatch, match="refusing factor_frame dump"):
        load_factor_frame(conn, dates=store.dates[:1], expected=wrong)


def test_load_factor_frame_accepts_matching_fingerprint(store, conn) -> None:
    actual = dump_fingerprint(conn, "factor_frame")
    df = load_factor_frame(conn, dates=store.dates[:1], expected=actual)
    assert df.height == len(store.symbols)


def test_fingerprint_none_skips_validation(store, conn) -> None:
    """The opt-out exists but must be written at the call site."""
    assert load_factor_frame(conn, dates=store.dates[:1], expected=None).height > 0


def test_tampered_fingerprint_payload_is_rejected() -> None:
    fp = SchemaFingerprint.of(["a"], producer="p")
    payload = fp.to_json().replace('"a"', '"b"')
    with pytest.raises(SchemaMismatch, match="self-inconsistent"):
        SchemaFingerprint.from_json(payload)


# --------------------------------------------------------------------------
# tick streaming -- never materialise a full day
# --------------------------------------------------------------------------


def test_full_day_tick_read_is_guarded(store, conn) -> None:
    with pytest.raises(ValueError, match="allow_full_day"):
        load_ticks(conn, store.dates[0])


def test_full_day_tick_read_allowed_when_explicit(store, conn) -> None:
    df = load_ticks(conn, store.dates[0], allow_full_day=True)
    assert df.height > 0


def test_load_ticks_with_symbols(store, conn) -> None:
    sym = store.symbols[0]
    df = load_ticks(conn, store.dates[0], symbols=[sym])
    assert set(df["symbol"].to_list()) == {sym}
    assert df.height == 4800  # 4 hours of 3-second snapshots


def test_iter_ticks_streams_in_bounded_batches(store, conn) -> None:
    """Peak memory is one batch, not one day."""
    chunks = list(iter_ticks(conn, store.dates[:2], symbols=[store.symbols[0]], batch_rows=1000))
    assert len(chunks) > 2
    assert all(c.height <= 1000 for _, c in chunks)
    assert sum(c.height for _, c in chunks) == 2 * 4800
    assert {d for d, _ in chunks} == set(store.dates[:2])


def test_stream_helper_yields_polars(store, conn) -> None:
    batches = list(stream(conn, "SELECT * FROM daily", batch_rows=10))
    assert batches and all(isinstance(b, pl.DataFrame) for b in batches)


def test_ticks_are_right_labelled(store, conn) -> None:
    df = load_ticks(conn, store.dates[0], symbols=[store.symbols[0]])
    first = df["ts"][0]
    assert first == dt.datetime.combine(store.dates[0], dt.time(9, 30, 3))


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_store_generation_is_deterministic(tmp_path) -> None:
    """Same seed, same bytes -- the invariant the whole stack rests on."""
    a = make_store(tmp_path / "a", n_symbols=8, end="2024-01-10")
    b = make_store(tmp_path / "b", n_symbols=8, end="2024-01-10")

    assert a.symbols == b.symbols
    assert a.suspensions == b.suspensions
    assert a.fingerprint.digest() == b.fingerprint.digest()

    ca, cb = open_db(a.root), open_db(b.root)
    try:
        assert load_daily(ca).equals(load_daily(cb))
        assert load_factor_frame(ca).equals(load_factor_frame(cb))
    finally:
        ca.close()
        cb.close()
