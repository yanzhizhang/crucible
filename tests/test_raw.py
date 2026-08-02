"""The raw exchange standard: scaling, timestamps, phase codes, sequencing.

These are pure-stdlib and depend on nothing else in crucible, so they are the
cheapest place to catch a data-contract regression.

Every expected value here traces to a specification clause, not to observed
behaviour of the code -- otherwise the test only proves the code is consistent
with itself.
"""

from __future__ import annotations

import datetime as dt

import pytest

from quarry.raw import (
    Exchange,
    RecordType,
    Side,
    Timestamps,
    TradingPhase,
    decode_sse_time,
    decode_szse_time,
    find_sequence_gaps,
    merged_sequence_gaps,
    from_vendor_float_price,
    normalize_sse_order,
    normalize_szse_trade,
    parse_sse_phase,
    replay_key,
    scaling_for,
    to_amount,
    to_price,
    to_quantity,
)

# --------------------------------------------------------------------------
# scaling -- the 10x trap
# --------------------------------------------------------------------------


def test_venue_price_scales_differ() -> None:
    """SSE is N13(5), SZSE is N13(4). Same price, different wire integers."""
    assert to_price(1_864_000, Exchange.XSHG) == pytest.approx(18.64)
    assert to_price(186_400, Exchange.XSHE) == pytest.approx(18.64)


def test_crossing_venue_scales_is_a_tenfold_error() -> None:
    """The failure this module exists to prevent: plausible but 10x wrong."""
    assert to_price(186_400, Exchange.XSHG) == pytest.approx(1.864)


def test_quantity_and_amount_scales() -> None:
    assert to_quantity(1_000, Exchange.XSHG) == pytest.approx(1.0)  # N15(3)
    assert to_quantity(100, Exchange.XSHE) == pytest.approx(1.0)  # N15(2)
    assert to_amount(100, Exchange.XSHG) == pytest.approx(1.0)  # N16(2)
    assert to_amount(10_000, Exchange.XSHE) == pytest.approx(1.0)  # N18(4)


def test_scaling_lookup_is_per_exchange() -> None:
    assert scaling_for(Exchange.XSHG).price_decimals == 5
    assert scaling_for(Exchange.XSHE).price_decimals == 4


# --------------------------------------------------------------------------
# vendor float round-trip
# --------------------------------------------------------------------------


def test_float_price_must_round_not_truncate() -> None:
    """1.13 is not representable in binary float; truncation loses a tick."""
    assert from_vendor_float_price(1.13, Exchange.XSHE) == 11_300
    assert int(1.13 * 10**4) == 11_299  # the bug being avoided


def test_truncation_error_is_common_not_exotic() -> None:
    """Measured, not asserted from intuition: ~5% of realistic CNY prices."""
    affected = sum(
        1 for c in range(100, 200_000) if int(c / 100.0 * 10**4) != round(c / 100.0 * 10**4)
    )
    assert affected > 10_000
    assert 0.03 < affected / 199_900 < 0.10


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------


def test_sse_time_needs_the_trade_date() -> None:
    """SSE transmits HHMMSSsss only; the date rides on a separate field."""
    assert decode_sse_time(20260731, 93015123) == dt.datetime(2026, 7, 31, 9, 30, 15, 123_000)


def test_szse_time_is_self_contained() -> None:
    assert decode_szse_time(20260731093015123) == dt.datetime(2026, 7, 31, 9, 30, 15, 123_000)


def test_both_venues_decode_to_the_same_instant() -> None:
    assert decode_sse_time(20260731, 93015123) == decode_szse_time(20260731093015123)


def test_midnight_and_boundary_times() -> None:
    assert decode_sse_time(20260731, 92500000) == dt.datetime(2026, 7, 31, 9, 25, 0)
    assert decode_sse_time(20260731, 150000000) == dt.datetime(2026, 7, 31, 15, 0, 0)


# --------------------------------------------------------------------------
# point-in-time gating
# --------------------------------------------------------------------------


def test_pit_gates_on_arrival_not_exchange_time() -> None:
    """Keying on exchange time assumes zero wire latency -- a look-ahead."""
    t = Timestamps(exchange_ts=1_000, arrival_ts=1_350, broker_ts=1_100)
    assert t.pit() == 1_350
    assert t.wire_latency_ns == 350


def test_pit_falls_back_to_exchange_time_when_uncaptured() -> None:
    """Best available, but optimistic -- it assumes instant delivery."""
    assert Timestamps(exchange_ts=1_000).pit() == 1_000
    assert Timestamps(exchange_ts=1_000).wire_latency_ns is None


def test_clock_skew_is_detected() -> None:
    """Causal order is exchange <= broker <= arrival.

    The broker server is upstream of our capture box, so arrival is last. An
    earlier version of this test asserted the reverse; 600k real records
    (broker <= arrival on 100.0% of them) settled it.
    """
    # Arrival before the exchange generated it -- impossible either way.
    assert not Timestamps(exchange_ts=1_000, arrival_ts=900).is_causal

    # Broker between exchange and arrival: the normal case.
    assert Timestamps(exchange_ts=1_000, arrival_ts=1_100, broker_ts=1_050).is_causal

    # Broker *after* our own capture: impossible.
    assert not Timestamps(exchange_ts=1_000, arrival_ts=1_100, broker_ts=1_200).is_causal


# --------------------------------------------------------------------------
# trading phase
# --------------------------------------------------------------------------


def test_continuous_phase_is_tradable() -> None:
    f = parse_sse_phase("T110")
    assert f.phase is TradingPhase.CONTINUOUS
    assert f.can_trade and f.is_listed and f.is_tradable
    assert not f.accepts_orders  # 4th char '0'


def test_suspension_is_authoritative_not_inferred() -> None:
    """'P' says suspended outright -- better than guessing from volume == 0."""
    f = parse_sse_phase("P010")
    assert f.phase is TradingPhase.SUSPENDED
    assert f.is_halted and not f.is_tradable


def test_terminal_circuit_break_is_halted() -> None:
    assert parse_sse_phase("N010").is_halted


def test_auction_phases_are_flagged() -> None:
    """Fills in a call auction are single-price, not continuous."""
    assert parse_sse_phase("C11 ").is_auction
    assert parse_sse_phase("U11 ").is_auction
    assert not parse_sse_phase("T110").is_auction


def test_unlisted_is_not_tradable() -> None:
    assert not parse_sse_phase("T101").is_listed


def test_malformed_phase_code_degrades_safely() -> None:
    """A bad code must not raise mid-replay, and must not read as tradable."""
    f = parse_sse_phase("")
    assert f.phase is TradingPhase.UNKNOWN
    assert not f.is_tradable


# --------------------------------------------------------------------------
# venue divergence: where cancels live
# --------------------------------------------------------------------------


def test_sse_cancels_ride_the_order_stream() -> None:
    assert normalize_sse_order("0") is RecordType.ORDER_ADD
    assert normalize_sse_order("4") is RecordType.ORDER_CANCEL


def test_szse_cancels_ride_the_trade_stream() -> None:
    """Treating every SZSE trade record as an execution inflates volume."""
    assert normalize_szse_trade("F") is RecordType.TRADE
    assert normalize_szse_trade("4") is RecordType.ORDER_CANCEL


def test_both_venues_normalise_to_one_vocabulary() -> None:
    assert normalize_sse_order("4") is normalize_szse_trade("4")


def test_lending_sides_are_not_directional_flow() -> None:
    """SZSE 'G'/'F' are securities lending, not buying or selling pressure."""
    assert Side.BUY.is_directional and Side.SELL.is_directional
    assert not Side.BORROW.is_directional
    assert not Side.LEND.is_directional


# --------------------------------------------------------------------------
# sequencing
# --------------------------------------------------------------------------


def test_replay_key_is_channel_then_sequence() -> None:
    """Not timestamp: TransactTime is millisecond and ties constantly."""
    assert replay_key(3, 100) < replay_key(3, 101)
    assert replay_key(2, 999) < replay_key(3, 1)


def test_sequence_gaps_are_reported() -> None:
    assert find_sequence_gaps([1, 2, 3, 7, 8, 10]) == [(4, 6), (9, 9)]


def test_gap_before_first_record_is_reported() -> None:
    """ApplSeqNum starts at 1; starting at 3 means two records were lost."""
    assert find_sequence_gaps([3, 4]) == [(1, 2)]


def test_contiguous_stream_has_no_gaps() -> None:
    assert find_sequence_gaps([1, 2, 3]) == []
    assert find_sequence_gaps([]) == []


# --------------------------------------------------------------------------
# corrections forced by real colo data (feitu v2, 2026-04-21 10:30)
# --------------------------------------------------------------------------


def test_order_and_trade_share_one_sequence_space() -> None:
    """Each stream alone looks ~50% lossy; merged it is contiguous.

    Measured over a full minute across 14 channels and both venues: per-stream
    gaps were 845,453 (order) and 1,418,420 (trade), merged gaps were 0. This
    encodes that finding in miniature -- odd sequence numbers on the order
    stream, even on the trade stream.
    """
    order_seqs = [1, 3, 5, 7, 9]
    trade_seqs = [2, 4, 6, 8, 10]

    # Checked alone, each stream accuses the other's records of being loss.
    assert find_sequence_gaps(order_seqs) == [(2, 2), (4, 4), (6, 6), (8, 8)]
    assert find_sequence_gaps(trade_seqs) == [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]

    # Merged, the channel is provably complete.
    assert merged_sequence_gaps(order_seqs, trade_seqs) == []


def test_merged_sequence_gaps_still_detects_real_loss() -> None:
    """Merging must not paper over genuine packet loss."""
    assert merged_sequence_gaps([1, 3], [2, 6]) == [(4, 5)]


def test_broker_timestamp_sits_between_exchange_and_arrival() -> None:
    """Measured chain: exchange <= broker <= arrival, on 100% of 600k records.

    The broker server is upstream of our capture box, so arrival is last. The
    original model had arrival before broker, which flagged every real record
    as clock skew.
    """
    ok = Timestamps(exchange_ts=1_000, broker_ts=1_000, arrival_ts=1_067_000_000)
    assert ok.is_causal

    # Broker after our own capture is physically impossible.
    assert not Timestamps(
        exchange_ts=1_000, broker_ts=2_000_000_000, arrival_ts=1_000_000_000
    ).is_causal

    # Broker before the exchange generated the record is equally impossible.
    assert not Timestamps(exchange_ts=5_000, broker_ts=1_000, arrival_ts=9_000).is_causal


def test_degenerate_broker_stamp_is_causal_not_suspicious() -> None:
    """``broker_ts == exchange_ts`` means "not measured", not "zero latency".

    The feitu archive copies the exchange stamp into ``serverTs``; that must
    not be read as a causality violation.
    """
    assert Timestamps(exchange_ts=7_000, broker_ts=7_000, arrival_ts=7_000_067).is_causal


def test_duplicate_sequence_numbers_do_not_create_gaps() -> None:
    assert find_sequence_gaps([1, 2, 2, 3]) == []


# --------------------------------------------------------------------------
# symbol routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("symbol", "venue"),
    [
        ("600519", Exchange.XSHG),
        ("688017", Exchange.XSHG),
        ("510300", Exchange.XSHG),
        ("000001", Exchange.XSHE),
        ("300750", Exchange.XSHE),
    ],
)
def test_symbol_venue_inference(symbol: str, venue: Exchange) -> None:
    assert Exchange.of_symbol(symbol) is venue


def test_wind_suffix() -> None:
    assert Exchange.XSHG.wind_suffix == ".SH"
    assert Exchange.XSHE.wind_suffix == ".SZ"
