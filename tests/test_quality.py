"""Each quality check must pass clean data and catch the fault it exists for."""

from pathlib import Path

import polars as pl
import pytest

from quarry.quality import (
    Level,
    check_book_sanity,
    check_daily_totals,
    check_latency,
    check_minute_coverage,
    check_sequence,
    check_trade_prices,
    check_volume_identity,
    grade,
    load_thresholds,
    session_minutes,
    skipped,
    verdict,
)

TH = load_thresholds(Path(__file__).resolve().parents[1] / "cfg" / "md_quality.toml")
T0 = 1_781_488_800_000_000_000  # 2026-06-15 10:00:00 +08:00 in ns UTC
MS = 1_000_000


def _levels(results, check):
    return {r.scope: r.level for r in results if r.check == check}


def test_grade_and_verdict():
    assert grade(0, 1, 2) is Level.PASS
    assert grade(1, 1, 2) is Level.WARN
    assert grade(2, 1, 2) is Level.FAIL
    assert grade(5, None, None) is Level.PASS
    rs = [skipped("x", "y", "z")]
    assert verdict(rs) is Level.PASS
    rs.append(check_minute_coverage(pl.DataFrame({"minute": ["0930"], "rows": [1]}), "k", TH)[0])
    assert verdict(rs) is Level.FAIL


def test_session_minutes():
    m = session_minutes([("09:30", "11:30"), ("13:00", "14:57")])
    assert len(m) == 120 + 117
    assert m[0] == "0930"
    assert m[-1] == "1456"


@pytest.mark.parametrize(("drop", "level"), [((), Level.PASS), (("1024",), Level.FAIL)])
def test_minute_coverage(drop, level):
    minutes = [
        m for m in session_minutes([("09:30", "11:30"), ("13:00", "14:57")]) if m not in drop
    ]
    man = pl.DataFrame({"minute": minutes, "rows": [10] * len(minutes)})
    (r,) = check_minute_coverage(man, "quotation", TH)
    assert r.level is level
    assert r.value == len(drop)


def test_minute_coverage_counts_empty_files_as_missing():
    minutes = session_minutes([("09:30", "11:30"), ("13:00", "14:57")])
    man = pl.DataFrame({"minute": minutes, "rows": [0] + [10] * (len(minutes) - 1)})
    (r,) = check_minute_coverage(man, "order", TH)
    assert r.level is Level.FAIL
    assert "0930" in r.detail


def _ticks(lat_ms, n=2000, exch="XSHE"):
    ex = [T0 + i * 10 * MS for i in range(n)]
    return pl.LazyFrame(
        {
            "exchange": [exch] * n,
            "exchange_ts": ex,
            "arrival_ts": [e + int(lat_ms * MS) for e in ex],
        }
    )


def test_latency_clean_is_pass():
    rs = check_latency(_ticks(20), "order", TH)
    lv = {r.check: r.level for r in rs}
    assert lv["timestamps.negative_latency_frac"] is Level.PASS
    assert lv["timestamps.latency_p99_ms"] is Level.PASS
    assert lv["timestamps.missing"] is Level.PASS
    p99 = next(r for r in rs if r.check == "timestamps.latency_p99_ms").value
    assert p99 == pytest.approx(20, rel=0.01)


def test_latency_clock_behind_exchange_fails():
    rs = check_latency(_ticks(-50), "order", TH)
    assert _levels(rs, "timestamps.negative_latency_frac")["order/XSHE"] is Level.FAIL


def test_latency_small_negative_within_tolerance_passes():
    rs = check_latency(_ticks(-2), "quotation", TH)
    assert _levels(rs, "timestamps.negative_latency_frac")["quotation/XSHE"] is Level.PASS


def test_latency_open_burst_is_backlog_not_steady_state():
    steady = _ticks(20)
    burst_t = T0 - 29 * 60 * 1_000 * MS  # 09:31 local
    burst = pl.LazyFrame(
        {"exchange": ["XSHE"], "exchange_ts": [burst_t], "arrival_ts": [burst_t + 60_000 * MS]}
    )
    rs = check_latency(pl.concat([steady, burst]), "transaction", TH)
    lv = {r.check: r for r in rs}
    assert lv["timestamps.latency_p99_ms"].level is Level.PASS
    assert lv["timestamps.burst_backlog_s"].level is Level.WARN
    assert lv["timestamps.burst_backlog_s"].value == pytest.approx(60.0)
    assert "09:30" in lv["timestamps.burst_backlog_s"].detail


def test_latency_ignores_status_records():
    ok = _ticks(20).with_columns(pl.lit("trade").alias("record_type"))
    status = pl.LazyFrame(
        {
            "exchange": ["XSHE"],
            "exchange_ts": [T0 - 4 * 3600 * 1_000 * MS],
            "arrival_ts": [T0],
            "record_type": ["status"],
        }
    )
    rs = check_latency(pl.concat([ok, status]), "order", TH)
    assert {r.check: r for r in rs}["timestamps.burst_backlog_s"].level is Level.PASS


def test_latency_zero_timestamps_fail():
    lf = _ticks(10).with_columns(pl.lit(0).alias("exchange_ts"))
    assert _levels(check_latency(lf, "order", TH), "timestamps.missing")["order/XSHE"] is Level.FAIL


def _seq_frames(order_ids, trade_ids, exch="XSHE", ch=2011):
    def mk(ids):
        return pl.LazyFrame(
            {
                "exchange": [exch] * len(ids),
                "channel_id": [ch] * len(ids),
                "seq_id": ids,
                "index_id": ids,
            },
            schema_overrides={"seq_id": pl.Int64, "index_id": pl.Int64},
        )

    return mk(order_ids), mk(trade_ids)


def test_sequence_merged_stream_contiguous():
    o, t = _seq_frames(list(range(1, 100, 2)), list(range(2, 101, 2)))
    rs = check_sequence(o, t, TH)
    assert _levels(rs, "sequence.missing_frac")["XSHE/seq_id"] is Level.PASS
    assert _levels(rs, "sequence.duplicates")["XSHE/seq_id"] is Level.PASS


def test_sequence_gap_fails():
    o, t = _seq_frames(list(range(1, 50)), list(range(60, 100)))
    rs = check_sequence(o, t, TH)
    r = next(x for x in rs if x.check == "sequence.missing_frac" and x.scope == "XSHE/seq_id")
    assert r.level is Level.FAIL
    assert "10 ids" in r.detail


def test_sequence_duplicates_flagged():
    o, t = _seq_frames([1, 2, 3], [3, 4])
    rs = check_sequence(o, t, TH)
    assert _levels(rs, "sequence.duplicates")["XSHE/seq_id"] is Level.WARN


def _quotes(total_volume, total_amount, b1=10.0, a1=10.01):
    return pl.LazyFrame(
        {
            "exchange": ["XSHG"],
            "symbol": ["600000"],
            "is_a_share": [True],
            "status": [3],
            "total_volume": [total_volume],
            "total_amount": [total_amount],
            "high_limited": [11.0],
            "low_limited": [9.0],
            "bid_px": [[b1] + [0.0] * 9],
            "ask_px": [[a1] + [0.0] * 9],
        },
        schema_overrides={"bid_px": pl.Array(pl.Float64, 10), "ask_px": pl.Array(pl.Float64, 10)},
    )


def _trades(prices, vols):
    n = len(prices)
    return pl.LazyFrame(
        {
            "exchange": ["XSHG"] * n,
            "symbol": ["600000"] * n,
            "record_type": ["trade"] * n,
            "price": prices,
            "volume": vols,
        }
    )


def test_volume_identity():
    t = _trades([10.0, 10.01], [100, 200])
    ok = check_volume_identity(_quotes(300, 10.0 * 100 + 10.01 * 200), t, TH)
    assert {r.level for r in ok} == {Level.PASS}
    bad = check_volume_identity(_quotes(400, 3002.0), t, TH)
    assert _levels(bad, "consistency.volume_mismatch_frac")["stock/XSHG"] is Level.FAIL


def test_crossed_book():
    assert check_book_sanity(_quotes(1, 1.0), TH)[0].level is Level.PASS
    assert check_book_sanity(_quotes(1, 1.0, b1=10.02, a1=10.01), TH)[0].level is Level.FAIL


def test_trade_prices():
    q = _quotes(1, 1.0)
    ok = check_trade_prices(q, _trades([10.0, 10.5], [100, 100]), TH)
    assert {r.level for r in ok} == {Level.PASS}
    bad = check_trade_prices(q, _trades([10.005, 11.5], [100, 100]), TH)
    lv = {r.check: r.level for r in bad}
    assert lv["consistency.off_tick_prices"] is Level.FAIL
    assert lv["consistency.price_outside_limits"] is Level.FAIL


def _ref(vol, amt, symbol="600000"):
    return pl.DataFrame(
        {
            "exchange": ["XSHG"],
            "symbol": [symbol],
            "is_a_share": [True],
            "volume": [vol],
            "amount": [amt],
        }
    )


def test_daily_totals_vs_reference():
    t = _trades([10.0, 10.01], [100, 200]).with_columns(pl.lit(True).alias("is_a_share"))
    amt = 10.0 * 100 + 10.01 * 200
    ok = check_daily_totals(t, _ref(300, amt), "tdx_eod", TH)
    assert {r.level for r in ok} == {Level.PASS}
    bad = check_daily_totals(t, _ref(400, amt), "tdx_eod", TH)
    assert _levels(bad, "multisource.daily_volume_mismatch_frac")["tdx_eod/XSHG"] is Level.FAIL


def test_daily_totals_flags_stock_missing_from_capture():
    t = _trades([10.0], [100]).with_columns(pl.lit(True).alias("is_a_share"))
    ref = pl.concat([_ref(100, 1000.0), _ref(500, 5000.0, symbol="600001")])
    rs = check_daily_totals(t, ref, "tdx_eod", TH)
    cov = next(r for r in rs if r.check == "multisource.coverage_missing_frac")
    assert cov.level is Level.FAIL
    assert "1 traded per tdx_eod" in cov.detail
