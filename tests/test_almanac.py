"""Step 1 gate: calendar correctness, PIT universe, and the stale-price trap."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from almanac.adjust import adjust, cumulative_factors, event_factors
from almanac.calendar import COMMODITY_NIGHT, EQUITY, TradingCalendar, parse_freq
from almanac.masks import apply_masks, build_masks, limit_pct
from almanac.universe import Membership, Universe
from crucible.errors import UniverseError
from quarry.loaders import load_daily

# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("3s", 3), ("30s", 30), ("1min", 60), ("5min", 300), ("1h", 3600), ("1d", 86400)],
)
def test_parse_freq(text: str, seconds: int) -> None:
    assert parse_freq(text) == dt.timedelta(seconds=seconds)


@pytest.mark.parametrize("bad", ["", "0s", "1M", "week", "-5min", "1.5min"])
def test_parse_freq_rejects_ambiguous_units(bad: str) -> None:
    with pytest.raises(ValueError, match="unsupported freq|must be positive|ambiguous freq"):
        parse_freq(bad)


def test_equity_slot_counts() -> None:
    """09:30-11:30 and 13:00-14:57 continuous, plus one close-auction slot."""
    cal = TradingCalendar()
    day = cal.sessions("2024-01-02", "2024-01-05")[0]

    minute = cal.slot_grid(day, "1min")
    assert len(minute) == 120 + 117 + 1

    three_sec = cal.slot_grid(day, "3s")
    assert len(three_sec) == 120 * 20 + 117 * 20 + 1


def test_slots_are_right_labelled() -> None:
    """The first label is open+freq, never the open itself.

    A slot labelled at the open would contain no elapsed time, and a feature
    reading it would be seeing the coming interval.
    """
    cal = TradingCalendar()
    day = cal.sessions("2024-01-02", "2024-01-05")[0]
    grid = cal.slot_grid(day, "1min")

    assert grid[0] == dt.datetime.combine(day, dt.time(9, 31))
    assert grid[-1] == dt.datetime.combine(day, dt.time(15, 0))
    # The afternoon session restarts at 13:01, not 13:00.
    assert dt.datetime.combine(day, dt.time(13, 1)) in grid.to_list()
    assert dt.datetime.combine(day, dt.time(13, 0)) not in grid.to_list()


def test_slot_grid_refuses_ragged_frequency() -> None:
    """7 minutes does not tile a 120-minute session; a short final bar biases it."""
    cal = TradingCalendar()
    day = cal.sessions("2024-01-02", "2024-01-05")[0]
    with pytest.raises(ValueError, match="does not divide"):
        cal.slot_grid(day, "7min")


def test_slot_grid_rejects_non_session() -> None:
    cal = TradingCalendar()
    with pytest.raises(ValueError, match="not a trading session"):
        cal.slot_grid("2024-01-06", "1min")  # a Saturday


def test_sessions_exclude_mainland_holidays() -> None:
    """Spring Festival 2024 (Feb 10-17) is closed."""
    cal = TradingCalendar()
    sessions = set(cal.sessions("2024-02-08", "2024-02-20"))
    assert dt.date(2024, 2, 12) not in sessions
    assert dt.date(2024, 2, 8) in sessions
    assert dt.date(2024, 2, 19) in sessions


def test_shift_moves_by_sessions_not_days() -> None:
    cal = TradingCalendar()
    friday = dt.date(2024, 1, 5)
    assert cal.shift(friday, 1) == dt.date(2024, 1, 8)
    assert cal.shift(dt.date(2024, 1, 8), -1) == friday


def test_night_session_belongs_to_next_trading_date() -> None:
    """A commodity night session trades on the previous calendar evening."""
    cal = TradingCalendar(session=COMMODITY_NIGHT)
    day = cal.sessions("2024-01-03", "2024-01-05")[0]
    grid = cal.slot_grid(day, "1min", include_night=True)
    assert grid[0].date() == day - dt.timedelta(days=1)
    assert grid[0].time() > dt.time(20, 0)
    assert grid[-1].date() == day


def test_auction_windows() -> None:
    cal = TradingCalendar()
    day = dt.date(2024, 1, 2)
    assert cal.in_open_auction(dt.datetime.combine(day, dt.time(9, 20)))
    assert not cal.in_continuous(dt.datetime.combine(day, dt.time(9, 20)))
    assert cal.in_close_auction(dt.datetime.combine(day, dt.time(14, 58)))
    assert cal.in_continuous(dt.datetime.combine(day, dt.time(10, 0)))
    assert not cal.in_continuous(dt.datetime.combine(day, dt.time(12, 0)))


def test_session_spec_rejects_inverted_intervals() -> None:
    from almanac.calendar import SessionSpec

    with pytest.raises(ValueError, match="empty or inverted"):
        SessionSpec("bad", ((dt.time(11, 30), dt.time(9, 30)),))


# --------------------------------------------------------------------------
# universe
# --------------------------------------------------------------------------


def test_membership_without_effective_dates_is_rejected() -> None:
    """The survivorship-bias guard: no effective date means no PIT answer."""
    bad = pl.DataFrame(
        {
            "index_code": ["000300.SH"],
            "symbol": ["600000"],
            "start_date": [None],
            "end_date": [None],
        }
    ).with_columns(pl.col("start_date").cast(pl.Date), pl.col("end_date").cast(pl.Date))
    with pytest.raises(UniverseError, match="null start_date"):
        Membership(bad)


def test_universe_is_point_in_time() -> None:
    """A name that joined in June is not a member in March."""
    m = Membership.from_frame(
        pl.DataFrame(
            {
                "index_code": ["000300.SH", "000300.SH"],
                "symbol": ["600000", "600001"],
                "start_date": [dt.date(2024, 1, 1), dt.date(2024, 6, 1)],
                "end_date": [None, None],
            }
        )
    )
    listings = pl.DataFrame(
        {
            "symbol": ["600000", "600001"],
            "list_date": [dt.date(2020, 1, 1)] * 2,
            "delist_date": [None, None],
        }
    ).with_columns(pl.col("delist_date").cast(pl.Date))
    u = Universe(membership=m, listings=listings)

    assert u.at("2024-03-01", "000300.SH") == {"600000"}
    assert u.at("2024-07-01", "000300.SH") == {"600000", "600001"}


def test_universe_respects_removal_date() -> None:
    m = Membership.from_frame(
        pl.DataFrame(
            {
                "index_code": ["000300.SH"],
                "symbol": ["600000"],
                "start_date": [dt.date(2024, 1, 1)],
                "end_date": [dt.date(2024, 6, 12)],
            }
        )
    )
    listings = pl.DataFrame(
        {"symbol": ["600000"], "list_date": [dt.date(2020, 1, 1)], "delist_date": [None]}
    ).with_columns(pl.col("delist_date").cast(pl.Date))
    u = Universe(membership=m, listings=listings)

    assert u.at("2024-06-11", "000300.SH") == {"600000"}
    with pytest.raises(UniverseError, match="no constituents"):
        u.at("2024-06-12", "000300.SH")


def test_universe_without_membership_refuses_index_rule() -> None:
    with pytest.raises(UniverseError, match="needs a Membership table"):
        Universe().at("2024-01-02", "000300.SH")


# --------------------------------------------------------------------------
# masks -- the step 1 gate
# --------------------------------------------------------------------------


def test_raw_fixture_contains_the_stale_price_trap(store, conn) -> None:
    """Precondition: the vendor-style feed really does carry a stale close.

    If this fails the gate test below proves nothing, because there would be
    no trap to catch.
    """
    daily = load_daily(conn)
    sym = next(s for s, days in store.suspensions.items() if days)
    day = store.suspensions[sym][0]

    row = daily.filter(
        (pl.col("symbol") == sym) & (pl.col("ts").dt.date() == day)
    )
    assert row.height == 1
    assert row["volume"][0] == 0.0
    assert row["close"][0] == pytest.approx(row["prev_close"][0])


def test_suspension_produces_null_not_stale_price(store, conn) -> None:
    """THE STEP 1 GATE.

    A suspended session must come out of the pipeline as null. A stale close
    would read as a 0% return and hand every reversal factor a free signal.
    """
    daily = load_daily(conn)
    masks = build_masks(daily)
    masked = apply_masks(daily, masks, columns=["close", "vwap", "volume"])

    checked = 0
    for sym, days in store.suspensions.items():
        for day in days:
            row = masked.filter((pl.col("symbol") == sym) & (pl.col("ts").dt.date() == day))
            assert row.height == 1, f"{sym} {day} vanished from the grid"
            assert row["close"][0] is None, f"{sym} {day} kept a stale close"
            assert row["vwap"][0] is None
            checked += 1

    assert checked == store.n_suspensions > 0


def test_masking_preserves_the_grid(store, conn) -> None:
    """Nulling, not dropping: the panel stays rectangular for cross-sectional ops."""
    daily = load_daily(conn)
    masks = build_masks(daily)
    masked = apply_masks(daily, masks, columns=["close"])
    assert masked.height == daily.height


def test_unmasked_rows_are_untouched(store, conn) -> None:
    """Masking must not perturb tradable observations."""
    daily = load_daily(conn)
    masks = build_masks(daily)
    masked = apply_masks(daily, masks, columns=["close"])

    joined = daily.join(masked.select("ts", "symbol", masked_close="close"), on=["ts", "symbol"])
    survivors = joined.filter(pl.col("masked_close").is_not_null())
    assert survivors.height > 0
    assert (survivors["close"] - survivors["masked_close"]).abs().max() == 0.0


def test_limit_up_is_flagged(store, conn) -> None:
    daily = load_daily(conn)
    masks = build_masks(daily)

    for sym, days in store.limit_ups.items():
        for day in days:
            row = masks.frame.filter(
                (pl.col("symbol") == sym) & (pl.col("ts").dt.date() == day)
            )
            assert row["limit_up"][0], f"{sym} {day} closed at the band but was not flagged"


def test_one_word_board_is_flagged(store, conn) -> None:
    daily = load_daily(conn)
    masks = build_masks(daily)

    for sym, days in store.one_word_boards.items():
        for day in days:
            row = masks.frame.filter(
                (pl.col("symbol") == sym) & (pl.col("ts").dt.date() == day)
            )
            assert row["one_word_board"][0], f"{sym} {day} was locked but not flagged"


def test_limit_up_blocks_buying_but_not_selling(store, conn) -> None:
    """A locked-up name can be exited, just not entered."""
    daily = load_daily(conn)
    masks = build_masks(daily)
    sym = next(s for s, d in store.limit_ups.items() if d)
    day = store.limit_ups[sym][0]

    def flag(side: str) -> bool:
        row = masks.tradable(side).filter(
            (pl.col("symbol") == sym) & (pl.col("ts").dt.date() == day)
        )
        return bool(row["tradable"][0])

    assert not flag("buy")
    # Selling is only blocked if the day was also a one-word board.
    if day not in store.one_word_boards.get(sym, ()):
        assert flag("sell")


def test_unknown_symbols_are_untradable_by_default(conn) -> None:
    """A coverage gap must not read as a clean name."""
    daily = load_daily(conn)
    masks = build_masks(daily)
    ghost = daily.head(1).with_columns(symbol=pl.lit("999999"))
    out = apply_masks(ghost, masks, columns=["close"])
    assert out["close"][0] is None


@pytest.mark.parametrize(
    ("symbol", "is_st", "expected"),
    [
        ("600000", False, 0.10),
        ("600000", True, 0.05),
        ("300001", False, 0.20),
        ("300001", True, 0.20),
        ("688001", False, 0.20),
        ("830001", False, 0.30),
    ],
)
def test_limit_bands_are_board_dependent(symbol: str, is_st: bool, expected: float) -> None:
    assert limit_pct(symbol, is_st=is_st) == expected


# --------------------------------------------------------------------------
# adjustment
# --------------------------------------------------------------------------


def test_backward_adjustment_is_point_in_time_stable() -> None:
    """Recomputing on a longer history must not change earlier values.

    This is the whole reason backward adjustment is the default. The same test
    against forward adjustment fails by construction.
    """
    days = [dt.datetime(2024, 1, d, 15, 0) for d in (2, 3, 4, 5)]
    grid = pl.DataFrame({"ts": days, "symbol": ["600000"] * 4})
    events = pl.DataFrame(
        {"ts": [days[1], days[3]], "symbol": ["600000"] * 2, "factor": [0.98, 0.97]}
    )

    full = cumulative_factors(events, grid, mode="backward")
    prefix = cumulative_factors(
        events.filter(pl.col("ts") <= days[2]), grid.head(3), mode="backward"
    )

    assert full.head(3)["adj_factor"].to_list() == pytest.approx(prefix["adj_factor"].to_list())


def test_forward_adjustment_warns_and_is_not_stable() -> None:
    days = [dt.datetime(2024, 1, d, 15, 0) for d in (2, 3, 4, 5)]
    bars = pl.DataFrame(
        {"ts": days, "symbol": ["600000"] * 4, "close": [10.0, 10.0, 10.0, 10.0]}
    )
    events = pl.DataFrame({"ts": [days[3]], "symbol": ["600000"], "factor": [0.95]})

    with pytest.warns(UserWarning, match="not point-in-time"):
        adjust(bars, events, mode="forward")


def test_adjust_keeps_the_unadjusted_price() -> None:
    """Cost models and limit bands need the traded price, not the adjusted one."""
    days = [dt.datetime(2024, 1, d, 15, 0) for d in (2, 3)]
    bars = pl.DataFrame({"ts": days, "symbol": ["600000"] * 2, "close": [10.0, 9.75], "volume": [100.0, 100.0]})
    events = pl.DataFrame({"ts": [days[1]], "symbol": ["600000"], "factor": [0.975]})

    out = adjust(bars, events, mode="backward")
    assert "close_raw" in out.columns
    assert out["close_raw"].to_list() == [10.0, 9.75]
    # The ex-day drop is removed from the adjusted series.
    assert out["close"][1] == pytest.approx(10.0, rel=1e-6)


def test_event_factors_from_cash_dividend() -> None:
    actions = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 3, 15, 0)],
            "symbol": ["600000"],
            "prev_close": [10.0],
            "cash_div": [0.25],
        }
    )
    out = event_factors(actions)
    assert out["factor"][0] == pytest.approx(0.975)


def test_volume_scales_inversely_to_price() -> None:
    """A split halves the price and doubles the shares; turnover is invariant."""
    days = [dt.datetime(2024, 1, d, 15, 0) for d in (2, 3)]
    bars = pl.DataFrame(
        {"ts": days, "symbol": ["600000"] * 2, "close": [10.0, 5.0], "volume": [100.0, 200.0]}
    )
    events = pl.DataFrame({"ts": [days[1]], "symbol": ["600000"], "factor": [0.5]})
    out = adjust(bars, events, mode="backward")

    turnover = out["close"] * out["volume"]
    assert turnover[0] == pytest.approx(turnover[1])
