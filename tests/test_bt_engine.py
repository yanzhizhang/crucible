"""Backtest engine on hand-built order books: queue position, book walking, T+1, PnL identity."""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from bt import engine as E  # noqa: E402
from bt.pnl import decompose, summarize, to_frame, with_fees  # noqa: E402
from toll.costs import CostModel  # noqa: E402

S = 1_000_000_000
LO = 900


def _events(rows):
    cols = list(zip(*rows, strict=True))
    return [np.array(c, dtype=np.int64) for c in cols]


def _run(rows, intents, venue=E.VENUE_SZSE, **kw):
    kind, xts, ats, side, tick, qty, id_a, id_b, otype = _events(rows)
    p = dict(fill_model=E.FILL_EXCHANGE, lat_sig=0, lat_send=0, lat_report=0, passive=False,
             passive_timeout=60 * S, hold=1 * S, trip_qty=100, max_walk=10, t1=False,
             base_qty=10**9, eod_local=10**6 * S, sess=E.ALL_DAY)
    p.update(kw)
    it = np.array([t for t, _ in intents], dtype=np.int64)
    isd = np.array([s for _, s in intents], dtype=np.int64)
    return E.simulate(kind, xts, ats, side, tick, qty, id_a, id_b, otype, LO, 200, venue, it, isd,
                      p["fill_model"], p["lat_sig"], p["lat_send"], p["lat_report"], p["passive"],
                      p["passive_timeout"], p["hold"], p["trip_qty"], p["max_walk"], p["t1"],
                      p["base_qty"], p["eod_local"], p["sess"])[:3]


def ev(k, t, side, tick, qty, a, b=0, ot=2):
    return (k, t * S, t * S + 10_000_000, side, tick, qty, a, b, ot)


QUEUE_BOOK = [
    ev(0, 1, +1, 1000, 300, 1),  # bid 10.00 x300 (ahead of us)
    ev(0, 2, -1, 1001, 500, 2),  # ask 10.01
    ev(0, 3, +1, 1000, 200, 3),  # bid 10.00 x200 (ahead of us)
    # our passive buy joins at t=3.5 behind 500
    ev(1, 4, +1, 0, 200, 3),  # order 3 cancels: ahead 500 -> 300
    ev(2, 5, -1, 1000, 300, 1, 1),  # sell hits order 1 (ahead): ahead -> 0, no fill for us yet
    ev(0, 6, +1, 1000, 100, 4),  # order 4 joins BEHIND us
    ev(2, 7, -1, 1000, 100, 4, 4),  # sell hits order 4 -> by priority it is us who fills
    ev(0, 9, +1, 999, 1000, 5),  # bid 9.99 appears (exit liquidity)
    ev(0, 11, -1, 1005, 100, 6),
]


def test_passive_queue_position():
    F, stats, _ = _run(QUEUE_BOOK, [(int(3.5 * S), +1)], passive=True)
    entry = F[F[:, 1] == E.LEG_ENTRY]
    exit_ = F[F[:, 1] == E.LEG_EXIT]
    assert entry.shape[0] == 1
    assert entry[0, 9] == 1  # passive
    assert entry[0, 10] == 500  # queue ahead at entry
    assert entry[0, 5] == 7 * S  # filled only when the flow reached us
    assert entry[0, 4] + LO == 1000
    assert exit_.shape[0] == 1
    assert exit_[0, 4] + LO == 999  # exited into the 9.99 bid once it existed
    assert stats[E.S_PASSIVE_FILLED] == 1


def test_no_fill_while_orders_ahead_trade():
    rows = QUEUE_BOOK[:5] + [ev(0, 20, -1, 1005, 100, 9)]  # stop before anyone behind us trades
    F, _, _ = _run(rows, [(int(3.5 * S), +1)], passive=True, passive_timeout=100 * S)
    assert (F[:, 1] == E.LEG_ENTRY).sum() == 0 or F[F[:, 1] == E.LEG_ENTRY][0, 9] == 0


def test_aggressive_walks_levels():
    rows = [
        ev(0, 1, +1, 999, 500, 1),
        ev(0, 2, -1, 1001, 100, 2),
        ev(0, 3, -1, 1002, 200, 3),
        ev(0, 9, +1, 998, 100, 4),
    ]
    F, _, _ = _run(rows, [(int(3.5 * S), +1)], trip_qty=250)
    e = F[F[:, 1] == E.LEG_ENTRY][0]
    assert e[3] == 250
    assert (e[4] + LO) / 100 == pytest.approx((100 * 10.01 + 150 * 10.02) / 250)


def test_t1_blocks_sells_beyond_base():
    rows = [ev(0, 1, +1, 999, 10**6, 1), ev(0, 2, -1, 1001, 10**6, 2), ev(0, 50, +1, 998, 1, 3)]
    F, stats, _ = _run(rows, [(3 * S, +1), (10 * S, +1)], t1=True, base_qty=100)
    assert stats[E.S_SKIP_T1] == 1
    assert (F[:, 1] == E.LEG_ENTRY).sum() == 1


def test_pnl_identity():
    F, _, m_end = _run(QUEUE_BOOK, [(int(3.5 * S), +1)], passive=True)
    fills = decompose(with_fees(to_frame(F, LO, m_end, "000001", "20260615"), CostModel(), "20260615"))
    s = summarize(fills)
    parts = s["signal"] + s["latency"] + s["spread"] + s["fees"] + s["restore"]
    assert parts == pytest.approx(s["total"], abs=1e-9)
    assert fills.filter(pl.col("side") < 0)["fee"].min() > fills.filter(pl.col("side") > 0)["fee"].max() - 1e-9


def _tape(rows, venue):
    kind, xts, ats, side, tick, qty, id_a, id_b, otype = _events(rows)
    z = np.zeros(0, np.int64)
    _, stats, _, tb, ta = E.simulate(kind, xts, ats, side, tick, qty, id_a, id_b, otype, LO, 200,
                                     venue, z, z, 2, 0, 0, 0, False, 0, 0, 100, 10, False, 0, 2**62,
                                     E.ALL_DAY)
    return stats, tb + LO, ta + LO


def test_szse_market_remainder_rests_at_fill_price():
    rows = [
        ev(0, 1, -1, 1001, 100, 1),  # ask 10.01 x100
        ev(0, 2, -1, 1003, 100, 5),  # ask 10.03
        ev(0, 3, +1, 0, 300, 2, ot=1),  # market buy 300 (counterparty best, rest to limit)
        ev(2, 3, +1, 1001, 100, 2, 1),  # fills 100 at 10.01; 200 left, held while in flight
        ev(2, 4, -1, 1001, 50, 2, 9),  # a later sell hits it: the remainder rests at 10.01
    ]
    stats, tb, _ = _tape(rows, E.VENUE_SZSE)
    assert tb[3] != 1001  # not yet in the book during its own burst
    assert tb[4] == 1001  # rests at its fill price once its burst is over
    assert stats[E.S_UNRESOLVED_PASSIVE] == 0


def test_szse_fak_remainder_cancelled():
    rows = [
        ev(0, 1, +1, 999, 100, 7),  # bid 9.99
        ev(0, 2, -1, 1001, 100, 1),
        ev(0, 3, +1, 0, 300, 2, ot=1),
        ev(2, 3, +1, 1001, 100, 2, 1),
        ev(1, 3, +1, 0, 200, 2),  # FAK: exchange cancels the remainder
    ]
    stats, tb, _ = _tape(rows, E.VENUE_SZSE)
    assert tb[4] == 999
    assert stats[E.S_UNRESOLVED_CANCEL] == 0


def test_sse_aggressor_never_added_is_not_an_error():
    rows = [
        ev(0, 1, -1, 1001, 100, 1),
        ev(0, 1, +1, 999, 100, 3),
        ev(2, 2, +1, 1001, 100, 77, 1),  # aggressor 77 was never published (SSE)
    ]
    stats, tb, ta = _tape(rows, E.VENUE_SSE)
    assert stats[E.S_UNRESOLVED_PASSIVE] == 0
    assert tb[2] == 999


def test_szse_marketable_limit_never_crosses_the_book():
    rows = [
        ev(0, 1, +1, 999, 100, 7),  # bid 9.99
        ev(0, 2, -1, 1001, 100, 1),  # ask 10.01
        ev(0, 3, -1, 998, 300, 2),  # marketable sell limit 9.98, published before its trades
        ev(2, 3, -1, 999, 100, 7, 2),  # it takes the 9.99 bid
        ev(0, 4, +1, 990, 100, 8),  # next unrelated event: remainder 200 now rests at 9.98
    ]
    stats, tb, ta = _tape(rows, E.VENUE_SZSE)
    assert ta[2] == 1001  # still 10.01 while the aggressive sell is in flight (no crossed book)
    assert ta[4] == 998  # its remainder rests at its limit afterwards


def _ev_frame(rows):
    cols = ("kind", "xts", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")
    df = pl.DataFrame([dict(zip(cols, r, strict=True)) for r in rows])
    return df.with_columns(pl.lit(300750, pl.Int32).alias("symbol_id"),
                           pl.lit(3554, pl.Int16).alias("market_id"))


def test_szse_market_tags_match_prototype_decision_tree():
    from bt.events import (MTAG_CP_BEST, MTAG_FAK, MTAG_FOK_ZERO, MTAG_SWEEP_FULL,
                           MTAG_UNFILLED, szse_market_tags)

    ms = 1_000_000
    rows = [
        ev(0, 1, +1, 0, 300, 10, ot=1), ev(2, 1, +1, 1001, 100, 10, 1),  # cp_best: 1 price, rests
        ev(0, 2, +1, 0, 300, 20, ot=1), ev(2, 2, +1, 1001, 100, 20, 2),  # fak: auto cancel
        (1, 2 * S + 50 * ms, 2 * S + 60 * ms, 1, 0, 200, 20, 0, 0),
        ev(0, 3, +1, 0, 200, 30, ot=1), ev(2, 3, +1, 1001, 100, 30, 3),  # sweep_full: 2 prices
        ev(2, 3, +1, 1002, 100, 30, 4),
        ev(0, 4, +1, 0, 100, 40, ot=1), (1, 4 * S + 5 * ms, 4 * S + 9 * ms, 1, 0, 100, 40, 0, 0),
        ev(0, 5, +1, 0, 100, 50, ot=1),  # unfilled, no cancel
    ]
    tags = dict(szse_market_tags(_ev_frame(rows)).select("id_a", "tag").iter_rows())
    assert tags == {10: MTAG_CP_BEST, 20: MTAG_FAK, 30: MTAG_SWEEP_FULL, 40: MTAG_FOK_ZERO,
                    50: MTAG_UNFILLED}


def test_szse_fak_tagged_remainder_never_rests():
    rows = [
        ev(0, 1, +1, 999, 100, 7),
        ev(0, 2, -1, 1001, 100, 1),
        ev(0, 3, +1, 0, 300, 2, ot=12),  # tagged fak by the pre-pass
        ev(2, 3, +1, 1001, 100, 2, 1),
        ev(0, 3, -1, 1005, 100, 9),  # unrelated event before the exchange's cancel arrives
        ev(1, 3, +1, 0, 200, 2),
    ]
    _, tb, _ = _tape(rows, E.VENUE_SZSE)
    assert tb[4] == 999  # the fak remainder never showed up as a 10.01 bid
