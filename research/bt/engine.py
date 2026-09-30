"""Event-driven single-stock simulator: market-by-order book replay + virtual strategy orders.

One call replays one stock's day of events (see :mod:`bt.events`) in exchange sequence order,
maintaining the full order book (every resting order, its level, remaining size and priority),
and runs a simple round-trip strategy against it. Our orders are **virtual**: they are matched
against the reconstructed book but never change it (no market impact -- stated assumption).

Clocks
------
* A decision at local time ``t`` may only use what had *arrived* by ``t``: the "known" book is the
  replayed state as of the last event with (running max) arrival time <= ``t``.
* An order decided at ``t`` reaches the exchange at exchange time ``t + lat_send`` (the capture
  clock and the exchange clock agree to a few ms, see the quality gate) and meets the *true* book
  there. At the open the capture lags the exchange by up to ~80 s, so known and true books can
  differ a lot exactly when volume peaks -- the latency cost this simulator exists to show.

Fill models (``fill_model``)
----------------------------
0 ``mid_known``   fill instantly at the known mid (the idealised upper bound)
1 ``touch_known`` fill instantly at the known touch (pay the spread, no latency)
2 ``exchange``    the order meets the true book at arrival: aggressive orders walk the levels
                  (at most ``max_walk`` ticks); passive entries join the queue behind every
                  resting order at their price and fill only when the real flow reaches them
                  (orders ahead trade or cancel first; a trade through our price fills us)

A trip is: entry (side from the intent) -> hold ``hold`` ns after the entry is known filled ->
exit (always aggressive under model 2). One open trip per stock. With ``t1`` on, every trip's
sell leg must come out of the base position (today's buys are not sellable, T+1), trips stop at
the end-of-day decision time, and anything still open is closed there (the *restore* leg).
Without ``t1``, trips still open at the end are marked to the closing mid (leg MARK, no cost).

Every fill carries ``m0`` (known mid at the decision) and ``m1`` (true mid when the order reached
the exchange), which is what makes the exact PnL decomposition in :mod:`bt.pnl` possible.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import Dict

FILL_MID_KNOWN, FILL_TOUCH_KNOWN, FILL_EXCHANGE = 0, 1, 2
LEG_ENTRY, LEG_EXIT, LEG_RESTORE, LEG_MARK = 0, 1, 2, 3
_IDLE, _INFLIGHT, _REST, _HOLD, _EXITING, _DONE = 0, 1, 2, 3, 4, 5
_A_NONE, _A_ENTRY, _A_CROSS, _A_EXIT = 0, 1, 2, 3
_INF = np.int64(2**62)
# stats vector layout
S_UNRESOLVED_CANCEL, S_UNRESOLVED_PASSIVE, S_OUT_OF_BAND, S_SKIP_NO_BOOK, S_SKIP_T1 = 0, 1, 2, 3, 4
S_SKIP_BUSY, S_ABANDON_NO_LIQ, S_TRIPS, S_PASSIVE_FILLED, S_PASSIVE_CROSSED = 5, 6, 7, 8, 9
S_SKIP_SESSION, S_DEFER_SESSION, S_WALK_CROSSED = 10, 11, 12
N_STATS = 13
#: one window covering everything (tests, the book audit)
ALL_DAY = np.array([[0, 2**62]], np.int64)


@njit(cache=True, nogil=True)
def _slot(oid, key):
    """Slot of a resting order, or -1 (a typed miss, which ``dict.get(k, -1)`` is not in numba)."""
    if key in oid:
        return oid[key]
    return np.int64(-1)


@njit(cache=True, nogil=True)
def _known(tape_a, tape_bid, tape_ask, m, t):
    """(bid_idx, ask_idx) of the book known at local time t, from the first m tape rows."""
    k = np.searchsorted(tape_a[:m], t, side="right") - 1
    if k < 0:
        return -1, -1
    return tape_bid[k], tape_ask[k]


@njit(cache=True, nogil=True)
def _session_next(sess, t):
    """``t`` if it lies in a continuous-trading window, else the next window's start (or _INF).

    Outside continuous trading (call auctions, lunch) the replay book holds auction orders and
    is crossed, so nothing may be matched against it.
    """
    for w in range(sess.shape[0]):
        if t < sess[w, 0]:
            return sess[w, 0]
        if t < sess[w, 1]:
            return t
    return _INF


@njit(cache=True, nogil=True)
def _walk(sd, q, bidv, askv, best_bid, best_ask, n_ticks, max_walk):
    """Aggressive fill against the true book (no impact): returns (filled, notional in idx*qty)."""
    filled = 0
    notional = 0
    if best_bid >= 0 and best_ask < n_ticks and best_bid >= best_ask:
        return -1, 0  # crossed true book: refuse (caller counts it)
    if sd > 0:
        if best_ask >= n_ticks:
            return 0, 0
        t = best_ask
        lim = min(best_ask + max_walk, n_ticks - 1)
        while t <= lim and filled < q:
            v = askv[t]
            if v > 0:
                take = min(v, q - filled)
                filled += take
                notional += take * t
            t += 1
    else:
        if best_bid < 0:
            return 0, 0
        t = best_bid
        lim = max(best_bid - max_walk, 0)
        while t >= lim and filled < q:
            v = bidv[t]
            if v > 0:
                take = min(v, q - filled)
                filled += take
                notional += take * t
            t -= 1
    return filled, notional


VENUE_SSE, VENUE_SZSE = 0, 1  # mirrors bt.venues (kept literal: numba needs module constants)
MTAG_CP_BEST, MTAG_FAK, MTAG_FOK_ZERO = 11, 12, 15  # mirrors bt.events SZSE market-order tags


@njit(cache=True, nogil=True)
def _level_add(sd, lvl, q, bidv, askv, best_bid, best_ask):
    """Add resting size at a level; returns the updated best levels."""
    if sd > 0:
        bidv[lvl] += q
        if lvl > best_bid:
            best_bid = lvl
    else:
        askv[lvl] += q
        if lvl < best_ask:
            best_ask = lvl
    return best_bid, best_ask


@njit(cache=True, nogil=True)
def _take(oid, key, s, q, o_side, o_tick, o_rem, o_placed, bidv, askv, best_bid, best_ask, n_ticks):
    """Remove q from order slot s (its level only if it rests in the book)."""
    lvl = o_tick[s]
    if o_placed[s] == 1:
        if o_side[s] > 0:
            bidv[lvl] = max(bidv[lvl] - q, 0)
            if lvl == best_bid and bidv[lvl] == 0:
                while best_bid >= 0 and bidv[best_bid] == 0:
                    best_bid -= 1
        else:
            askv[lvl] = max(askv[lvl] - q, 0)
            if lvl == best_ask and askv[lvl] == 0:
                while best_ask < n_ticks and askv[best_ask] == 0:
                    best_ask += 1
    o_rem[s] -= q
    if o_rem[s] <= 0 and key in oid:
        oid.pop(key)
    return best_bid, best_ask


@njit(cache=True, nogil=True)
def _new_order(oid, key, sd, lvl, q, i, placed, o_side, o_tick, o_rem, o_prio, o_placed, nslot):
    """Register an order in a fresh slot; returns the next free slot."""
    oid[key] = nslot
    o_side[nslot] = sd
    o_tick[nslot] = lvl
    o_rem[nslot] = q
    o_prio[nslot] = i
    o_placed[nslot] = placed
    return nslot + 1


@njit(cache=True, nogil=True)
def _cancel_or_trade(i, kind, side, qty, id_a, id_b, oid, o_side, o_tick, o_rem, o_placed,
                     bidv, askv, best_bid, best_ask, n_ticks, stats):
    """Shared removal path: a cancel removes one order, a trade decrements both sides."""
    k = kind[i]
    for j in range(2 if k == 2 else 1):
        key = id_a[i] if j == 0 else id_b[i]
        s = _slot(oid, key)
        if s < 0:
            if k == 1:
                stats[S_UNRESOLVED_CANCEL] += 1
            elif (j == 0 and side[i] < 0) or (j == 1 and side[i] > 0):
                stats[S_UNRESOLVED_PASSIVE] += 1
            continue
        q = qty[i] if (qty[i] > 0 and qty[i] < o_rem[s]) else o_rem[s]
        best_bid, best_ask = _take(oid, key, s, q, o_side, o_tick, o_rem, o_placed, bidv, askv,
                                   best_bid, best_ask, n_ticks)
    return best_bid, best_ask


@njit(cache=True, nogil=True)
def _apply_sse(i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
               o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats):
    """SSE: adds are always priced (marketable orders appear only as their resting remainder)."""
    if kind[i] == 0:
        sd = side[i]
        if sd != 0:
            lvl = tick[i] - lo_tick
            if lvl < 0 or lvl >= n_ticks:
                stats[S_OUT_OF_BAND] += 1
            else:
                nslot = _new_order(oid, id_a[i], sd, lvl, qty[i], i, 1, o_side, o_tick, o_rem,
                                   o_prio, o_placed, nslot)
                best_bid, best_ask = _level_add(sd, lvl, qty[i], bidv, askv, best_bid, best_ask)
    else:
        best_bid, best_ask = _cancel_or_trade(i, kind, side, qty, id_a, id_b, oid, o_side, o_tick,
                                              o_rem, o_placed, bidv, askv, best_bid, best_ask,
                                              n_ticks, stats)
    return best_bid, best_ask, nslot


@njit(cache=True, nogil=True)
def _apply_szse(i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
                o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats, pkey):
    """SZSE: orders are published at entry, *before* their trades.

    A marketable order (market, or a limit crossing the opposite best) is therefore held out of
    the book while its own trades/cancel arrive (``pkey`` = its key, -1 when none) and only its
    remainder rests afterwards: a limit at its limit price, a market order at its fill price
    (counterparty-best-to-limit) -- unless a cancel for it arrives first (FAK / FOK / best-5).
    Placing it on arrival would briefly cross the book and offer a virtual order liquidity that
    was never there. Own-best (``orderType`` 3) is priced at the own side's best on arrival.
    """
    k = kind[i]
    # its own burst = its cancel, or a trade where it is the aggressor (as the passive side of a
    # trade it is already resting, so the remainder must be in the book first)
    own = (k == 1 and id_a[i] == pkey) or (
        k == 2 and ((side[i] > 0 and id_a[i] == pkey) or (side[i] < 0 and id_b[i] == pkey)))
    if pkey >= 0 and not own:
        s = _slot(oid, pkey)
        if s >= 0 and o_placed[s] == 0 and o_rem[s] > 0 and 0 <= o_tick[s] < n_ticks:
            o_placed[s] = 1
            best_bid, best_ask = _level_add(o_side[s], o_tick[s], o_rem[s], bidv, askv,
                                            best_bid, best_ask)
        pkey = -1
    if k == 0:
        sd = side[i]
        ot = otype[i]
        if sd != 0:
            if ot == 1 or ot >= MTAG_CP_BEST:
                # market order: priced by its fill; tags fak / fok_zero never rest (placed = 2)
                never = ot == MTAG_FAK or ot == MTAG_FOK_ZERO
                nslot = _new_order(oid, id_a[i], sd, -1, qty[i], i, 2 if never else 0, o_side,
                                   o_tick, o_rem, o_prio, o_placed, nslot)
                pkey = id_a[i]
            else:
                lvl = (best_bid if sd > 0 else best_ask) if ot == 3 else tick[i] - lo_tick
                if lvl < 0 or lvl >= n_ticks:
                    stats[S_OUT_OF_BAND] += 1
                else:
                    crosses = (sd > 0 and best_ask < n_ticks and lvl >= best_ask) or (
                        sd < 0 and best_bid >= 0 and lvl <= best_bid)
                    nslot = _new_order(oid, id_a[i], sd, lvl, qty[i], i, 0 if crosses else 1,
                                       o_side, o_tick, o_rem, o_prio, o_placed, nslot)
                    if crosses:
                        pkey = id_a[i]
                    else:
                        best_bid, best_ask = _level_add(sd, lvl, qty[i], bidv, askv, best_bid,
                                                        best_ask)
        return best_bid, best_ask, nslot, pkey
    if k == 2:
        # a market order learns its price from its fill
        for j in range(2):
            key = id_a[i] if j == 0 else id_b[i]
            s = _slot(oid, key)
            if s >= 0 and o_placed[s] == 0 and o_tick[s] < 0:
                o_tick[s] = tick[i] - lo_tick
    best_bid, best_ask = _cancel_or_trade(i, kind, side, qty, id_a, id_b, oid, o_side, o_tick,
                                          o_rem, o_placed, bidv, askv, best_bid, best_ask,
                                          n_ticks, stats)
    return best_bid, best_ask, nslot, pkey


@njit(cache=True, nogil=True)
def _last_mid(last_bid, last_ask, fallback):
    """Mid of the last uncrossed true book, else ``fallback``."""
    if last_bid >= 0 and last_ask >= 0:
        return (last_bid + last_ask) / 2.0
    return fallback


@njit(cache=True, nogil=True)
def simulate(kind, xts, ats, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, venue,
             int_t, int_side, fill_model, lat_sig, lat_send, lat_report, passive,
             passive_timeout, hold, trip_qty, max_walk, t1, base_qty, eod_local, sess):
    """Replay one stock and simulate the trips. See the module docstring for semantics.

    Returns ``(fills, stats, m_end, tape_bid, tape_ask)``: ``fills`` is a float64 matrix with columns
    trip, leg, side, qty, price, x_time, t_decision, m0, m1, passive, ahead_at_entry;
    ``stats`` counts book/strategy anomalies (``S_*``); ``m_end`` is the closing mid (book index);
    ``tape_bid`` / ``tape_ask`` are the last valid (uncrossed) best levels after each event
    (book index, -1 before the first), used to audit the replay against exchange snapshots.
    ``sess`` is an ``(k, 2)`` array of continuous-trading windows ``[start, end)`` on the ``xts``
    clock: entries arriving outside them are skipped, exits and passive crosses wait for the next
    window (and stay open for the mark / restore when there is none).
    """
    n = kind.shape[0]
    bidv = np.zeros(n_ticks, np.int64)
    askv = np.zeros(n_ticks, np.int64)
    oid = Dict.empty(key_type=types.int64, value_type=types.int64)
    o_side = np.zeros(n + 1, np.int64)
    o_tick = np.zeros(n + 1, np.int64)
    o_rem = np.zeros(n + 1, np.int64)
    o_prio = np.zeros(n + 1, np.int64)
    o_placed = np.zeros(n + 1, np.int8)
    sz_pending = np.int64(-1)
    nslot = 0
    best_bid = -1
    best_ask = n_ticks
    tape_a = np.empty(n, np.int64)
    tape_bid = np.empty(n, np.int64)
    tape_ask = np.empty(n, np.int64)
    amax = np.int64(0)
    last_bid = -1
    last_ask = -1
    stats = np.zeros(N_STATS, np.int64)

    nint = int_t.shape[0]
    maxf = 8 * nint + 64
    F = np.zeros((maxf, 11), np.float64)
    nf = 0

    state = _IDLE
    ii = 0
    trip = -1
    t_side = 0
    t_filled = 0
    last_fill_x = np.int64(0)
    act = _A_NONE
    act_x = _INF
    act_dec = np.int64(0)
    act_qty = 0
    dec_m0 = 0.0
    dec_m1 = 0.0
    end_local = np.int64(0)
    rest_tick = 0
    rest_rem = 0
    rest_prio = 0
    rest_ahead = 0
    rest_ahead0 = 0
    sold_today = 0
    eod_x = eod_local + (lat_send if fill_model == FILL_EXCHANGE else 0)
    eod_done = False
    m_end = -1.0

    for i in range(n + 1):
        xe = xts[i] if i < n else _INF
        # ------------------------------------------------ actions due before event i
        while True:
            # earliest of: pending action, next intent (when idle), end-of-day
            cand = _INF
            which = 0
            if act != _A_NONE and act_x < cand:
                cand = act_x
                which = 1
            if state == _IDLE and not eod_done:
                while ii < nint and int_t[ii] + lat_sig < end_local:
                    stats[S_SKIP_BUSY] += 1
                    ii += 1
                if ii < nint:
                    dx = int_t[ii] + lat_sig + (lat_send if fill_model == FILL_EXCHANGE else 0)
                    if dx < cand:
                        cand = dx
                        which = 2
            if not eod_done and eod_x < cand:
                cand = eod_x
                which = 3
            if cand > xe or cand == _INF:
                break
            m = i  # tape rows available
            if which == 3:
                eod_done = True
                if t1 and (state == _REST or state == _HOLD or state == _EXITING or state == _INFLIGHT):
                    if state == _INFLIGHT:
                        state = _DONE
                        act = _A_NONE
                    elif t_filled > 0:
                        # restore: close everything still open, aggressively, now
                        kb, ka = _known(tape_a, tape_bid, tape_ask, m, eod_local)
                        m0 = (kb + ka) / 2.0 if kb >= 0 and ka >= 0 else -1.0
                        m1 = (best_bid + best_ask) / 2.0 if best_bid >= 0 and best_ask < n_ticks and best_bid < best_ask else _last_mid(last_bid, last_ask, m0)
                        q_left = t_filled
                        f, notion = _walk(-t_side, q_left, bidv, askv, best_bid, best_ask, n_ticks, max_walk)
                        if f < 0:
                            stats[S_WALK_CROSSED] += 1
                            f = 0
                        if f > 0 and nf < maxf:
                            F[nf, 0] = trip; F[nf, 1] = LEG_RESTORE; F[nf, 2] = -t_side
                            F[nf, 3] = f; F[nf, 4] = notion / f; F[nf, 5] = cand; F[nf, 6] = eod_local
                            F[nf, 7] = m0; F[nf, 8] = m1; F[nf, 9] = 0; F[nf, 10] = 0
                            nf += 1
                        state = _DONE
                        act = _A_NONE
                    else:
                        state = _DONE
                        act = _A_NONE
                continue
            if which == 2:
                # new trip from intent ii
                t_dec = int_t[ii] + lat_sig
                sd = int_side[ii]
                ii += 1
                if _session_next(sess, cand) != cand:
                    stats[S_SKIP_SESSION] += 1
                    continue
                if t_dec >= eod_local:
                    eod_done = eod_done  # no new trips at / after the end-of-day decision
                    stats[S_SKIP_BUSY] += 1
                    continue
                if t1 and sold_today + trip_qty > base_qty:
                    stats[S_SKIP_T1] += 1
                    continue
                kb, ka = _known(tape_a, tape_bid, tape_ask, m, t_dec)
                if kb < 0 or ka < 0 or kb >= ka:
                    stats[S_SKIP_NO_BOOK] += 1
                    continue
                m0 = (kb + ka) / 2.0
                trip += 1
                stats[S_TRIPS] += 1
                t_side = sd
                t_filled = 0
                if t1:
                    sold_today += trip_qty  # the sell leg of this trip comes out of the base
                if fill_model != FILL_EXCHANGE:
                    px = m0 if fill_model == FILL_MID_KNOWN else (ka if sd > 0 else kb)
                    if nf < maxf:
                        F[nf, 0] = trip; F[nf, 1] = LEG_ENTRY; F[nf, 2] = sd; F[nf, 3] = trip_qty
                        F[nf, 4] = px; F[nf, 5] = cand; F[nf, 6] = t_dec; F[nf, 7] = m0; F[nf, 8] = m0
                        F[nf, 9] = 0; F[nf, 10] = 0
                        nf += 1
                    t_filled = trip_qty
                    last_fill_x = cand
                    state = _HOLD
                    act = _A_EXIT
                    act_dec = t_dec + hold
                    act_x = act_dec
                    act_qty = trip_qty
                    continue
                m1 = (best_bid + best_ask) / 2.0 if best_bid >= 0 and best_ask < n_ticks and best_bid < best_ask else _last_mid(last_bid, last_ask, m0)
                dec_m0 = m0
                dec_m1 = m1
                if passive:
                    lvl = best_bid if sd > 0 else best_ask
                    if lvl >= 0 and lvl < n_ticks:
                        rest_tick = lvl
                        rest_rem = trip_qty
                        rest_prio = i
                        rest_ahead = bidv[lvl] if sd > 0 else askv[lvl]
                        rest_ahead0 = rest_ahead
                        state = _REST
                        act = _A_CROSS
                        act_dec = t_dec + passive_timeout
                        act_x = act_dec + lat_send
                        continue
                f, notion = _walk(sd, trip_qty, bidv, askv, best_bid, best_ask, n_ticks, max_walk)
                if f < 0:
                    stats[S_WALK_CROSSED] += 1
                    f = 0
                if f == 0:
                    stats[S_ABANDON_NO_LIQ] += 1
                    state = _IDLE
                    end_local = t_dec
                    if t1:
                        sold_today -= trip_qty
                    continue
                if nf < maxf:
                    F[nf, 0] = trip; F[nf, 1] = LEG_ENTRY; F[nf, 2] = sd; F[nf, 3] = f
                    F[nf, 4] = notion / f; F[nf, 5] = cand; F[nf, 6] = t_dec; F[nf, 7] = m0; F[nf, 8] = m1
                    F[nf, 9] = 0; F[nf, 10] = 0
                    nf += 1
                t_filled = f
                last_fill_x = cand
                state = _HOLD
                act = _A_EXIT
                act_dec = cand + lat_report + hold
                act_x = act_dec + lat_send
                act_qty = f
                continue
            # which == 1: pending action
            nx = _session_next(sess, cand)
            if nx != cand:
                stats[S_DEFER_SESSION] += 1
                if nx == _INF:
                    act = _A_NONE  # no window left: stays open for the mark
                else:
                    act_dec += nx - cand
                    act_x = nx
                continue
            a = act
            act = _A_NONE
            if a == _A_CROSS:
                # passive timeout: cross whatever is still resting
                if state == _REST:
                    stats[S_PASSIVE_CROSSED] += 1 if rest_rem > 0 else 0
                    if rest_rem > 0:
                        kb, ka = _known(tape_a, tape_bid, tape_ask, m, act_dec)
                        m0 = (kb + ka) / 2.0 if kb >= 0 and ka >= 0 else dec_m0
                        m1 = (best_bid + best_ask) / 2.0 if best_bid >= 0 and best_ask < n_ticks and best_bid < best_ask else _last_mid(last_bid, last_ask, m0)
                        f, notion = _walk(t_side, rest_rem, bidv, askv, best_bid, best_ask, n_ticks, max_walk)
                        if f < 0:
                            stats[S_WALK_CROSSED] += 1
                            f = 0
                        if f > 0 and nf < maxf:
                            F[nf, 0] = trip; F[nf, 1] = LEG_ENTRY; F[nf, 2] = t_side; F[nf, 3] = f
                            F[nf, 4] = notion / f; F[nf, 5] = cand; F[nf, 6] = act_dec; F[nf, 7] = m0
                            F[nf, 8] = m1; F[nf, 9] = 0; F[nf, 10] = 0
                            nf += 1
                            t_filled += f
                            last_fill_x = cand
                        rest_rem = 0
                    if t_filled > 0:
                        state = _HOLD
                        act = _A_EXIT
                        act_dec = last_fill_x + lat_report + hold
                        act_x = act_dec + lat_send
                        act_qty = t_filled
                    else:
                        state = _IDLE
                        end_local = act_dec
                        if t1:
                            sold_today -= trip_qty
                continue
            if a == _A_EXIT:
                kb, ka = _known(tape_a, tape_bid, tape_ask, m, act_dec)
                m0 = (kb + ka) / 2.0 if kb >= 0 and ka >= 0 and kb < ka else -1.0
                if fill_model != FILL_EXCHANGE:
                    if m0 < 0:
                        act = _A_EXIT
                        act_dec = act_dec + 1_000_000_000
                        act_x = act_dec
                        continue
                    px = m0 if fill_model == FILL_MID_KNOWN else (kb if t_side > 0 else ka)
                    if nf < maxf:
                        F[nf, 0] = trip; F[nf, 1] = LEG_EXIT; F[nf, 2] = -t_side; F[nf, 3] = act_qty
                        F[nf, 4] = px; F[nf, 5] = cand; F[nf, 6] = act_dec; F[nf, 7] = m0; F[nf, 8] = m0
                        F[nf, 9] = 0; F[nf, 10] = 0
                        nf += 1
                    t_filled = 0
                    state = _IDLE
                    end_local = act_dec
                    continue
                m1 = (best_bid + best_ask) / 2.0 if best_bid >= 0 and best_ask < n_ticks and best_bid < best_ask else _last_mid(last_bid, last_ask, m0)
                f, notion = _walk(-t_side, act_qty, bidv, askv, best_bid, best_ask, n_ticks, max_walk)
                if f < 0:
                    stats[S_WALK_CROSSED] += 1
                    f = 0
                if f > 0 and nf < maxf:
                    F[nf, 0] = trip; F[nf, 1] = LEG_EXIT; F[nf, 2] = -t_side; F[nf, 3] = f
                    F[nf, 4] = notion / f; F[nf, 5] = cand; F[nf, 6] = act_dec; F[nf, 7] = m0
                    F[nf, 8] = m1; F[nf, 9] = 0; F[nf, 10] = 0
                    nf += 1
                t_filled -= f
                if t_filled > 0:
                    # no liquidity for the rest right now: retry one second later
                    act = _A_EXIT
                    act_dec = act_dec + 1_000_000_000
                    act_x = act_dec + lat_send
                    act_qty = t_filled
                else:
                    state = _IDLE
                    end_local = act_dec
                continue
        if i == n:
            break

        # ------------------------------------------------ resting passive order vs event i
        k = kind[i]
        if state == _REST and rest_rem > 0:
            if k == 2 and side[i] == -t_side:
                tt = tick[i] - lo_tick
                through = (t_side > 0 and tt < rest_tick) or (t_side < 0 and tt > rest_tick)
                fq = 0
                if through:
                    fq = rest_rem
                elif tt == rest_tick:
                    pk = id_b[i] if side[i] > 0 else id_a[i]
                    s = _slot(oid, pk)
                    if s >= 0 and o_prio[s] < rest_prio and rest_ahead > 0:
                        rest_ahead = max(rest_ahead - qty[i], 0)
                    elif s < 0 and rest_ahead > 0:
                        rest_ahead = max(rest_ahead - qty[i], 0)
                    else:
                        fq = min(qty[i], rest_rem)
                if fq > 0 and nf < maxf:
                    F[nf, 0] = trip; F[nf, 1] = LEG_ENTRY; F[nf, 2] = t_side; F[nf, 3] = fq
                    F[nf, 4] = rest_tick; F[nf, 5] = xts[i]; F[nf, 6] = act_dec - passive_timeout
                    F[nf, 7] = dec_m0; F[nf, 8] = dec_m1; F[nf, 9] = 1; F[nf, 10] = rest_ahead0
                    nf += 1
                    rest_rem -= fq
                    t_filled += fq
                    last_fill_x = xts[i]
                    if rest_rem == 0:
                        stats[S_PASSIVE_FILLED] += 1
                        state = _HOLD
                        act = _A_EXIT
                        act_dec = last_fill_x + lat_report + hold
                        act_x = act_dec + lat_send
                        act_qty = t_filled
            elif k == 1:
                s = _slot(oid, id_a[i])
                if s >= 0 and o_side[s] == t_side and o_tick[s] == rest_tick and o_prio[s] < rest_prio:
                    rest_ahead = max(rest_ahead - min(qty[i] if qty[i] > 0 else o_rem[s], o_rem[s]), 0)

        # ------------------------------------------------ apply event i to the book (per venue)
        if venue == VENUE_SZSE:
            best_bid, best_ask, nslot, sz_pending = _apply_szse(
                i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
                o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats, sz_pending)
        else:
            best_bid, best_ask, nslot = _apply_sse(
                i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
                o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats)
        # ------------------------------------------------ tape of the (known) book
        if ats[i] > amax:
            amax = ats[i]
        tape_a[i] = amax
        if best_bid >= 0 and best_ask < n_ticks and best_bid < best_ask:
            last_bid = best_bid
            last_ask = best_ask
            m_end = (best_bid + best_ask) / 2.0
        tape_bid[i] = last_bid
        tape_ask[i] = last_ask

    # trips still open at the end without t1: mark to the closing mid, no cost
    if t_filled > 0 and (state == _HOLD or state == _REST or state == _EXITING) and nf < maxf:
        F[nf, 0] = trip; F[nf, 1] = LEG_MARK; F[nf, 2] = -t_side; F[nf, 3] = t_filled
        F[nf, 4] = m_end; F[nf, 5] = xts[n - 1]; F[nf, 6] = xts[n - 1]; F[nf, 7] = m_end
        F[nf, 8] = m_end; F[nf, 9] = 0; F[nf, 10] = 0
        nf += 1
    return F[:nf], stats, m_end, tape_bid, tape_ask
