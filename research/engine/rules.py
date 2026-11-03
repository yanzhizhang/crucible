r"""The update rules: what a snapshot and what a trade do to the state (the solved part of PM's engine).

``EMAS`` is the table of solved averages: prefix -> (clock, half-life s, value). Values are computed from the
state *after* the event's raw fields are applied (``price``, ``mid``, ``logp``, the day range in price units ``hl``,
the true range against the open ``tr_open``, the log return against the previous event ``logret``, the running
VWAP ``rvwap``, the constant ``one``). A NaN value skips the update (as ``feature_intraday.ema_at`` does).
"""

from __future__ import annotations

import math

from state import HJ_SAMPLE, SESSION, XO_GRID, Stock

EMAS: dict[str, tuple[str, float, str]] = {
    "zzpk.p": ("snap", 300, "price"), "zzab2.r": ("snap", 300, "price"), "zzup.pf": ("snap", 300, "price"),
    "zzxu.p": ("snap", 300, "price"), "zzup.ps": ("snap", 600, "price"), "zzxu.a": ("snap", 300, "hl"),
    "zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
    "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one"),
    "zzbd.pEmaS": ("snap", 100, "logp"), "zzeb.p": ("snap", 100, "mid"), "zzbd.rEmaS": ("snap", 200, "logret"),
    "zzba.p": ("both", 300, "snap_px"), "zzba.r": ("both", 300, "hl_trade"),  # decided on 97: 1e-4 exact every day
    "zzbd.pEmaT": ("trade", 100, "logp"), "zzbd.rEmaT": ("trade", 200, "logret"), "zzqj.r": ("trade", 30, "rvwap"),
    # variants: "<pm key>|<tag>" is scored against the PM key next to the rule above (verify strips the tag).
    # clocks: snap / trade / both (updated by every event); "_wall" = no lunch removal (afternoon t + 5400),
    # "_exch" = the exchange stamp's trading clock; "|c<N>" (none now) = seeded from yesterday's PM state decayed by N s
    # both = snapshots and trades; bothc = both plus the SZSE cancel records of the transaction stream; bothn = both with each
    # event's own value; all = everything incl. orders (state value). zzup.ps: 1e-6 residual ~1e-7 relative, unexplained --
    # lunch, exchange clock, carry (c/w/wt seeds), price-less snapshots, 09:30 start all tried and all identical.
    # zzba: both-event clock (snapshots and trades; cancels / orders tested and excluded), 1e-4 exact on every day. The 1e-6
    # residual (zzba.p 19%, zzba.r 4%) only hits keys both feeds tick, so it is the two feeds' interleaving at PM's receiver
    # differing from ours -- not recoverable from data. The last trade price and price-less snapshots as 0 were worse.
}


def tc(s: float) -> float:
    """Trading clock of an exchange second of day (dcp_build.tc)."""
    return s - 34200 if s <= 41400 else (7200.0 if s < 46800 else s - 46800 + 7200)


def clock_time(mod: str, t: float, s: float) -> float:
    """The time an ema sees: t (local trading clock), t without lunch removal, the exchange stamp's clock; NaN = skip (o930 before 09:30)."""
    if mod == "wall":
        return t + 5400.0 if t > 7200 else t
    if mod == "exch":
        return tc(s)
    if mod == "o930":
        return t if s >= 34200 else math.nan
    return t


def _log(x: float) -> float:
    return math.log(x) if x > 0 else math.nan


def snap_value(st: Stock, what: str) -> float:
    """Value of a snapshot-clock average's input from the state just updated by the snapshot."""
    if what == "price":
        return st.snap_px
    if what == "mid":
        return st.mid
    if what == "logp":
        return _log(st.snap_px)
    if what == "hl":
        return st.hi_f - st.lo_f
    if what == "hl0":  # a missing range counts as 0 (and the update still carries weight)
        v = st.hi_f - st.lo_f
        return v if v == v else 0.0
    if what == "hl_valid":  # indicator: this snapshot had a range
        return 1.0 if st.hi_f - st.lo_f == st.hi_f - st.lo_f else 0.0
    if what == "tr_open":
        return max(st.hi_f, st.op) - min(st.lo_f, st.op)
    if what == "logret":
        return _log(st.snap_px / st.snap_prev) if st.snap_prev == st.snap_prev and st.snap_prev > 0 else math.nan
    if what == "one":
        return 1.0
    if what == "hl_trade":
        return st.high - st.low
    if what == "tro":  # true range of the day's trades against the snapshot open
        return max(st.high, st.op) - min(st.low, st.op)
    if what == "snap_px":
        return st.snap_px
    if what == "price0":  # a price-less snapshot counts as 0
        return st.snap_px if st.snap_px == st.snap_px else 0.0
    if what == "price_pc":  # a price-less snapshot counts as the previous close
        return st.snap_px if st.snap_px == st.snap_px else st.pc
    if what == "rvwap":
        return st.amt / st.vol if st.vol > 0 else math.nan
    if what == "last_px":
        return st.last
    raise KeyError(what)


def trade_value(st: Stock, what: str) -> float:
    """Value of a trade-clock average's input from the state just updated by the trade."""
    if what == "price":
        return st.trade_px
    if what in ("hl", "hl_trade"):
        return st.high - st.low
    if what == "tro":
        return max(st.high, st.op) - min(st.low, st.op)
    if what == "mid":
        return st.mid
    if what == "snap_px":
        return st.snap_px
    if what == "hl_snap":
        return st.hi_f - st.lo_f
    if what == "logp":
        return _log(st.trade_px)
    if what == "logret":
        return _log(st.trade_px / st.trade_prev) if st.trade_prev == st.trade_prev and st.trade_prev > 0 else math.nan
    if what == "rvwap":
        return st.amt / st.vol if st.vol > 0 else math.nan
    if what == "last_px":
        return st.last
    if what == "price0":
        return st.snap_px if st.snap_px == st.snap_px else 0.0
    raise KeyError(what)


def on_snap(st: Stock, t: float, s: float, lsod: float, price: float, b1: float, a1: float, hi: float, lo: float,
            op: float, pc: float, amount: float) -> None:
    """Apply a snapshot received at trading-clock t (exchange second of day s, local second of day lsod)."""
    st.t_fast = t
    st.snap_prev, st.snap_px = st.snap_px, price if price > 0 else math.nan
    st.bid, st.ask = (b1 if b1 > 0 else math.nan), (a1 if a1 > 0 else math.nan)
    st.mid = (b1 + a1) / 2 if b1 > 0 and a1 > 0 else st.snap_px
    st.hi_f, st.lo_f = (hi if hi > 0 else math.nan), (lo if lo > 0 else math.nan)
    if op > 0:
        st.op = op
    if pc > 0:
        st.pc = pc
    if 34200 <= s < 53820 and price > 0 and not (st.hm >= price):
        st.hm = price
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base != "trade":
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, snap_value(st, what))
    st.zzfb.feed(t, amount)
    st.jb_px.feed(t, st.snap_px)
    for k, ts in enumerate(HJ_SAMPLE):  # last snapshot received at or before each sample
        if t <= ts:
            st.hj[k] = st.mid
    for fam, n in XO_GRID.items():  # last snapshot received at or before 14:57:00 - (n-1-k) x 10 s, local time
        if price > 0:
            g = st.grid[fam]
            for k in range(n):
                if lsod <= 53820 - (n - 1 - k) * 10:
                    g[k] = price


def on_order(st: Stock, t: float, s: float, price: float = 0.0, vol: float = 0.0, side: float = 0.0) -> None:
    """Apply an order event received at trading-clock t: the all-event averages move, the order-flow imbalances count it."""
    if 34200 <= s < 53820 and price > 0:
        amt = vol * price
        st.nb_ord[0] += amt * (side > 0)
        st.nb_ord[1] += amt * (side < 0)
        st.nb_ordvol[0] += vol * (side > 0)
        st.nb_ordvol[1] += vol * (side < 0)
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base == "all":
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, snap_value(st, what))


def on_cancel(st: Stock, t: float, s: float) -> None:
    """Apply a cancel record of the transaction stream (SZSE): the bothc / all averages move, on the state as it stands."""
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base in ("bothc", "all", "tradec"):
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, trade_value(st, what) if base == "tradec" else snap_value(st, what))


def on_trade(st: Stock, t: float, s: float, price: float, vol: float, side: float) -> None:
    """Apply a trade received at trading-clock t (exchange second of day s); side +1 buy / -1 sell / 0."""
    st.t_slow = t
    st.trade_prev, st.trade_px = st.trade_px, price if price > 0 else math.nan
    if price > 0:
        st.high = price if not (st.high >= price) else st.high
        st.low = price if not (st.low <= price) else st.low
        if st.open != st.open:
            st.open = price
        st.last = price
    amt = vol * price
    b, sl = (side > 0), (side < 0)
    if 34200 <= s < 53820:  # zzgh.nb variants: trade count, amount at the last snapshot price
        st.nb_cnt[0] += b
        st.nb_cnt[1] += sl
        px = st.snap_px if st.snap_px == st.snap_px else price
        st.nb_snappx[0] += vol * px * b
        st.nb_snappx[1] += vol * px * sl
    st.vol_all += vol  # zzqj.v / tpv variants: the whole day, the day without the close auction, the local-clock session
    st.amt_all += amt
    if 34200 <= s < 54000:
        st.vol_o += vol
        st.amt_o += amt
    if 0 <= t < SESSION:
        st.vol_loc += vol
        st.amt_loc += amt
    if s >= 34200:
        st.vol += vol
        st.amt += amt
        if s < 53820:
            if side > 0:
                st.buy += amt
            elif side < 0:
                st.sell += amt
            st.nb_vol[0] += vol * b
            st.nb_vol[1] += vol * sl
            st.zzgh.feed(t, amt if side > 0 else 0.0, amt if side < 0 else 0.0)
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base != "snap":  # trade / tradec / both / bothc / bothn / all
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, trade_value(st, what) if base in ("trade", "tradec", "bothn") else snap_value(st, what))
