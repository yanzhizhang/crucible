r"""The update rules: what a snapshot and what a trade do to the state (the solved part of PM's engine).

``EMAS`` is the table of solved averages: prefix -> (clock, half-life s, value). Values are computed from the
state *after* the event's raw fields are applied (``price``, ``mid``, ``logp``, the day range in price units ``hl``,
the true range against the open ``tr_open``, the log return against the previous event ``logret``, the running
VWAP ``rvwap``, the constant ``one``). A NaN value skips the update (as ``feature_intraday.ema_at`` does).
"""

from __future__ import annotations

import math

from state import HJ_SAMPLE, XO_GRID, Stock

EMAS: dict[str, tuple[str, float, str]] = {
    "zzpk.p": ("snap", 300, "price"), "zzab2.r": ("snap", 300, "price"), "zzup.pf": ("snap", 300, "price"),
    "zzxu.p": ("snap", 300, "price"), "zzup.ps": ("snap", 600, "price"), "zzxu.a": ("snap", 300, "hl"),
    "zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
    "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one"),
    "zzbd.pEmaS": ("snap", 100, "logp"), "zzeb.p": ("snap", 100, "mid"), "zzbd.rEmaS": ("snap", 200, "logret"),
    "zzba.p": ("snap", 240, "price"), "zzba.r": ("snap", 300, "tr_open"),
    "zzbd.pEmaT": ("trade", 100, "logp"), "zzbd.rEmaT": ("trade", 200, "logret"), "zzqj.r": ("trade", 30, "rvwap"),
    # variants: "<pm key>|<tag>" is scored against the PM key next to the rule above (verify strips the tag).
    # clocks: snap / trade / both (updated by every event); "_wall" = no lunch removal (afternoon t + 5400),
    # "_exch" = the exchange stamp's trading clock; "|c<N>" (none now) = seeded from yesterday's PM state decayed by N s
    "zzba.p|t300snap": ("trade", 300, "snap_px"), "zzba.p|b300snap": ("both", 300, "snap_px"), "zzba.p|b300mid": ("both", 300, "mid"),
    "zzba.p|t300snapw": ("trade_wall", 300, "snap_px"), "zzba.p|t300snapx": ("trade_exch", 300, "snap_px"),
    "zzba.r|t300": ("trade", 300, "hl_trade"), "zzba.r|t300tro": ("trade", 300, "tro"), "zzba.r|b300hl": ("both", 300, "hl_trade"),
    "zzba.r|b300tro": ("both", 300, "tro"), "zzba.r|t300w": ("trade_wall", 300, "hl_trade"),
    "zzup.ps|w": ("snap_wall", 600, "price"), "zzup.ps|x": ("snap_exch", 600, "price"),
    "zzpk.p|w": ("snap_wall", 300, "price"), "zzbd.rEmaS|w": ("snap_wall", 200, "logret"),
}


def tc(s: float) -> float:
    """Trading clock of an exchange second of day (dcp_build.tc)."""
    return s - 34200 if s <= 41400 else (7200.0 if s < 46800 else s - 46800 + 7200)


def clock_time(mod: str, t: float, s: float) -> float:
    """The time an ema sees: t (local trading clock), t without lunch removal, or the exchange stamp's clock."""
    if mod == "wall":
        return t + 5400.0 if t > 7200 else t
    if mod == "exch":
        return tc(s)
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
            e.update(clock_time(mod, t, s), snap_value(st, what))
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
    st.nb_all[0] += amt * b
    st.nb_all[1] += amt * sl
    st.nb_vall[0] += vol * b
    st.nb_vall[1] += vol * sl
    if s < 53820:
        st.nb_open[0] += amt * b
        st.nb_open[1] += amt * sl
    if s >= 34200:
        st.vol += vol
        st.amt += amt
        if s < 54000:
            st.nb_1500[0] += amt * b
            st.nb_1500[1] += amt * sl
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
        if base != "snap":
            e.update(clock_time(mod, t, s), trade_value(st, what) if base == "trade" else snap_value(st, what))
