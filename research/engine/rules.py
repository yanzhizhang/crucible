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
}


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
    if what == "tr_open":
        return max(st.hi_f, st.op) - min(st.lo_f, st.op)
    if what == "logret":
        return _log(st.snap_px / st.snap_prev) if st.snap_prev == st.snap_prev and st.snap_prev > 0 else math.nan
    if what == "one":
        return 1.0
    raise KeyError(what)


def trade_value(st: Stock, what: str) -> float:
    """Value of a trade-clock average's input from the state just updated by the trade."""
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
        if clock == "snap":
            e.update(t, snap_value(st, what))
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
    if s >= 34200:
        st.vol += vol
        st.amt += amt
        if s < 53820:
            if side > 0:
                st.buy += amt
            elif side < 0:
                st.sell += amt
            st.zzgh.feed(t, amt if side > 0 else 0.0, amt if side < 0 else 0.0)
    for e, clock, what in st.ema.values():
        if clock == "trade":
            e.update(t, trade_value(st, what))
