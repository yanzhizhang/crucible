r"""The update rules: what a snapshot and what a trade do to the state (the solved part of PM's engine).

``EMAS`` is the table of solved averages, assembled from the family classes (``families/``): prefix -> (clock,
half-life s, value). Values are computed from the state *after* the event's raw fields are applied (``price``, ``mid``, ``logp``, the day range in price units ``hl``,
the true range against the open ``tr_open``, the log return against the previous event ``logret``, the running
VWAP ``rvwap``, the constant ``one``). A NaN value skips the update (as ``feature_intraday.ema_at`` does).
"""

from __future__ import annotations

import math

from families import EMAS as _FAMILY_EMAS
from families.zzeg import chan_high, chan_low, chan_span  # noqa: F401 -- re-exported
from state import SESSION, Stock

# The table of solved averages is assembled from the family classes: prefix -> (clock, half-life s, value).
# variants: "<pm key>|<tag>" is scored against the PM key next to the rule above (verify strips the tag).
# clocks: snap / trade / both (updated by every event); "_wall" = no lunch removal (afternoon t + 5400),
# "_exch" = the exchange stamp's trading clock; "|c<N>" = seeded from yesterday's PM state decayed by N s
# both = snapshots and trades; bothc = both plus the SZSE cancel records of the transaction stream; bothn = both with each
# event's own value; all = everything incl. orders (state value).
EMAS: dict[str, tuple[str, float, str]] = dict(_FAMILY_EMAS)


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
    if what == "span":  # zzeg.c: (high + low) / 2 of the previous 3 days and today's snapshot high / low fields
        return chan_span(st)
    if what == "mh_dd2":  # zzmh.r: (2 x drawdown from zzmh.hm)^2 (R124b: PM / ((hm - p) / hm)^2 = 4.0000 on 294 stocks)
        d = (st.hm - st.snap_px) / st.hm
        return 4.0 * d * d
    if what == "rvwap":
        return st.amt / st.vol if st.vol > 0 else math.nan
    if what == "last_px":
        return st.last
    if what in ("hl_sp", "hl_spall", "hl_cont", "r_sp", "r_spall", "r_cont"):  # narrower ranges and the day range over them
        v = {"sp": st.hm - st.lm, "spall": st.sp_hi - st.sp_lo, "cont": st.c_high - st.c_low}[what.partition("_")[2]]
        if what.startswith("hl_"):
            return v
        return (st.hi_f - st.lo_f) / v if v > 0 else math.nan
    if what in ("ema_tr", "r_ema_dema"):  # zzvx: the range average as an input (double smoothing), and average / double average
        e, d = st.ema.get("zzvx.trEmaD"), st.ema.get("zzvx.trDEmaD")
        ev = e[0].value() if e else math.nan
        if what == "ema_tr":
            return ev
        dv = d[0].value() if d else math.nan
        return ev / dv if dv > 0 else math.nan
    if what == "tr_pc":  # true range against the previous close: the day's range widened to include it
        return max(st.hi_f, st.pc) - min(st.lo_f, st.pc)
    if what == "tr_ratio":  # true range / the day's own range (1 unless the previous close is outside the range)
        return (max(st.hi_f, st.pc) - min(st.lo_f, st.pc)) / (st.hi_f - st.lo_f) if st.hi_f - st.lo_f > 0 else math.nan
    if what == "trade_ratio":  # the trades' day range / the snapshot fields' range
        return (st.high - st.low) / (st.hi_f - st.lo_f) if st.hi_f - st.lo_f > 0 else math.nan
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
    if s >= 34200 and price > 0 and not (st.hm >= price):  # no end: the SH closing snapshot (stamped after 15:00:00) counts (R193)
        st.hm = price
    if 34200 <= s < 53820 and price > 0 and not (st.lm <= price):
        st.lm = price
    if price > 0:
        if not (st.sp_hi >= price):
            st.sp_hi = price
        if not (st.sp_lo <= price):
            st.sp_lo = price
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base != "trade":
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, snap_value(st, what))
    for f in st.fams:
        f.on_snap(st, t, s, lsod, amount)


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
        if 34200 <= s < 53820:
            st.c_high = price if not (st.c_high >= price) else st.c_high
            st.c_low = price if not (st.c_low <= price) else st.c_low
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
    for f in st.fams:
        f.on_trade(st, t, s, amt, side)
    for e, clock, what in st.ema.values():
        base, _, mod = clock.partition("_")
        if base != "snap":  # trade / tradec / both / bothc / bothn / all
            tt = clock_time(mod, t, s)
            if tt == tt:
                e.update(tt, trade_value(st, what) if base in ("trade", "tradec", "bothn") else snap_value(st, what))
