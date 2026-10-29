r"""The per-stock state of PM's intraday factor engine: what the day-end ``.jz`` file is a snapshot of.

Three building blocks, each a PM key family:

* ``Ema`` -- ``<p>.sum / .wgt / .t``: weight 1 per update, decayed by ``exp(-dt ln2 / half)`` on the trading clock
  (09:30 = 0, lunch removed, local arrival); the value PM uses is ``sum / wgt``;
* ``LastBefore`` -- a time grid whose slot holds the last value received before the slot's end (``zzfb``: the
  snapshot's total amount; the price grid ``zzjb`` is cut from);
* ``SlotSum`` -- a time grid of per-slot sums (``zzgh``: buy and sell amount, binned at the close).

``Stock`` owns them plus the scalars (high / low / open / last, volume and amount from 09:30, the continuous-session
amount imbalance ``zzgh.nb``, the snapshot high ``zzmh.hm``, the 10 s price grids ``zzxo.r`` / ``zzui.r``, the
quarter-hour mids ``zzhj.c``) and ``dump()`` writes them under PM's key names.
"""

from __future__ import annotations

import math

SESSION = 14220.0  # continuous session on the trading clock: 09:30-11:30 + 13:00-14:57
LN2 = math.log(2.0)
JB_DROP = 0.0304  # zzjb: 0 when the slot's price is this far below the previous close, else 4
GH_EDGES = (-0.6, -0.2, 0.2, 0.6)  # zzgh: per-slot amount imbalance -> code 0..4
GH_EMPTY = 2  # code of a slot without trades (to be confirmed against PM)
HJ_SAMPLE = [min(k * 900.0 + 870.0, 14190.0) for k in range(16)]  # zzhj.c: 30 s before each quarter-hour end, last at 14:56:30
XO_GRID = {"zzxo": 30, "zzui": 10}  # 10 s price grids ending at 14:57:00 local time (seconds of day)


class Ema:
    """One decayed average: sum, weight and the time of the last update."""

    __slots__ = ("k", "s", "t", "w")

    def __init__(self, half: float) -> None:
        self.k, self.s, self.w, self.t = LN2 / half, 0.0, 0.0, None

    def update(self, t: float, x: float) -> None:
        """Fold one value in at time t."""
        if x != x:  # NaN: no update, no decay (the next update decays from the last real one)
            return
        d = 1.0 if self.t is None else math.exp(-(t - self.t) * self.k)
        self.s, self.w, self.t = self.s * d + x, self.w * d + 1.0, t

    def value(self) -> float:
        """Return sum / weight."""
        return self.s / self.w if self.w else math.nan


class LastBefore:
    """Grid of n slots over the session: slot i = the last value fed strictly before its end (i + 1) * step."""

    __slots__ = ("cur", "nxt", "step", "vals")

    def __init__(self, n: int, fill: float = 0.0) -> None:
        self.step, self.vals, self.cur, self.nxt = SESSION / n, [fill] * n, math.nan, 0

    def feed(self, t: float, v: float) -> None:
        """Record a value received at t."""
        while self.nxt < len(self.vals) and (self.nxt + 1) * self.step <= t:
            if self.cur == self.cur:  # nothing received yet keeps the fill
                self.vals[self.nxt] = self.cur
            self.nxt += 1
        self.cur = v

    def close(self) -> list[float]:
        """Finish the remaining slots with the last value and return the grid."""
        while self.nxt < len(self.vals):
            if self.cur == self.cur:
                self.vals[self.nxt] = self.cur
            self.nxt += 1
        return self.vals


class SlotSum:
    """Grid of n slots: per-slot sums of one or more weights (events outside the session ignored)."""

    __slots__ = ("step", "vals")

    def __init__(self, n: int, parts: int) -> None:
        self.step, self.vals = SESSION / n, [[0.0] * n for _ in range(parts)]

    def feed(self, t: float, *w: float) -> None:
        """Add the weights to the slot of t."""
        if 0 <= t < SESSION:
            k = int(t // self.step)
            for part, x in zip(self.vals, w, strict=True):
                part[k] += x


class Stock:
    """Everything the engine keeps for one stock during the day."""

    def __init__(self, emas: dict[str, tuple[str, float, str]]) -> None:
        self.ema = {p: (Ema(half), clock, what) for p, (clock, half, what) in emas.items()}
        self.high = self.low = self.open = self.last = math.nan  # trades, auctions included
        self.vol = self.amt = 0.0  # from 09:30 exchange time, close auction included
        self.buy = self.sell = 0.0  # continuous-session amounts for zzgh.nb
        self.hm = math.nan  # highest snapshot price in the continuous session (exchange time)
        self.pc = self.op = math.nan  # previous close / open from the snapshots
        self.snap_px = self.snap_prev = math.nan  # last two snapshot prices (zzbd.rEmaS)
        self.trade_px = self.trade_prev = math.nan
        self.bid = self.ask = self.mid = math.nan
        self.hi_f = self.lo_f = math.nan  # the snapshot's own high / low fields
        self.t_fast = self.t_slow = None  # clocks: last snapshot / last trade received
        self.zzfb = LastBefore(711)
        self.jb_px = LastBefore(948, math.nan)
        self.zzgh = SlotSum(948, 2)
        self.hj = [math.nan] * 16  # mid as of each quarter-hour sample
        self.grid = {fam: [math.nan] * n for fam, n in XO_GRID.items()}

    def dump(self) -> dict[str, float]:
        """Write the state under PM's key names."""
        out: dict[str, float] = {}
        for p, (e, _clock, _) in self.ema.items():
            out[f"{p}.sum"], out[f"{p}.wgt"] = e.s, e.w
            if e.t is not None:
                out[f"{p}.t"] = e.t
        for k in ("zzpk.high", "zzvx.high", "zzxu.high"):
            out[k] = self.high
        for k in ("zzpk.low", "zzvx.low", "zzxu.low"):
            out[k] = self.low
        out["zzpk.open"] = self.open
        for k in ("zzxu.last", "zzxo.last", "cdc.lp"):
            out[k] = self.last
        out["zzqj.v"] = out["zzvm.c.v"] = self.vol
        out["zzqj.tpv"] = out["zzvm.c.tpv"] = self.amt
        out["zzgh.nb"] = (self.buy - self.sell) / (self.buy + self.sell) if self.buy + self.sell else math.nan
        out["zzmh.hm"] = self.hm
        for fam, g in self.grid.items():
            for k, v in enumerate(g):
                out[f"{fam}.r.{k}"] = v
        for k, v in enumerate(self.hj):
            out[f"zzhj.c.{k}"] = v
        for k, v in enumerate(self.zzfb.close()):
            out[f"zzfb.c.{k}"] = v
        for k, px in enumerate(self.jb_px.close()):
            out[f"zzjb.c.{k}"] = 0.0 if px == px and self.pc == self.pc and px <= self.pc * (1 - JB_DROP) else 4.0
        for k, (b, s) in enumerate(zip(*self.zzgh.vals, strict=True)):
            if b + s > 0:
                x = (b - s) / (b + s)
                code = sum(1 for e in GH_EDGES if e <= x)  # searchsorted, right: a value on an edge goes up
            else:
                code = GH_EMPTY
            out[f"zzgh.c.{k}"] = float(code)
        return out
