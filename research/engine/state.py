r"""The per-stock state of PM's intraday factor engine: what the day-end ``.jz`` file is a snapshot of.

Three building blocks, each a PM key family:

* ``Ema`` -- ``<p>.sum / .wgt / .t``: weight 1 per update, decayed by ``exp(-dt ln2 / half)`` on the trading clock
  (09:30 = 0, lunch removed, local arrival); the value PM uses is ``sum / wgt``;
* ``LastBefore`` -- a time grid whose slot holds the last value received before the slot's end (``zzfb``: the
  snapshot's total amount; the price grid ``zzjb`` is cut from);
* ``SlotSum`` -- a time grid of per-slot sums (``zzgh``: buy and sell amount, binned at the close).

``Stock`` is the shared tape plus one object per family (``families/``): the tape holds the raw running values several
families read (high / low / open / last, volume and amount from 09:30, buy / sell amounts, the snapshot high, the last
snapshot and trade prices); each family owns its averages, its grids and its keys. ``dump()`` writes every family's keys
under PM's names.
"""

from __future__ import annotations

import math
import re

SEED = re.compile(r"^(c|w|wt)(\d+)$")  # variant tags seeded from yesterday: c = sum + wgt, w = wgt only, wt = wgt only with t = 0


def seed_of(name: str) -> tuple[str, float] | None:
    """(kind, decay seconds) of a "<key>|<seed tag>" rule, None for every other rule."""
    m = SEED.match(name.partition("|")[2])
    return (m.group(1), float(m.group(2))) if m else None

SESSION = 14220.0  # continuous session on the trading clock: 09:30-11:30 + 13:00-14:57
LN2 = math.log(2.0)
JB_DROP = 0.0304  # zzjb: 0 when the slot's price is this far below the previous close, else 4
GH_EDGES = (-0.6, -0.2, 0.2, 0.6)  # zzgh: per-slot amount imbalance -> code 0..4
GH_EMPTY = 2  # code of a slot without trades (to be confirmed against PM)
HJ_SAMPLE = [min(k * 900.0 + 870.0, 14190.0) for k in range(16)]  # zzhj.c: 30 s before each quarter-hour end, last at 14:56:30


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
    """The shared tape of one stock for the day, plus its family objects and their averages."""

    def __init__(self, emas: dict[str, tuple[str, float, str]] | None = None, prev: dict[str, float] | None = None) -> None:
        from families import EMAS, ENGINE  # noqa: PLC0415 -- families import this module

        self.fams = [f() for f in ENGINE]
        self.ema = {p: (Ema(half), clock, what) for p, (clock, half, what) in (EMAS if emas is None else emas).items()}
        for p, (e, _, _) in self.ema.items():  # seeded variants: yesterday's state (sum / wgt, or wgt alone) decayed by N seconds
            sd = seed_of(p)
            base = p.partition("|")[0]
            if sd and prev and prev.get(f"{base}.wgt"):
                kind, n = sd
                d = math.exp(-n * e.k)
                e.s, e.w = (prev[f"{base}.sum"] * d if kind == "c" else 0.0), prev[f"{base}.wgt"] * d
                if kind == "wt":
                    e.t = 0.0
        self.high = self.low = self.open = self.last = math.nan  # trades, auctions included
        self.vol = self.amt = 0.0  # from 09:30 exchange time, close auction included
        self.buy = self.sell = 0.0  # continuous-session amounts for zzgh.nb
        # zzgh.nb variants, [buy, sell]: continuous volume, trade count, amount at the last snapshot price, order-flow amount / volume
        self.nb_vol, self.nb_cnt, self.nb_snappx, self.nb_ord, self.nb_ordvol = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
        # zzqj.v / tpv variants: whole day, without the close auction, local-clock session
        self.vol_all = self.amt_all = self.vol_o = self.amt_o = self.vol_loc = self.amt_loc = 0.0
        self.hm = math.nan  # highest snapshot price in the continuous session (exchange time)
        self.lm = math.nan  # lowest snapshot price in the continuous session
        self.sp_hi = self.sp_lo = math.nan  # highest / lowest snapshot price of the whole day
        self.c_high = self.c_low = math.nan  # highest / lowest trade price in the continuous session
        self.pc = self.op = math.nan  # previous close / open from the snapshots
        self.snap_px = self.snap_prev = math.nan  # last two snapshot prices (zzbd.rEmaS)
        self.trade_px = self.trade_prev = math.nan
        self.bid = self.ask = self.mid = math.nan
        self.hi_f = self.lo_f = math.nan  # the snapshot's own high / low fields
        self.ch_h = self.ch_l = math.nan  # zzeg.c: the previous 3 trading days' snapshot high / low (chan.bin)
        self.t_fast = self.t_slow = None  # clocks: last snapshot / last trade received

    def dump(self) -> dict[str, float]:
        """Write the state under PM's key names: the averages, then each family's own keys."""
        out: dict[str, float] = {}
        for p, (e, _clock, _) in self.ema.items():
            out[f"{p}.sum"], out[f"{p}.wgt"] = e.s, e.w
            if e.t is not None:
                out[f"{p}.t"] = e.t
        for f in self.fams:
            f.dump(self, out)
        return out
