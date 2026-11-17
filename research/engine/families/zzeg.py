"""zzeg: the 4-day channel (previous 3 days + today's snapshot high / low) and the 300 s average of its mid, carried overnight."""
import math

from .base import PINNED, Family


def nmax(a: float, b: float) -> float:
    """max skipping NaN (NaN only when both are)."""
    return b if a != a else (a if b != b else (b if b > a else a))


def nmin(a: float, b: float) -> float:
    """min skipping NaN (NaN only when both are)."""
    return b if a != a else (a if b != b else (b if b < a else a))


def chan_high(st) -> float:
    """zzeg.c.high."""
    return nmax(st.ch_h, st.hi_f)


def chan_low(st) -> float:
    """zzeg.c.low."""
    return nmin(st.ch_l, st.lo_f)


def chan_span(st) -> float:
    """zzeg.c.span = (high + low) / 2."""
    h, lo = chan_high(st), chan_low(st)
    return math.nan if h != h or lo != lo else (h + lo) / 2


class Zzeg(Family):
    stem, status = "zzeg", PINNED
    evidence = "state: zzeg.c.high / low / span and zzeg.c.rEma equal PM's .jz 0.98-1.00 (R111)"
    emas = {"zzeg.c.rEma|c0": ("snap", 300, "span")}

    def dump(self, st, out):
        out["zzeg.c.high"], out["zzeg.c.low"], out["zzeg.c.span"] = chan_high(st), chan_low(st), chan_span(st)
