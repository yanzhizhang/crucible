"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is being worked out against PM's day-end file. Known: the band is mid -+ 3 w (mid = (high + low) / 2
of the snapshot fields, w = the range average zzxu.a CARRIED from yesterday, R164); with the "simple" rule (the inactive
band recomputed every snapshot, the active one only tightens, a flip when the price crosses the active band; "|c0a") the
inactive band is exact on ~85% of stocks and the active one never; when the band starts makes no difference (R169) and our
band flips down too rarely, so PM's active band is TIGHTER than a ratchet of mid - 3 w. Under test: running extremes with
the carried width over more centres and multipliers, "|m.<centre>.<mult>.<width>" = the highest centre - mult x width (as
zzxu.lower) / the lowest centre + mult x width (as zzxu.upper) of the day; centre mid / px / p / hi / lo; mult 1, 1.5, 2,
2.5, 3; width c0 (zzxu.a carried) / tr (zzvx.trEma) / de (zzvx.trDEma). A stock that never flipped has its active band
equal to the right one.
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0
CENTRES = ("mid", "px", "p", "hi", "lo")
MULTS = (1.0, 1.5, 2.0, 2.5, 3.0)
WIDTHS = ("c0", "tr", "de")
WIDTH_KEY = {"c0": "zzxu.a|c0", "tr": "zzvx.trEmaD", "de": "zzvx.trDEmaD"}


class Zzxu(Family):
    stem, status = "zzxu", PINNED
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; zzxu.a carries overnight (R164); upper / lower / dir under test; columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl"), "zzxu.a|c0": ("snap", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.upper = self.lower = math.nan
        self.dir = 1.0
        n = len(CENTRES) * len(MULTS) * len(WIDTHS)
        self.max_lower, self.min_upper = [math.nan] * n, [math.nan] * n

    def on_snap(self, st, t, s, lsod, amount):
        val = lambda k: st.ema[k][0].value() if k in st.ema else math.nan  # noqa: E731
        px = st.snap_px
        if not (px == px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
        ws = [val(WIDTH_KEY[w]) for w in WIDTHS]
        i = 0
        for c in (mid, px, val("zzxu.p"), st.hi_f, st.lo_f):
            for m in MULTS:
                for w in ws:
                    if c == c and w == w:
                        lo, up = c - m * w, c + m * w
                        if not (self.max_lower[i] >= lo):
                            self.max_lower[i] = lo
                        if not (self.min_upper[i] <= up):
                            self.min_upper[i] = up
                    i += 1
        w = ws[0]
        if w != w:
            return
        bu, bl = mid + BAND_MULT * w, mid - BAND_MULT * w
        if self.upper != self.upper:
            self.upper, self.lower = bu, bl
        elif self.dir > 0:
            self.upper = bu
            if bl > self.lower:
                self.lower = bl
            if px < self.lower:
                self.dir, self.lower = -1.0, bl
        else:
            self.lower = bl
            if bu < self.upper:
                self.upper = bu
            if px > self.upper:
                self.dir, self.upper = 1.0, bu

    def dump(self, st, out):
        out["zzxu.high"], out["zzxu.low"], out["zzxu.last"] = st.high, st.low, st.last
        out["zzxu.upper|c0a"], out["zzxu.lower|c0a"], out["zzxu.dir|c0a"] = self.upper, self.lower, self.dir
        i = 0
        for c in CENTRES:
            for m in MULTS:
                for w in WIDTHS:
                    out[f"zzxu.lower|m.{c}.{m:g}.{w}"], out[f"zzxu.upper|m.{c}.{m:g}.{w}"] = self.max_lower[i], self.min_upper[i]
                    i += 1

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
