"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is being worked out against PM's day-end file. Known: upper - mid <= 3 w and mid - lower <= 3 w
exactly (mid = (high + low) / 2 of the snapshot fields, w = a range average), dir = +1 on 83%. R162: with the simple rule
("|st1": the inactive band is recomputed every snapshot, the active one only tightens) the INACTIVE band is right (upper
on 85% of the stocks where both say up) and the active, ratcheted one never is. So the ratchet's input is what is wrong:
"|m.<centre>.<width>.<start>" are running extremes of centre -+ 3 x width (the highest lower band / the lowest upper band
since the start), for centre mid / px / p, width a (zzxu.a) / tr (zzvx.trEma) / de (zzvx.trDEma), start a (first valid
snapshot) / c (09:30). A stock that never flipped has its active band equal to one of these.
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0
CENTRES = ("mid", "px", "p")
WIDTHS = ("a", "tr", "de")
STARTS = ("a", "c")


class Zzxu(Family):
    stem, status = "zzxu", PINNED
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; upper / lower / dir under test (R162: inactive band right, ratchet wrong); columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.upper = self.lower = math.nan
        self.dir = 1.0
        n = len(CENTRES) * len(WIDTHS) * len(STARTS)
        self.max_lower, self.min_upper = [math.nan] * n, [math.nan] * n

    def on_snap(self, st, t, s, lsod, amount):
        val = lambda k: st.ema[k][0].value() if k in st.ema else math.nan  # noqa: E731
        ws = (val("zzxu.a"), val("zzvx.trEmaD"), val("zzvx.trDEmaD"))
        px = st.snap_px
        if not (px == px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
        cs = (mid, px, val("zzxu.p"))
        i = 0
        for c in cs:
            for w in ws:
                for start in STARTS:
                    if c == c and w == w and (start == "a" or s >= 34200):
                        lo, up = c - BAND_MULT * w, c + BAND_MULT * w
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
        out["zzxu.upper|st1"], out["zzxu.lower|st1"], out["zzxu.dir|st1"] = self.upper, self.lower, self.dir
        i = 0
        for c in CENTRES:
            for w in WIDTHS:
                for start in STARTS:
                    out[f"zzxu.lower|m.{c}.{w}.{start}"], out[f"zzxu.upper|m.{c}.{w}.{start}"] = self.max_lower[i], self.min_upper[i]
                    i += 1

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
