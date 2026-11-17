"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is a first guess from PM's day-end file (R159): upper - mid <= 3 w and mid - lower <= 3 w
exactly (mid = (high + low) / 2, w = the range average), dir = +1 on 83% -- a trend band with multiplier 3. Rule tried
here: in an up trend the lower band only rises and the upper is recomputed; in a down trend the mirror; the trend flips
when the snapshot price crosses the active band. Dumped as "zzxu.upper|st" ... and scored against PM's zzxu.upper / lower / dir.
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0


class Zzxu(Family):
    stem, status = "zzxu", PINNED
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; upper / lower / dir a guess; columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.upper = self.lower = math.nan
        self.dir = 1.0

    def on_snap(self, st, t, s, lsod, amount):
        e = st.ema.get("zzxu.a")
        w = e[0].value() if e else math.nan
        px = st.snap_px
        if not (w == w and px == px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
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
        out["zzxu.upper|st"], out["zzxu.lower|st"], out["zzxu.dir|st"] = self.upper, self.lower, self.dir

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
