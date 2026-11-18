"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is being worked out against PM's day-end file. Known: the band is mid -+ 3 w (mid = (high + low) / 2
of the snapshot fields, w = the range average zzxu.a CARRIED from yesterday, R164); the rule is "simple" (the inactive band is
recomputed every snapshot, the active one only tightens, the trend flips when the price crosses the active band); the
inactive band is then exact on ~85% of stocks, the active one never. PM rebuilds the active band within the day (R165) and
its level (median mid - 1.1 w in an up trend) is what a ratchet set in the first minutes would give, so the open is what
matters: which snapshots feed the width, and from when the band runs. Variants, all the simple rule from a fresh band:
  width c0 = carried, fed by every snapshot with a range; c1 = carried, fed from 09:30 only
  start a = first valid snapshot, p = 09:25:00, c = 09:30:00         tags "|c0a" "|c0p" "|c0c" "|c1a" "|c1p" "|c1c"
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0
STARTS = {"a": 0.0, "p": 33900.0, "c": 34200.0}
BANDS = {f"{w}{st}": (f"zzxu.a|{w}", sec) for w in ("c0", "c1") for st, sec in STARTS.items()}  # tag -> (width average, first exchange second)


class Band:
    """One simple-rule band."""

    __slots__ = ("dir", "lower", "start", "upper", "width")

    def __init__(self, width: str, start: float) -> None:
        self.width, self.start = width, start
        self.upper, self.lower, self.dir = math.nan, math.nan, 1.0

    def step(self, bu: float, bl: float, px: float) -> None:
        """Take one snapshot: the basic bands and the price."""
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


class Zzxu(Family):
    stem, status = "zzxu", PINNED
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; zzxu.a carries overnight (R164); upper / lower / dir under test; columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl"), "zzxu.a|c0": ("snap", 300, "hl"), "zzxu.a|c1": ("snap_o930", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.bands = {tag: Band(width, start) for tag, (width, start) in BANDS.items()}

    def on_snap(self, st, t, s, lsod, amount):
        px = st.snap_px
        if not (px == px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
        for b in self.bands.values():
            if s < b.start:
                continue
            e = st.ema.get(b.width)
            w = e[0].value() if e else math.nan
            if w == w:
                b.step(mid + BAND_MULT * w, mid - BAND_MULT * w, px)

    def dump(self, st, out):
        out["zzxu.high"], out["zzxu.low"], out["zzxu.last"] = st.high, st.low, st.last
        for tag, b in self.bands.items():
            out[f"zzxu.upper|{tag}"], out[f"zzxu.lower|{tag}"], out[f"zzxu.dir|{tag}"] = b.upper, b.lower, b.dir

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
