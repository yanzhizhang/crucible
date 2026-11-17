"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is being worked out against PM's day-end file. Known: upper - mid <= 3 w and mid - lower <= 3 w
exactly (mid = (high + low) / 2 of the snapshot fields, w = a range average), dir = +1 on 83%; the band rule is "simple"
(the inactive band is recomputed every snapshot, the active one only tightens; the trend flips when the price crosses the
active band), and with it the inactive band is right and the active one is not (R162). A plain running extreme is not the
active band either (R163). PM's active band always sits beyond the day's low / high, which a band started fresh at the open
(when the range average is ~0) cannot do: so the state probably comes from YESTERDAY. Variants under test, all the simple rule:
"|b1" fresh width, fresh band; "|b2" width carried from yesterday (zzxu.a|c0), fresh band; "|b3" width and band carried;
"|b4" fresh width, band carried.
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0
BANDS = {"b1": ("zzxu.a", False), "b2": ("zzxu.a|c0", False), "b3": ("zzxu.a|c0", True), "b4": ("zzxu.a", True)}  # tag -> (width average, band carried)


class Band:
    """One simple-rule band."""

    __slots__ = ("dir", "lower", "upper", "width")

    def __init__(self, width: str, seed: tuple[float, float, float] | None) -> None:
        self.width = width
        self.upper, self.lower, self.dir = seed if seed and seed[0] == seed[0] and seed[1] == seed[1] and seed[2] == seed[2] else (math.nan, math.nan, 1.0)

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
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; upper / lower / dir under test (R163: not a running extreme; carried state next); columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl"), "zzxu.a|c0": ("snap", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.bands: dict[str, Band] | None = None

    def on_snap(self, st, t, s, lsod, amount):
        if self.bands is None:  # the tape (and yesterday's band on it) exists only now
            self.bands = {tag: Band(width, st.band_seed if carried else None) for tag, (width, carried) in BANDS.items()}
        px = st.snap_px
        if not (px == px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
        for b in self.bands.values():
            e = st.ema.get(b.width)
            w = e[0].value() if e else math.nan
            if w == w:
                b.step(mid + BAND_MULT * w, mid - BAND_MULT * w, px)

    def dump(self, st, out):
        out["zzxu.high"], out["zzxu.low"], out["zzxu.last"] = st.high, st.low, st.last
        if self.bands is None:
            self.bands = {tag: Band(width, st.band_seed if carried else None) for tag, (width, carried) in BANDS.items()}
        for tag, b in self.bands.items():
            out[f"zzxu.upper|{tag}"], out[f"zzxu.lower|{tag}"], out[f"zzxu.dir|{tag}"] = b.upper, b.lower, b.dir

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
