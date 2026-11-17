"""zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).

SUBJECT TO CHANGE: the band is being worked out against PM's day-end file. Known (R159): upper - mid <= 3 w and
mid - lower <= 3 w exactly (mid = (high + low) / 2 of the snapshot fields, w = the range average), dir = +1 on 83%.
R160: the first rule ("|st1") gets the upper band right on 30-50% of stocks to 1e-8, so the band formula is right and
the ratchet / flip rule is not. Several rules run side by side and are scored against PM's zzxu.upper / lower / dir.
"""
import math

from .base import GUESSED, PINNED, Family

BAND_MULT = 3.0
# tag -> (rule, first direction, price): simple = the inactive band is recomputed, the active one only tightens;
# classic = each band tightens unless the previous price was beyond it; classic_cur = the same with the current price
BANDS = {"st1": ("simple", 1.0, "snap"), "st2": ("classic", 1.0, "snap"), "st3": ("classic_cur", 1.0, "snap"),
         "st4": ("classic", -1.0, "snap"), "st5": ("classic", 1.0, "mid"), "st6": ("simple", 1.0, "mid")}


class Band:
    """One band rule's state."""

    __slots__ = ("dir", "lower", "prev", "px_kind", "rule", "upper")

    def __init__(self, rule: str, first: float, px_kind: str) -> None:
        self.rule, self.dir, self.px_kind = rule, first, px_kind
        self.upper = self.lower = self.prev = math.nan

    def step(self, bu: float, bl: float, px: float) -> None:
        """Take one snapshot: the basic bands and the price."""
        if self.upper != self.upper:
            self.upper, self.lower, self.prev = bu, bl, px
            return
        if self.rule == "simple":
            if self.dir > 0:
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
        else:
            ref = px if self.rule == "classic_cur" else self.prev
            pu, pl_ = self.upper, self.lower
            if bu < pu or ref > pu:
                self.upper = bu
            if bl > pl_ or ref < pl_:
                self.lower = bl
            if self.dir > 0:
                if px < self.lower:
                    self.dir = -1.0
            elif px > self.upper:
                self.dir = 1.0
        self.prev = px


class Zzxu(Family):
    stem, status = "zzxu", PINNED
    evidence = "state: zzxu.p / zzxu.a / high / low / last equal PM's .jz; upper / lower / dir under test (R160); columns zzxu.a, zzxu.b wrong scale (R148)"
    emas = {"zzxu.p": ("snap", 300, "price"), "zzxu.a": ("snap", 300, "hl")}
    keys = {"zzxu.upper": GUESSED, "zzxu.lower": GUESSED, "zzxu.dir": GUESSED}

    def __init__(self) -> None:
        self.bands = {tag: Band(*spec) for tag, spec in BANDS.items()}

    def on_snap(self, st, t, s, lsod, amount):
        e = st.ema.get("zzxu.a")
        w = e[0].value() if e else math.nan
        if not (w == w and st.snap_px == st.snap_px and st.hi_f == st.hi_f and st.lo_f == st.lo_f):
            return
        mid = (st.hi_f + st.lo_f) / 2
        bu, bl = mid + BAND_MULT * w, mid - BAND_MULT * w
        for b in self.bands.values():
            b.step(bu, bl, st.snap_px if b.px_kind == "snap" else st.mid)

    def dump(self, st, out):
        out["zzxu.high"], out["zzxu.low"], out["zzxu.last"] = st.high, st.low, st.last
        for tag, b in self.bands.items():
            out[f"zzxu.upper|{tag}"], out[f"zzxu.lower|{tag}"], out[f"zzxu.dir|{tag}"] = b.upper, b.lower, b.dir

    @staticmethod
    def features(kv, px):
        """Stand-ins (wrong scale: PM's thresholds point at ratios of the band keys)."""
        return {"zzxu.a": kv["zzxu.a"] / kv["zzxu.low"], "zzxu.b": (px - kv["zzxu.a"]) / kv["zzxu.last"]}
