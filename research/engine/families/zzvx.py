"""zzvx: 30 s averages of the day range (trEma*), a second, lagging series (trDEma*) and a ratio (rEma*); day high / low.

SUBJECT TO CHANGE: trEma* = the snapshot's own high - low, exact against PM (R160). trDEma* is below it on ~9-13% of
stock-days, mostly by 1e-6..1e-4 relative, with the same weight and clock -- the footprint of a DOUBLE average (the
average of the average), which lags a range that only grows. Under test ("|tag" rows are scored against the key before
the bar): "|dema" = the 30 s average of trEma's value; rEma as the average of range / trEma ("|rte"), range / the double
average ("|rtd"), trEma / the double average ("|red"). The untagged trDEma* / rEma* rules are the old ones (range, 1).
"""
from .base import GUESSED, Family


class Zzvx(Family):
    stem, status = "zzvx", GUESSED
    evidence = "state: high / low and trEma* equal PM's .jz (R160: 1.0000); trDEma* 0.91-0.92 and rEma* 0.85-0.89 with the day range / 1; column zzvx.a wrong scale (R148)"
    emas = {"zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
            "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one"),
            "zzvx.trDEmaD|dema": ("snap", 30, "ema_tr"), "zzvx.rEmaD|rte": ("snap", 30, "r_tr_ema"), "zzvx.rEmaD|rtd": ("snap", 30, "r_tr_dema"),
            "zzvx.rEmaD|red": ("snap", 30, "r_ema_dema")}

    def dump(self, st, out):
        out["zzvx.high"], out["zzvx.low"] = st.high, st.low

    @staticmethod
    def features(kv, px):
        """Stand-in (wrong scale: PM's thresholds are a ratio just above 1, probably rEma itself)."""
        return {"zzvx.a": kv["zzvx.trDEmaD"] / px}
