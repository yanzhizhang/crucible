"""zzvx: the 30 s average of the day range (trEma*), its double average (trDEma*) and the average of their ratio (rEma*); day high / low.

trEma* = average of the snapshot's own high - low; trDEma* = the 30 s average of trEma's value (a double average: it lags a
range that only grows, so it is <= trEma and equal once settled); rEma* = the 30 s average of trEma / trDEma (1 when settled).
All three equal PM's day-end file on 1112 / 1113 (R162: 1.0000, 1.0000, 0.9967-1.0000). The D / O twins are the same number.
"""
from .base import PINNED, Family


class Zzvx(Family):
    stem, status = "zzvx", PINNED
    evidence = "state: high / low, trEma*, trDEma* (double average) and rEma* (average of trEma / trDEma) equal PM's .jz (R162); column zzvx.a open (lead: rEma itself -- shape fits the thresholds, R166; replacing it moves agreement with PM by < 0.007 either way, R167)"
    emas = {"zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "ema_tr"),
            "zzvx.trDEmaO": ("snap", 30, "ema_tr"), "zzvx.rEmaD": ("snap", 30, "r_ema_dema"), "zzvx.rEmaO": ("snap", 30, "r_ema_dema")}

    def dump(self, st, out):
        out["zzvx.high"], out["zzvx.low"] = st.high, st.low

    @staticmethod
    def features(kv, px):
        """Stand-in (wrong scale: PM's thresholds are a ratio just above 1, probably rEma itself)."""
        return {"zzvx.a": kv["zzvx.trDEmaD"] / px}
