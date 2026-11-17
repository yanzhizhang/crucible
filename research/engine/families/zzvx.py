"""zzvx: 30 s averages of the day range and of a wider "true" range, and of their ratio; day high / low.

SUBJECT TO CHANGE: in PM's day-end file trEma* >= trDEma* and they differ on ~13-17% of stock-days, and rEma* is 1 with a
tail above 1 (R158 / R159); the rules below set all four tr* to the day range and rEma* to 1, right only where the two agree.
The "|pc", "|trade", "|tr" and "|tratio" rules are the candidates under test (scored against the key before the bar).
"""
from .base import GUESSED, Family


class Zzvx(Family):
    stem, status = "zzvx", GUESSED
    evidence = "state: high / low equal PM's .jz; trDEma* = day range on ~93%; trEma* and rEma* wrong where PM's differ (R158); column zzvx.a wrong scale (R148)"
    emas = {"zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
            "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one"),
            "zzvx.trEmaD|pc": ("snap", 30, "tr_pc"), "zzvx.trEmaD|trade": ("snap", 30, "hl_trade"), "zzvx.trEmaD|tro": ("snap", 30, "tro"),
            "zzvx.rEmaD|tr": ("snap", 30, "tr_ratio"), "zzvx.rEmaD|tratio": ("snap", 30, "trade_ratio")}

    def dump(self, st, out):
        out["zzvx.high"], out["zzvx.low"] = st.high, st.low

    @staticmethod
    def features(kv, px):
        """Stand-in (wrong scale: PM's thresholds are a ratio just above 1, probably rEma itself)."""
        return {"zzvx.a": kv["zzvx.trDEmaD"] / px}
