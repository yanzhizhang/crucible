"""zzvx: 30 s averages of the day range (trEma*), of a narrower range (trDEma*) and of their ratio (rEma*); day high / low.

SUBJECT TO CHANGE: trEma* = the snapshot's own high - low, exact against PM (R160). trDEma* is narrower on ~9-13% of
stock-days and rEma* is above 1 there; the rules below still set trDEma* to the day range and rEma* to 1. Candidates
under test (scored against the key before the bar): the range of the snapshot PRICES in the continuous session ("|sp"),
of all snapshot prices ("|spall"), of the continuous session's trades ("|cont"), and the day range over each ("|rsp" ...).
"""
from .base import GUESSED, Family


class Zzvx(Family):
    stem, status = "zzvx", GUESSED
    evidence = "state: high / low and trEma* equal PM's .jz (R160: 1.0000); trDEma* 0.91-0.92 and rEma* 0.85-0.89 with the day range / 1; column zzvx.a wrong scale (R148)"
    emas = {"zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
            "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one"),
            "zzvx.trDEmaD|sp": ("snap", 30, "hl_sp"), "zzvx.trDEmaD|spall": ("snap", 30, "hl_spall"), "zzvx.trDEmaD|cont": ("snap", 30, "hl_cont"),
            "zzvx.rEmaD|rsp": ("snap", 30, "r_sp"), "zzvx.rEmaD|rspall": ("snap", 30, "r_spall"), "zzvx.rEmaD|rcont": ("snap", 30, "r_cont")}

    def dump(self, st, out):
        out["zzvx.high"], out["zzvx.low"] = st.high, st.low

    @staticmethod
    def features(kv, px):
        """Stand-in (wrong scale: PM's thresholds are a ratio just above 1, probably rEma itself)."""
        return {"zzvx.a": kv["zzvx.trDEmaD"] / px}
