"""zzvx: 30 s averages of the day range (four names, one value) and their update counts; day high / low."""
from .base import PINNED, Family


class Zzvx(Family):
    stem, status = "zzvx", PINNED
    evidence = "state: tr* / rEma* and high / low equal PM's .jz; column zzvx.a exact"
    emas = {"zzvx.trEmaD": ("snap", 30, "hl"), "zzvx.trEmaO": ("snap", 30, "hl"), "zzvx.trDEmaD": ("snap", 30, "hl"),
            "zzvx.trDEmaO": ("snap", 30, "hl"), "zzvx.rEmaD": ("snap", 30, "one"), "zzvx.rEmaO": ("snap", 30, "one")}

    def dump(self, st, out):
        out["zzvx.high"], out["zzvx.low"] = st.high, st.low

    @staticmethod
    def features(kv, px):
        return {"zzvx.a": kv["zzvx.trDEmaD"] / px}
