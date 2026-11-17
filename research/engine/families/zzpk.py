"""zzpk: 300 s snapshot-price average, day high / low / open."""
from .base import PINNED, Family


class Zzpk(Family):
    stem, status = "zzpk", PINNED
    evidence = "state: zzpk.p and high / low / open equal PM's .jz (dcp_build, R93 / 1116 day-end)"
    emas = {"zzpk.p": ("snap", 300, "price")}

    def dump(self, st, out):
        out["zzpk.high"], out["zzpk.low"], out["zzpk.open"] = st.high, st.low, st.open
