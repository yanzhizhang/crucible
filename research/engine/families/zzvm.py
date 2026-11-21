"""zzvm: volume and amount from 09:30, and zzvm.c.re = the 300 s trade-clock average of the running VWAP from 09:30."""
from .base import PINNED, Family


class Zzvm(Family):
    stem, status = "zzvm", PINNED
    evidence = "state: zzvm.c.v / tpv equal PM's .jz; zzvm.c.re = trade clock, half 300 s, running VWAP from 09:30, no carry: value exact on 1.000 / 1.000 / 0.993 (dysl2, 1118-1120, R200-R201)"
    emas = {"zzvm.c.re": ("trade", 300, "rvwap")}

    def dump(self, st, out):
        out["zzvm.c.v"], out["zzvm.c.tpv"] = st.vol, st.amt
