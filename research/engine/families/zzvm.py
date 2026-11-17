"""zzvm: volume and amount from 09:30; zzvm.c.re (a multi-day carried trade-price average) is not built."""
from .base import GUESSED, PINNED, Family


class Zzvm(Family):
    stem, status = "zzvm", PINNED
    evidence = "state: zzvm.c.v / tpv equal PM's .jz; zzvm.c.re lead only (carried trade-price average, 81-84%)"
    keys = {"zzvm.c.re": GUESSED}

    def dump(self, st, out):
        out["zzvm.c.v"], out["zzvm.c.tpv"] = st.vol, st.amt
