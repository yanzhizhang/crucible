"""zzqj: 30 s trade-clock average of the running VWAP; volume and amount from 09:30."""
from .base import PINNED, Family


class Zzqj(Family):
    stem, status = "zzqj", PINNED
    evidence = "state: zzqj.r / v / tpv equal PM's .jz; the |all / |noclose / |loc keys are rejected variants kept for the record"
    emas = {"zzqj.r": ("trade", 30, "rvwap")}

    def dump(self, st, out):
        out["zzqj.v"], out["zzqj.tpv"] = st.vol, st.amt
        out["zzqj.v|all"], out["zzqj.tpv|all"] = st.vol_all, st.amt_all
        out["zzqj.v|noclose"], out["zzqj.tpv|noclose"] = st.vol_o, st.amt_o
        out["zzqj.v|loc"], out["zzqj.tpv|loc"] = st.vol_loc, st.amt_loc
