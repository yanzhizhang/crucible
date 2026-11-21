"""zzmh: the continuous session's highest snapshot price and the 1 s average of (2 x drawdown from it)^2."""
from .base import PINNED, Family


class Zzmh(Family):
    stem, status = "zzmh", PINNED
    evidence = "state: zzmh.hm and zzmh.r equal PM's .jz (R124b: PM / ((hm - p) / hm)^2 = 4.0000 on 294 stocks)"
    emas = {"zzmh.r": ("snap", 1, "mh_dd2")}

    def dump(self, st, out):
        out["zzmh.hm"] = st.hm
        out["zzmh.hm|le1500"], out["zzmh.hm|all"] = st.hm_c, st.sp_hi
        out["zzmh.hm|ge930"], out["zzmh.hm|ge930t"] = st.hm_o, st.hm_ot
        out["zzmh.hm|last"] = max(st.hm, st.last) if st.hm == st.hm and st.last == st.last else st.hm
