"""zzmh: the highest snapshot price from 09:30:00 on (no end: the closing snapshot counts) and the 10 s average of (2 x drawdown from it)^2 (cfg.json term[57] zzmh hl=10)."""
from .base import PINNED, Family


class Zzmh(Family):
    stem, status = "zzmh", PINNED
    evidence = "state: zzmh.hm and zzmh.r equal PM's .jz (R124b: PM / ((hm - p) / hm)^2 = 4.0000 on 294 stocks; hm without an end time: 1.0 on 1118 / 1119 / 1120, R193)"
    emas = {"zzmh.r": ("snap", 10, "mh_dd2")}  # hl=10 (cfg); the old 1 s gave PM/ours wgt = 8/7 on SH (r232)

    def dump(self, st, out):
        out["zzmh.hm"] = st.hm
