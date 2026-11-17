"""cdc: PM's common block; cdc.lp = the last trade price."""
from .base import PINNED, Family


class Cdc(Family):
    stem, status = "cdc", PINNED
    evidence = "state: cdc.lp equals PM's .jz"

    def dump(self, st, out):
        out["cdc.lp"] = st.last
