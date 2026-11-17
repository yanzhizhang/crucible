"""zzjb: per 15 s slot, code 0 when the slot's snapshot price is 3.04% or more below the previous close, else 4."""
import math

from state import JB_DROP, LastBefore

from .base import GUESSED, Family


class Zzjb(Family):
    stem, status = "zzjb", GUESSED
    evidence = "guess: zzjb.c two-valued code cut from the price grid"

    def __init__(self) -> None:
        self.px = LastBefore(948, math.nan)

    def on_snap(self, st, t, s, lsod, amount):
        self.px.feed(t, st.snap_px)

    def dump(self, st, out):
        for k, px in enumerate(self.px.close()):
            out[f"zzjb.c.{k}"] = 0.0 if px == px and st.pc == st.pc and px <= st.pc * (1 - JB_DROP) else 4.0
