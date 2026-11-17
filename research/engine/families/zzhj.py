"""zzhj: the mid 30 s before each quarter-hour end (16 samples, the last at 14:56:30)."""
import math

from state import HJ_SAMPLE

from .base import GUESSED, OPEN, Family


class Zzhj(Family):
    stem, status = "zzhj", GUESSED
    evidence = "guess: zzhj.c quarter-hour mids; zzhj.p open (weight 14.93 ~ a 30 s snapshot clock, amount-like values, 58-61%)"
    keys = {"zzhj.p": OPEN}

    def __init__(self) -> None:
        self.c = [math.nan] * 16

    def on_snap(self, st, t, s, lsod, amount):
        for k, ts in enumerate(HJ_SAMPLE):  # last snapshot received at or before each sample
            if t <= ts:
                self.c[k] = st.mid

    def dump(self, st, out):
        for k, v in enumerate(self.c):
            out[f"zzhj.c.{k}"] = v
