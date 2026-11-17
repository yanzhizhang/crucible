"""zzxo: last trade price; a 30-slot 10 s snapshot-price grid ending at 14:57:00 local."""
import math

from .base import GUESSED, PINNED, Family


class PriceGrid(Family):
    """n slots: slot k = the last snapshot price received at or before 14:57:00 - (n - 1 - k) x 10 s, local time."""

    n = 0

    def __init__(self) -> None:
        self.g = [math.nan] * self.n

    def on_snap(self, st, t, s, lsod, amount):
        if st.snap_px == st.snap_px:
            for k in range(self.n):
                if lsod <= 53820 - (self.n - 1 - k) * 10:
                    self.g[k] = st.snap_px

    def dump(self, st, out):
        for k, v in enumerate(self.g):
            out[f"{self.stem}.r.{k}"] = v


class Zzxo(PriceGrid):
    stem, status, n = "zzxo", PINNED, 30
    evidence = "state: zzxo.last equals PM's .jz; the zzxo.r grid is a guess"
    keys = {"zzxo.r.*": GUESSED}

    def dump(self, st, out):
        out["zzxo.last"] = st.last
        super().dump(st, out)
