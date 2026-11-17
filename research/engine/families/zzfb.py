"""zzfb: the snapshot's total amount on a 711-slot grid (last value before each slot's end)."""
from state import LastBefore

from .base import GUESSED, Family


class Zzfb(Family):
    stem, status = "zzfb", GUESSED
    evidence = "guess: zzfb.c = last snapshot amount per 20 s slot"

    def __init__(self) -> None:
        self.c = LastBefore(711)

    def on_snap(self, st, t, s, lsod, amount):
        self.c.feed(t, amount)

    def dump(self, st, out):
        for k, v in enumerate(self.c.close()):
            out[f"zzfb.c.{k}"] = v
