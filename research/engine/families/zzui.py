"""zzui: a 10-slot 10 s snapshot-price grid ending at 14:57:00 local (guess; the lead is a ~45 s price average)."""
from .base import GUESSED
from .zzxo import PriceGrid


class Zzui(PriceGrid):
    stem, status, n = "zzui", GUESSED, 10
    evidence = "guess: grid form; zzui.r fits a ~45 s snapshot-price average at 97% (R143), not built yet"
