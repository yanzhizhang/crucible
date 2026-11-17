"""zzba: 300 s averages, updated by snapshots and trades, of the snapshot price and of the trades' day range."""
from .base import PINNED, Family


class Zzba(Family):
    stem, status = "zzba", PINNED
    evidence = "state: zzba.p / zzba.r 1e-4 exact every day (1e-6 residual = feed interleaving); column zzba open (lead: ratio p / r)"
    emas = {"zzba.p": ("both", 300, "snap_px"), "zzba.r": ("both", 300, "hl_trade")}
