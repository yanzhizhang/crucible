"""zzup: fast (300 s) over slow (600 s) snapshot-price average."""
from .base import PINNED, Family


class Zzup(Family):
    stem, status = "zzup", PINNED
    evidence = "state: zzup.pf / zzup.ps equal PM's .jz (ps: 1e-6 residual ~1e-7 relative); column zzup exact"
    emas = {"zzup.pf": ("snap", 300, "price"), "zzup.ps": ("snap", 600, "price")}

    @staticmethod
    def features(kv, px):
        return {"zzup": kv["zzup.pf"] / kv["zzup.ps"]}
