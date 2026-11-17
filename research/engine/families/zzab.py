"""zzab: 300 s snapshot-price average (key prefix zzab2)."""
from .base import PINNED, Family


class Zzab(Family):
    stem, status = "zzab", PINNED
    evidence = "state: zzab2.r equals PM's .jz (dcp_build)"
    emas = {"zzab2.r": ("snap", 300, "price")}
