"""zzeb: 100 s mid average; zzeb.r is not explained."""
from .base import OPEN, PINNED, Family


class Zzeb(Family):
    stem, status = "zzeb", PINNED
    evidence = "state: zzeb.p equals PM's .jz; zzeb.r open (no clock x half-life x input fits, R143 / R145)"
    emas = {"zzeb.p": ("snap", 100, "mid")}
    keys = {"zzeb.r": OPEN}
