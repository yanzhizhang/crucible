"""zzbd: log-price (100 s) and log-return (200 s) averages on the snapshot clock and on the trade clock."""
from .base import PINNED, Family


class Zzbd(Family):
    stem, status = "zzbd", PINNED
    evidence = "state: pEmaS / rEmaS / pEmaT / rEmaT equal PM's .jz (dcp_build)"
    emas = {"zzbd.pEmaS": ("snap", 100, "logp"), "zzbd.rEmaS": ("snap", 200, "logret"),
            "zzbd.pEmaT": ("trade", 100, "logp"), "zzbd.rEmaT": ("trade", 200, "logret")}
