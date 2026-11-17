"""Families with PM state we cannot rebuild yet: no code, only what is known."""
from .base import OPEN, Family


class Zzcb(Family):
    stem, status = "zzcb", OPEN
    evidence = "open: zzcb.r fits no clock x half-life x input (R143 / R145); trees tie zzeu / zzef / zzel to it (R147)"


class Zztx(Family):
    stem, status = "zztx", OPEN
    evidence = "open: zztx.r fits no clock x half-life x input (R143 / R145)"
