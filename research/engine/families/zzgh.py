"""zzgh: per 15 s slot the buy / sell amount imbalance as a code 0..4; the session's amount imbalance zzgh.nb."""
import math

from state import GH_EDGES, GH_EMPTY, SlotSum

from .base import GUESSED, Family


def imb(b: float, s: float) -> float:
    """(buy - sell) / (buy + sell)."""
    return (b - s) / (b + s) if b + s else math.nan


class Zzgh(Family):
    stem, status = "zzgh", GUESSED
    evidence = "guess: zzgh.c codes; zzgh.nb matches PM on 0.4% only (1116 day-end); the nb|* keys are variants under test"

    def __init__(self) -> None:
        self.c = SlotSum(948, 2)

    def on_trade(self, st, t, s, amt, side):
        if 34200 <= s < 53820:
            self.c.feed(t, amt if side > 0 else 0.0, amt if side < 0 else 0.0)

    def dump(self, st, out):
        out["zzgh.nb"] = imb(st.buy, st.sell)
        for tag, (b, s) in (("vol", st.nb_vol), ("cnt", st.nb_cnt), ("snappx", st.nb_snappx), ("ord", st.nb_ord), ("ordvol", st.nb_ordvol)):
            out[f"zzgh.nb|{tag}"] = imb(b, s)
        for k, (b, s) in enumerate(zip(*self.c.vals, strict=True)):
            if b + s > 0:
                x = (b - s) / (b + s)
                code = sum(1 for e in GH_EDGES if e <= x)  # searchsorted, right: a value on an edge goes up
            else:
                code = GH_EMPTY
            out[f"zzgh.c.{k}"] = float(code)
