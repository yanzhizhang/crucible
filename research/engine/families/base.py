r"""One class per PM factor family: what the family owns in the engine and in the model.

A family (``zzup``, ``zzgh`` ...) is PM's unit: its day-end ``.jz`` keys share the prefix and its model columns share the
name. A ``Family`` subclass holds, in one place:

* ``status`` -- ``pinned`` (matches PM's own state / column), ``guessed`` (a stand-in, SUBJECT TO CHANGE) or ``open``
  (not built), with one ``evidence`` line saying what it was checked against;
* ``emas`` -- its decayed averages: rule name -> (clock, half-life s, input), the rows of the engine's rule table;
* its own running state beyond the shared tape (time grids) and the hooks that update it;
* ``dump`` -- its keys under PM's names;
* ``columns`` -- its model columns: column -> (status, formula); exact columns are also code in ``features``.

The shared tape (``state.Stock``) holds the raw running values several families read: last snapshot / trade price, day
high / low / open, volume and amount, best bid / ask, the snapshot's own high / low. Hooks run after the tape took the event.
"""

from __future__ import annotations

PINNED, GUESSED, OPEN, WRONG = "pinned", "guessed", "open", "wrong"


class Family:
    """Base: no averages, no extra state, no keys."""

    stem = ""
    status = GUESSED
    evidence = ""
    emas: dict[str, tuple[str, float, str]] = {}
    keys: dict[str, str] = {}  # key (or "prefix.*") -> status, where it differs from the family's
    columns: dict[str, tuple[str, str]] = {}  # filled from the baseline table by columns.attach

    def on_snap(self, st, t: float, s: float, lsod: float, amount: float) -> None:
        """A snapshot was applied to the tape."""

    def on_trade(self, st, t: float, s: float, amt: float, side: float) -> None:
        """A trade was applied to the tape."""

    def dump(self, st, out: dict[str, float]) -> None:
        """Write the family's non-average keys under PM's names."""

    @staticmethod
    def features(kv: dict[str, float], px: float) -> dict[str, float]:
        """The family's exact model columns from its state values (averages as sum / wgt) and the last price."""
        return {}
