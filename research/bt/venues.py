"""Venue / instrument rules for the backtest: one entry per (exchange, instrument class).

SSE and SZSE publish their order flow differently, and the rules that matter for a book replay
and a queue simulation differ by venue *and* by instrument class (stock, ETF, convertible bond,
later futures / options). Everything venue-specific lives here, behind one small interface, so
adding a class means adding an entry -- not sprinkling ``if exchange == ...`` through the engine.

Each :class:`Rules` gives

* the **event builder** (raw feitu streams -> the engine's common encoding, see :mod:`bt.events`),
* the **engine dispatch code** selecting the venue's event handler inside the numba kernel,
* instrument economics: tick size, lot, T+0/T+1, stamp duty, price-limit source,
* the **auction clearing tie rule**, recorded for when our own orders take part in an auction.

Replay semantics per venue (measured; see ``docs/BACKTEST.md``)
---------------------------------------------------------------

SSE (XSHG) stocks
    * order key ``order_id``; cancels ride the **order** stream (``update_type == 2``);
    * a marketable order is published only *after* matching, as its resting remainder -- its
      trades reference an aggressor that is never added (harmless: only the passive side matters);
    * adds are always priced (market-to-limit remainders carry the limit they rest at).

SZSE (XSHE) stocks
    * order key ``seq_id``; cancels ride the **trade** stream (``trade_type == 2``);
    * every order is published at entry, including marketable ones (the book is briefly crossed
      until their trades arrive);
    * ``orderType`` 3 = own-side best: priced at the own side's best when it arrives;
    * ``orderType`` 1 = market, which merges four official types (counterparty-best with the
      remainder turned into a limit, FAK, FOK, best-5). A pre-pass (:func:`bt.events.
      szse_market_tags`, after the prototype's ``classify_szse_core``) tags each one from its own
      profile; the replay holds it out of the book until its burst ends, then rests the remainder
      at its fill price -- except FAK / FOK tags, which never rest;
    * trade aggressor = the leg with the larger sequence number.

SSE orders are replayed exactly as published: the records carry nothing to deduce a FAK / FOK
from, and the marketable part never appears in the book anyway.

Continuous trading is 09:30-11:30 and 13:00-14:57 on both venues (:data:`CONTINUOUS`). Outside
it the replay book holds auction orders and is crossed, so the engine never matches against it:
entries are skipped, exits wait for the next window.

Call auction (both venues): the replay uses the published auction trades, so no clearing is
computed. The first three clearing criteria (max volume, then the usual imbalance rules) are the
same on both venues; only the final tie-break differs (per
``sht-demo/prototype/simulation/me_types.hpp``, ``auction_price_rule``):

* SZSE ``nearest_ref`` -- scan the whole tick grid for the minimum-imbalance range, then take the
  tick nearest the reference price (open auction: previous close; close / intraday auction: last
  trade). The clearing price may be a tick nobody submitted.
* SSE ``midpoint`` -- scan submitted prices only (never an empty tick); if two or more tie, take
  the midpoint of the highest and lowest, rounded up to the tick.

This becomes relevant only when simulated orders join an auction (e.g. restoring the base in
the close auction); until then the published auction trades are the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

VENUE_SSE, VENUE_SZSE = 0, 1


@dataclass(frozen=True)
class Rules:
    """Everything the backtest needs to know about one (exchange, instrument class)."""

    exchange: str
    klass: str
    engine_code: int
    tick_size: float
    lot: int
    t_plus: int
    stamp_duty: bool
    auction_tie_rule: str
    #: how far after its (floored) timestamp a snapshot can reflect the book (measured: SZSE
    #: ~2 s on busy stocks, SSE within 1 s); used to align the replay audit
    snapshot_lag_s: float = 1.0
    implemented: bool = True
    todo: tuple[str, ...] = field(default_factory=tuple)


RULES: dict[tuple[str, str], Rules] = {
    ("XSHG", "stock"): Rules("XSHG", "stock", VENUE_SSE, 0.01, 100, 1, True, "midpoint"),
    ("XSHE", "stock"): Rules("XSHE", "stock", VENUE_SZSE, 0.01, 100, 1, True, "nearest_ref", 3.0),
    ("XSHG", "etf"): Rules(
        "XSHG", "etf", VENUE_SSE, 0.001, 100, 1, False, "midpoint", implemented=False,
        todo=("0.001 tick", "T+0 for bond/gold/cross-border ETFs", "no stamp duty",
              "creation/redemption records in the order stream")),
    ("XSHE", "etf"): Rules(
        "XSHE", "etf", VENUE_SZSE, 0.001, 100, 1, False, "nearest_ref", implemented=False,
        todo=("0.001 tick", "T+0 for some ETFs", "no stamp duty")),
    ("XSHG", "cb"): Rules(
        "XSHG", "cb", VENUE_SSE, 0.001, 10, 0, False, "midpoint", implemented=False,
        todo=("0.001 tick", "T+0", "lot = 10 bonds", "+-20% limit and 30%/57% halts",
              "no stamp duty")),
    ("XSHE", "cb"): Rules(
        "XSHE", "cb", VENUE_SZSE, 0.001, 10, 0, False, "nearest_ref", implemented=False,
        todo=("0.001 tick", "T+0", "lot = 10 bonds", "+-20% limit", "no stamp duty")),
}

_STOCK = {"XSHG": ("600", "601", "603", "605", "688", "689"),
          "XSHE": ("000", "001", "002", "003", "300", "301", "302")}
_ETF = {"XSHG": ("51", "52", "56", "58"), "XSHE": ("15", "16")}
_CB = {"XSHG": ("110", "111", "113", "118"), "XSHE": ("123", "127", "128")}


#: continuous trading, Beijing time (A-share stocks, both venues; the close auction starts 14:57)
CONTINUOUS = (("09:30:00", "11:30:00"), ("13:00:00", "14:57:00"))


def continuous_windows(date: str) -> np.ndarray:
    """``(k, 2)`` int64 ``[start, end)`` windows of continuous trading on the exchange-ts clock (UTC ns)."""
    d = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    off = np.timedelta64(8, "h")
    return np.array([[(np.datetime64(f"{d}T{a}", "ns") - off).astype(np.int64),
                      (np.datetime64(f"{d}T{b}", "ns") - off).astype(np.int64)] for a, b in CONTINUOUS],
                    np.int64)


def classify(symbol: str, exchange: str) -> str:
    """Instrument class from the code range (the vendor's symbol type flag is not reliable)."""
    for klass, table in (("stock", _STOCK), ("etf", _ETF), ("cb", _CB)):
        if symbol.startswith(table.get(exchange, ())):
            return klass
    return "other"


def rules_for(symbol: str, exchange: str) -> Rules:
    """Rules for one instrument; refuses classes that are registered but not implemented."""
    klass = classify(symbol, exchange)
    r = RULES.get((exchange, klass))
    if r is None:
        raise NotImplementedError(f"no backtest rules for {exchange} {klass} ({symbol})")
    if not r.implemented:
        raise NotImplementedError(
            f"{exchange} {klass} ({symbol}) is registered but not implemented yet; needs: "
            + "; ".join(r.todo))
    return r
