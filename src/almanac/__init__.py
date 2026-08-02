"""almanac -- alignment.

Everything that decides *when* an observation is valid and *whether* it could
have been traded:

* :class:`TradingCalendar` -- sessions and right-labelled intraday slot grids
  down to 3 seconds, for equities, index futures and night-session commodities.
* :class:`Universe` -- point-in-time constituent resolution that refuses
  membership tables without effective dates.
* :class:`Masks` -- suspension, price limits, one-word boards, newly listed;
  applied to features and labels together.
* :func:`adjust` -- backward corporate-action adjustment that keeps the
  unadjusted price alongside.
"""

from __future__ import annotations

from almanac.adjust import adjust, cumulative_factors, event_factors
from almanac.calendar import (
    CFFEX_BOND,
    CFFEX_INDEX,
    COMMODITY_NIGHT,
    EQUITY,
    SessionSpec,
    TradingCalendar,
    parse_freq,
)
from almanac.masks import FLAGS, Masks, apply_masks, build_masks, limit_pct, limit_prices
from almanac.universe import Membership, Universe, universe

__all__ = [
    "CFFEX_BOND",
    "CFFEX_INDEX",
    "COMMODITY_NIGHT",
    "EQUITY",
    "FLAGS",
    "Masks",
    "Membership",
    "SessionSpec",
    "TradingCalendar",
    "Universe",
    "adjust",
    "apply_masks",
    "build_masks",
    "cumulative_factors",
    "event_factors",
    "limit_pct",
    "limit_prices",
    "parse_freq",
    "universe",
]
