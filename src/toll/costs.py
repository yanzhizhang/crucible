"""Transaction costs for A-share equities.

Costs are decomposed by component rather than collapsed into one number,
because the components behave completely differently as you scale. Stamp duty
and commission are linear in notional and never go away. Market impact grows
with the *square root* of participation, which means doubling size raises the
impact cost per share by only ~41% -- but it is also the term that eventually
kills a strategy, and you cannot see which one is binding from a single "cost
bps" figure.

The tearsheet in :mod:`herald` attributes PnL loss to spread vs impact vs tax
directly off this decomposition.

A-share specifics that catch people out:

* **Stamp duty is sell-side only.** Applying it to both legs overstates
  round-trip cost by half.
* **Lot size is 100 shares** on buys. Sells may be odd-lot (the remainder of a
  position), which is why rounding is side-aware.
* **Minimum commission** is typically 5 CNY per order, which dominates for
  small tickets -- a 3000 CNY trade pays ~17bp in commission alone, not 2.5bp.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import polars as pl

from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = [
    "CostModel",
    "STAMP_DUTY_CUT_DATE",
    "stamp_duty_for",
    "round_lots",
    "square_root_impact",
]

STAMP_DUTY_CUT_DATE = dt.date(2023, 8, 28)
"""Date the MOF halved A-share stamp duty from 0.10% to 0.05%."""


def stamp_duty_for(day: dt.date | str) -> float:
    """Historical sell-side stamp duty rate in effect on ``day``.

    Backtests spanning 2023 must use this rather than a constant. A single rate
    across the cut date misprices every trade on one side of it, and for a
    high-turnover strategy stamp duty is often the largest single cost line.
    """
    d = day if isinstance(day, dt.date) else dt.date.fromisoformat(str(day))
    return 0.0005 if d >= STAMP_DUTY_CUT_DATE else 0.0010


@dataclass(frozen=True)
class CostModel:
    """A-share cost model with an explicit component breakdown.

    Parameters
    ----------
    stamp_duty:
        Sell-side only. Defaults to the pre-2023 0.10%; use
        :func:`stamp_duty_for` for a sample spanning the cut.
    commission:
        Broker commission per side, as a fraction of notional.
    min_commission:
        Per-order floor in CNY. Dominates small tickets.
    transfer_fee:
        Exchange transfer fee (过户费), charged both sides since 2022.
    impact_alpha:
        Coefficient in the square-root impact law. ~0.5 is the widely
        replicated value; calibrate against your own fills before trusting it
        at size.
    permanent_fraction:
        Share of impact that does not decay. The permanent part moves the price
        against everyone, the temporary part reverts after you stop trading.
        Splitting them matters for multi-slice execution, where only the
        permanent component compounds across slices.
    lot_size:
        Board lot for buys. 100 shares on all A-share boards.
    """

    stamp_duty: float = 0.0010
    commission: float = 0.00025
    min_commission: float = 5.0
    transfer_fee: float = 0.00001
    impact_alpha: float = 0.5
    permanent_fraction: float = 1.0 / 3.0
    lot_size: int = 100

    def __post_init__(self) -> None:
        for name in ("stamp_duty", "commission", "transfer_fee", "impact_alpha"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if not 0.0 <= self.permanent_fraction <= 1.0:
            raise ValueError(
                f"permanent_fraction must be in [0, 1], got {self.permanent_fraction}"
            )
        if self.lot_size < 1:
            raise ValueError(f"lot_size must be >= 1, got {self.lot_size}")

    # -- impact ----------------------------------------------------------

    def impact(self, shares: np.ndarray, adv: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Square-root impact as a fraction of price.

        ``impact = alpha * sigma * sqrt(Q / V)`` where ``Q`` is shares traded
        and ``V`` is average daily volume in shares. ``sigma`` is daily return
        volatility as a fraction.

        Participation above 100% of ADV is not extrapolated blindly -- it is
        clipped at 1.0 with the square root still applied, because the model has
        no validity there and the honest answer is "this is untradeable", not a
        number.
        """
        return square_root_impact(shares, adv, sigma, alpha=self.impact_alpha)

    def deadband_threshold(
        self,
        price: float,
        shares: float,
        *,
        adv: float,
        sigma: float,
    ) -> float:
        """Minimum expected return that justifies a round trip, as a fraction.

        Derived analytically from the costs actually incurred rather than set
        as a constant. A fixed 20bp deadband is wrong in both directions at
        once: too wide for a liquid megacap, far too narrow for a small cap
        where you are 5% of ADV.

        Includes both legs of commission and transfer fee, one leg of stamp
        duty, and impact on entry and exit.

        Returns
        -------
        Fractional return threshold. Signals weaker than this lose money after
        costs no matter how good the IC looks.
        """
        if price <= 0:
            raise ValueError(f"price must be > 0, got {price}")
        notional = price * abs(shares)
        if notional <= 0:
            return 0.0

        comm_rate = max(self.commission, self.min_commission / notional)
        explicit = 2.0 * comm_rate + 2.0 * self.transfer_fee + self.stamp_duty
        one_way = float(
            self.impact(np.array([abs(shares)]), np.array([adv]), np.array([sigma]))[0]
        )
        return explicit + 2.0 * one_way

    # -- application -----------------------------------------------------

    def apply(
        self,
        trades: Frame,
        *,
        shares_col: str = "shares",
        price_col: str = "price",
        adv_col: str = "adv",
        sigma_col: str = "sigma",
        round_to_lots: bool = True,
    ) -> Frame:
        """Cost every trade, decomposed by component.

        Parameters
        ----------
        trades:
            Long frame with ``ts``, ``symbol``, signed ``shares`` (positive =
            buy), ``price``, and optionally ``adv`` and ``sigma`` for impact.
        round_to_lots:
            Round buys down to whole lots. Sells are left alone so a position
            can be fully exited; rounding a sell up would create a short.

        Returns
        -------
        The trades with added columns:
        ``shares_filled``, ``notional``, ``commission``, ``stamp_duty``,
        ``transfer_fee``, ``impact_temporary``, ``impact_permanent``,
        ``cost_total``, ``cost_bps``.

        Notes
        -----
        Rounding down on buys means the requested and filled quantities differ.
        Downstream position accounting must use ``shares_filled``; using the
        request is a slow-leaking discrepancy that shows up as unexplained
        tracking error.
        """
        want = flavor(trades)
        lf = to_polars(trades)
        require_columns(lf, (TS, SYMBOL, shares_col, price_col), where="CostModel.apply")

        shares = lf[shares_col].to_numpy().astype(float)
        price = lf[price_col].to_numpy().astype(float)

        filled = round_lots(shares, self.lot_size) if round_to_lots else shares
        qty = np.abs(filled)
        notional = qty * price

        with np.errstate(divide="ignore", invalid="ignore"):
            comm = np.where(notional > 0, np.maximum(notional * self.commission, self.min_commission), 0.0)
        stamp = np.where(filled < 0, notional * self.stamp_duty, 0.0)
        transfer = notional * self.transfer_fee

        if adv_col in lf.columns and sigma_col in lf.columns:
            adv = lf[adv_col].to_numpy().astype(float)
            sigma = lf[sigma_col].to_numpy().astype(float)
            frac = self.impact(qty, adv, sigma)
        else:
            frac = np.zeros_like(qty)

        impact_cost = frac * notional
        perm = impact_cost * self.permanent_fraction
        temp = impact_cost - perm
        total = comm + stamp + transfer + impact_cost

        with np.errstate(divide="ignore", invalid="ignore"):
            bps = np.where(notional > 0, total / notional * 1e4, 0.0)

        out = lf.with_columns(
            shares_filled=pl.Series("shares_filled", filled),
            notional=pl.Series("notional", notional),
            commission=pl.Series("commission", comm),
            stamp_duty=pl.Series("stamp_duty", stamp),
            transfer_fee=pl.Series("transfer_fee", transfer),
            impact_temporary=pl.Series("impact_temporary", temp),
            impact_permanent=pl.Series("impact_permanent", perm),
            cost_total=pl.Series("cost_total", total),
            cost_bps=pl.Series("cost_bps", bps),
        )
        return restore(out, want)


def square_root_impact(
    shares: np.ndarray,
    adv: np.ndarray,
    sigma: np.ndarray,
    *,
    alpha: float = 0.5,
    max_participation: float = 1.0,
) -> np.ndarray:
    """Square-root market impact as a fraction of price.

    ``alpha * sigma * sqrt(min(Q / V, max_participation))``.

    Zero or missing ADV yields zero impact rather than infinity -- a name with
    no volume should be caught by :mod:`almanac.masks` as untradeable, and
    producing ``inf`` here would poison every aggregate downstream instead of
    surfacing the real problem.
    """
    q = np.abs(np.asarray(shares, dtype=float))
    v = np.asarray(adv, dtype=float)
    s = np.asarray(sigma, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        participation = np.where(v > 0, q / v, 0.0)
    participation = np.clip(np.nan_to_num(participation), 0.0, max_participation)
    return alpha * np.nan_to_num(s) * np.sqrt(participation)


def round_lots(shares: np.ndarray, lot_size: int = 100) -> np.ndarray:
    """Round buys down to whole board lots; leave sells intact.

    Rounding is **side-aware on purpose**. Buys must be whole lots on A-share
    boards, and rounding down keeps the trade inside its intended size. Sells
    are left unrounded because the tail of a position is legitimately an odd
    lot and must be exitable -- rounding a sell to a lot boundary would either
    strand shares forever or flip the position short.
    """
    if lot_size < 1:
        raise ValueError(f"lot_size must be >= 1, got {lot_size}")
    arr = np.asarray(shares, dtype=float)
    buys = np.floor(np.abs(arr) / lot_size) * lot_size
    return np.where(arr > 0, buys, arr)
