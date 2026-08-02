"""Microstructure factors from tick-by-tick order and trade flow.

**This is a prism stand-in, not production.** It plays the producer role: it
emits a factor panel that the crucible pipeline then consumes through the same
loaders and gates it would use against a real C++ dump. Anything that survives
screening must be ported to prism before it can trade.

What the data allows, and what it does not
------------------------------------------
The available archive has the **order** and **transaction** streams but an
empty **quotation** stream. So there is no order book: no depth, no bid-ask
spread, no queue position. Every factor here is therefore built from *flow*
rather than *state*, and the ones that need a book are deliberately absent
rather than approximated:

* not built -- depth imbalance, quoted spread, book slope, queue position
* built instead -- order-flow imbalance, cancel dynamics, trade-flow toxicity,
  and two classical estimators (Roll, Kyle) that exist precisely because they
  recover spread and impact *without* quotes.

All operations are trailing and per symbol, so a value at slot ``t`` is
knowable at ``t``. Cross-sectional standardisation is left to :mod:`forge`.

Everything is polars-native and expression-based: the panel is millions of rows
and a Python-level loop over symbols would dominate the runtime.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = [
    "SLOT_AGGREGATES",
    "MICRO_FACTORS",
    "aggregate_to_slots",
    "build_factors",
    "add_forward_label",
]

_EPS: Final = 1e-12

SLOT_AGGREGATES: Final = (
    "n_trades", "volume", "amount", "vwap", "close", "high", "low",
    "buy_vol", "sell_vol", "n_buy", "n_sell", "max_trade",
    "n_add", "n_cancel", "add_vol", "cancel_vol",
    "buy_add_vol", "sell_add_vol", "buy_cancel_vol", "sell_cancel_vol",
    "n_market_order",
)

MICRO_FACTORS: Final = (
    "tfi", "ofi", "signed_vol_20", "cancel_ratio", "cancel_skew",
    "trade_intensity", "rv_20", "roll_spread", "kyle_lambda", "amihud",
    "vpin", "mom_20", "rev_3", "avg_trade_size", "large_trade_share",
    "order_size_imb", "market_order_ratio", "trade_to_order",
)


def aggregate_to_slots(
    orders: pl.DataFrame,
    trades: pl.DataFrame,
    *,
    slot_ns: int = 3_000_000_000,
    ts_col: str = "arrival_ts",
) -> pl.DataFrame:
    """Collapse raw ticks onto a right-labelled slot grid.

    Parameters
    ----------
    ts_col:
        **Defaults to ``arrival_ts``, not ``exchange_ts``.** A slot may only
        contain records we had actually received by its closing edge; bucketing
        on exchange time back-dates every record by the wire latency, which was
        measured at a 67 ms median and a 455 ms max on this feed -- up to 15%
        of a 3-second slot.

    Notes
    -----
    Slots are labelled by their **closing** edge, so slot ``t`` covers
    ``(t - slot_ns, t]`` and is complete at ``t``. This matches
    :mod:`almanac.calendar`.

    Direction on the trade stream is the aggressor side as the exchange reports
    it; ``dir == 0`` (unknown) contributes to volume but to neither side, since
    guessing would manufacture imbalance.
    """
    slot = ((pl.col(ts_col) // slot_ns) + 1) * slot_ns

    t = (
        trades.filter(pl.col("record_type") == "trade")
        .with_columns(slot.alias("slot"))
        .group_by("slot", "symbol")
        .agg(
            pl.len().alias("n_trades"),
            pl.col("volume").sum().alias("volume"),
            (pl.col("price") * pl.col("volume")).sum().alias("_pv"),
            pl.col("price").last().alias("close"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("volume").max().alias("max_trade"),
            pl.col("volume").filter(pl.col("side") == "buy").sum().alias("buy_vol"),
            pl.col("volume").filter(pl.col("side") == "sell").sum().alias("sell_vol"),
            pl.col("volume").filter(pl.col("side") == "buy").len().alias("n_buy"),
            pl.col("volume").filter(pl.col("side") == "sell").len().alias("n_sell"),
        )
        .with_columns(
            vwap=pl.col("_pv") / pl.col("volume").clip(lower_bound=_EPS),
            amount=pl.col("_pv"),
        )
        .drop("_pv")
    )

    is_add = pl.col("record_type") == "order_add"
    is_cxl = pl.col("record_type") == "order_cancel"
    o = (
        orders.with_columns(slot.alias("slot"))
        .group_by("slot", "symbol")
        .agg(
            pl.col("volume").filter(is_add).len().alias("n_add"),
            pl.col("volume").filter(is_cxl).len().alias("n_cancel"),
            pl.col("volume").filter(is_add).sum().alias("add_vol"),
            pl.col("volume").filter(is_cxl).sum().alias("cancel_vol"),
            pl.col("volume").filter(is_add & (pl.col("side") == "buy")).sum().alias("buy_add_vol"),
            pl.col("volume").filter(is_add & (pl.col("side") == "sell")).sum().alias("sell_add_vol"),
            pl.col("volume").filter(is_cxl & (pl.col("side") == "buy")).sum().alias("buy_cancel_vol"),
            pl.col("volume").filter(is_cxl & (pl.col("side") == "sell")).sum().alias("sell_cancel_vol"),
            # order_type 1 == market order (vendor encoding)
            pl.col("volume").filter(is_add & (pl.col("order_type") == 1)).len().alias("n_market_order"),
        )
    )

    panel = t.join(o, on=["slot", "symbol"], how="full", coalesce=True)
    filled = [
        pl.col(c).fill_null(0).alias(c)
        for c in SLOT_AGGREGATES
        if c in panel.columns and c not in ("vwap", "close", "high", "low")
    ]
    return panel.with_columns(filled).sort(["slot", "symbol"], maintain_order=True)


def build_factors(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Add microstructure factors. All trailing, all per symbol.

    Parameters
    ----------
    window:
        Lookback in slots for the rolling estimators. At 3s slots, 20 is one
        minute -- long enough to estimate a covariance, short enough that the
        microstructure regime has not changed underneath it.

    Notes
    -----
    Prices are forward-filled **within a symbol** across empty slots before
    returns are taken, because a slot with no trade is "price unchanged", not
    "price zero". Volume-type columns are *not* filled: an empty slot genuinely
    had no volume, and filling it would invent flow.
    """
    df = panel.sort(["symbol", "slot"], maintain_order=True)
    w = window

    df = df.with_columns(
        pl.col("close").forward_fill().over("symbol").alias("px"),
    ).with_columns(
        ret=(pl.col("px") / pl.col("px").shift(1).over("symbol") - 1.0),
        signed_vol=(pl.col("buy_vol") - pl.col("sell_vol")),
        tot_vol=(pl.col("buy_vol") + pl.col("sell_vol")),
    )

    dp = pl.col("ret")
    dp_lag = pl.col("ret").shift(1).over("symbol")

    return df.with_columns(
        # --- flow imbalance -------------------------------------------------
        # Trade-flow imbalance: aggressor-signed volume share. The most direct
        # read of who is crossing the spread.
        tfi=(pl.col("signed_vol") / pl.col("tot_vol").clip(lower_bound=_EPS)),
        # Order-flow imbalance in the Cont-Kukanov-Stoikov sense, adapted to a
        # book-free feed: added bid volume minus added ask volume, less the
        # cancels on each side. Cancels count with the opposite sign because
        # pulling a bid is economically the same as adding an ask.
        ofi=(
            (pl.col("buy_add_vol") - pl.col("buy_cancel_vol"))
            - (pl.col("sell_add_vol") - pl.col("sell_cancel_vol"))
        )
        / (pl.col("add_vol") + pl.col("cancel_vol")).clip(lower_bound=_EPS),
        signed_vol_20=(
            pl.col("signed_vol").rolling_sum(w, min_samples=w // 2).over("symbol")
            / pl.col("tot_vol").rolling_sum(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
        # --- liquidity provision / withdrawal -------------------------------
        # Cancel-to-add ratio: high values mean fleeting liquidity. A book-free
        # proxy for the quote instability that widens effective spreads.
        cancel_ratio=(pl.col("cancel_vol") / pl.col("add_vol").clip(lower_bound=_EPS)),
        cancel_skew=(
            (pl.col("buy_cancel_vol") - pl.col("sell_cancel_vol"))
            / pl.col("cancel_vol").clip(lower_bound=_EPS)
        ),
        trade_to_order=(pl.col("n_trades") / pl.col("n_add").clip(lower_bound=_EPS)),
        market_order_ratio=(pl.col("n_market_order") / pl.col("n_add").clip(lower_bound=_EPS)),
        order_size_imb=(
            (pl.col("buy_add_vol") / pl.col("n_buy").clip(lower_bound=1))
            - (pl.col("sell_add_vol") / pl.col("n_sell").clip(lower_bound=1))
        )
        / (pl.col("add_vol") / pl.col("n_add").clip(lower_bound=1)).clip(lower_bound=_EPS),
        # --- activity -------------------------------------------------------
        trade_intensity=(
            pl.col("n_trades")
            / pl.col("n_trades").rolling_mean(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
        avg_trade_size=(pl.col("volume") / pl.col("n_trades").clip(lower_bound=1)),
        large_trade_share=(pl.col("max_trade") / pl.col("volume").clip(lower_bound=_EPS)),
        # --- price process --------------------------------------------------
        rv_20=pl.col("ret").rolling_std(w, min_samples=w // 2).over("symbol"),
        mom_20=(pl.col("px") / pl.col("px").shift(w).over("symbol") - 1.0),
        rev_3=-(pl.col("px") / pl.col("px").shift(3).over("symbol") - 1.0),
        # Roll's implied spread: 2*sqrt(-cov(dp_t, dp_{t-1})). Recovers the
        # effective spread from bid-ask bounce alone, which is exactly the
        # estimator to reach for when the quote stream is missing. Positive
        # autocovariance means the bounce assumption fails (trending), so it is
        # clipped to null rather than reported as an imaginary spread.
        roll_spread=(
            2.0
            * (-(dp * dp_lag).rolling_mean(w, min_samples=w // 2).over("symbol"))
            .clip(lower_bound=0.0)
            .sqrt()
        ),
        # Kyle's lambda: price impact per unit signed volume, as a rolling
        # regression slope cov(ret, signed_vol) / var(signed_vol). High lambda
        # means a thin, easily-moved book.
        kyle_lambda=(
            (
                (pl.col("ret") * pl.col("signed_vol")).rolling_mean(w, min_samples=w // 2).over("symbol")
                - pl.col("ret").rolling_mean(w, min_samples=w // 2).over("symbol")
                * pl.col("signed_vol").rolling_mean(w, min_samples=w // 2).over("symbol")
            )
            / pl.col("signed_vol").rolling_var(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
        amihud=(
            pl.col("ret").abs().rolling_mean(w, min_samples=w // 2).over("symbol")
            / pl.col("amount").rolling_mean(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
        # VPIN-style toxicity: absolute order imbalance over total volume on a
        # trailing window. High values flag informed flow, which predicts
        # adverse selection for anyone providing liquidity into it.
        vpin=(
            pl.col("signed_vol").abs().rolling_sum(w, min_samples=w // 2).over("symbol")
            / pl.col("tot_vol").rolling_sum(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
    ).sort(["slot", "symbol"], maintain_order=True)


BOOK_FACTORS: Final = (
    "micro_tilt", "imb_l1", "imb_l5", "rel_spread",
    "order_count_imb", "bid_order_size", "ask_order_size", "book_slope",
    "consumption", "micro_mom_10", "micro_rev_3", "spread_z", "imb_persist",
)
# `depth_ratio` was removed: it computed
# (bid_depth5 - ask_depth5) / (bid_depth5 + ask_depth5), which is exactly
# `imbalance_l5`. The two returned byte-identical ICs.
#
# `micro_tilt` is kept but is NOT independent of `imb_l1`:
# microprice - mid == (spread / 2) * imbalance_l1, so after cross-sectional
# z-scoring the two produce the same ranking (measured IC +0.012700 vs
# +0.012698). The microprice edge lives in the level against the mid, which
# standardisation discards. Keep both only if you use the raw level.


def build_book_factors(panel: pl.DataFrame, *, window: int = 20) -> pl.DataFrame:
    """Factors that require a reconstructed order book.

    These are the ones a flow-only feed cannot produce, and they are the reason
    the MBO replay is worth its cost. The headline is ``micro_tilt``: the
    microprice minus the mid, in units of the spread. It answers "which side is
    about to get consumed", and it is consistently the strongest single
    short-horizon predictor in the microstructure literature.

    Expects the output of ``build_mbo_panel``. All windows are trailing and
    per symbol.
    """
    df = panel.sort(["symbol", "slot"], maintain_order=True)
    w = window

    two_sided = (pl.col("best_bid") > 0) & (pl.col("best_ask") > 0)
    spread = pl.when(two_sided).then(pl.col("spread")).otherwise(None)
    mid = pl.when(two_sided).then(pl.col("mid")).otherwise(None)
    micro = pl.when(two_sided).then(pl.col("microprice")).otherwise(None)

    df = df.with_columns(
        _mid=mid.forward_fill().over("symbol"),
        _micro=micro.forward_fill().over("symbol"),
        _spread=spread.forward_fill().over("symbol"),
    )

    return df.with_columns(
        # Microprice tilt in spread units. Bounded roughly [-0.5, 0.5]; positive
        # means the ask is thin relative to the bid, so the next move is up.
        micro_tilt=(
            (pl.col("_micro") - pl.col("_mid")) / pl.col("_spread").clip(lower_bound=_EPS)
        ),
        imb_l1=pl.col("imbalance_l1"),
        imb_l5=pl.col("imbalance_l5"),
        # Relative spread is the direct cost of crossing; it also proxies for
        # how much of any predicted move is actually capturable.
        rel_spread=(pl.col("_spread") / pl.col("_mid").clip(lower_bound=_EPS)),
        order_count_imb=(
            (pl.col("n_bid_orders") - pl.col("n_ask_orders"))
            / (pl.col("n_bid_orders") + pl.col("n_ask_orders")).clip(lower_bound=_EPS)
        ),
        # Average resting order size per side: small average size on the bid
        # means retail-like queue, large means institutional support.
        bid_order_size=(pl.col("bid_depth5") / pl.col("n_bid_orders").clip(lower_bound=1)),
        ask_order_size=(pl.col("ask_depth5") / pl.col("n_ask_orders").clip(lower_bound=1)),
        # How steeply depth builds away from touch. A flat book is easy to walk
        # through; a steep one absorbs size.
        book_slope=(
            (pl.col("bid_depth5") + pl.col("ask_depth5"))
            / (pl.col("bid_qty") + pl.col("ask_qty")).clip(lower_bound=_EPS)
        ),
        # Fraction of visible depth consumed within the slot -- an aggression
        # measure that flow alone cannot normalise.
        consumption=(
            pl.col("slot_qty")
            / (pl.col("bid_depth5") + pl.col("ask_depth5")).clip(lower_bound=_EPS)
        ),
        micro_mom_10=(pl.col("_micro") / pl.col("_micro").shift(10).over("symbol") - 1.0),
        micro_rev_3=-(pl.col("_micro") / pl.col("_micro").shift(3).over("symbol") - 1.0),
        # Spread relative to its own recent level: a widening book precedes
        # volatility and makes any signal more expensive to act on.
        spread_z=(
            (pl.col("_spread") - pl.col("_spread").rolling_mean(w, min_samples=w // 2).over("symbol"))
            / pl.col("_spread").rolling_std(w, min_samples=w // 2).over("symbol").clip(lower_bound=_EPS)
        ),
        # Does the imbalance persist or flicker? Persistent imbalance is real
        # pressure; flickering is quote noise.
        imb_persist=pl.col("imbalance_l1").rolling_mean(w, min_samples=w // 2).over("symbol"),
    ).sort(["slot", "symbol"], maintain_order=True)


def add_forward_label(
    panel: pl.DataFrame,
    *,
    horizon: int = 10,
    entry_lag: int = 1,
    label: str = "y",
) -> pl.DataFrame:
    """Forward return over ``horizon`` slots, entered ``entry_lag`` slots later.

    Uses an explicit slot-index join rather than ``shift``, for the reason
    given in :mod:`horizon.labels`: a symbol that is missing slots (no trades)
    would otherwise be paired with the wrong future.

    At 3s slots, ``horizon=10`` is 30 seconds and ``entry_lag=1`` means the
    decision at ``t`` is filled over the *next* slot -- the price stamped at
    ``t`` is already complete and cannot be traded.
    """
    idx = (
        panel.select(pl.col("slot").unique().sort())
        .with_row_index("_i")
        .with_columns(pl.col("_i").cast(pl.Int64))
    )
    df = panel.join(idx, on="slot", how="left")
    px = df.select("symbol", "_i", _px=pl.col("px"))

    entry = px.select("symbol", (pl.col("_i") - entry_lag).alias("_i"), _entry="_px")
    exit_ = px.select("symbol", (pl.col("_i") - entry_lag - horizon).alias("_i"), _exit="_px")

    return (
        df.join(entry, on=["symbol", "_i"], how="left")
        .join(exit_, on=["symbol", "_i"], how="left")
        .with_columns(
            pl.when(pl.col("_entry") > 0)
            .then(pl.col("_exit") / pl.col("_entry") - 1.0)
            .otherwise(None)
            .alias(label)
        )
        .drop("_i", "_entry", "_exit")
        .sort(["slot", "symbol"], maintain_order=True)
    )
