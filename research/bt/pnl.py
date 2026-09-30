"""Fees and the exact PnL decomposition of simulated fills.

For every fill with side ``s`` (+1 buy / -1 sell), quantity ``q``, price ``p``, the known mid at
the decision ``m0``, the true mid when the order reached the exchange ``m1``, and the day's
closing mid ``m_end`` (open positions are marked there):

    total   = s*q*(m_end - p) - fee
    signal  = s*q*(m_end - m0)      what the decision was worth at the mid it was made on
    latency = -s*q*(m1 - m0)        the mid moved between deciding and arriving
    spread  = -s*q*(p - m1)         crossing the spread / walking the book (negative) or
                                    earning it with a passive fill (positive)
    fees    = -fee

``signal + latency + spread + fees == total`` holds exactly, per fill. Over a round trip the
``m_end`` terms cancel, so ``signal`` becomes the mid-to-mid move between the two decisions.
Legs of kind RESTORE (forced end-of-day close under T+1) are reported as one extra bucket,
``restore``, so the cost of having to be flat is visible on its own.

Fees follow :class:`toll.CostModel`: commission with a per-order minimum (an order = one leg of
one trip, however many fills), transfer fee on both sides, stamp duty on sells only (rate by
date, :func:`toll.stamp_duty_for`). Impact is not charged separately: aggressive fills already
walk the real book.
"""

from __future__ import annotations

import polars as pl

from toll.costs import CostModel, stamp_duty_for

FILL_COLS = ("trip", "leg", "side", "qty", "price", "x_time", "t_decision", "m0", "m1",
             "passive", "ahead_at_entry")
LEG_NAMES = {0: "entry", 1: "exit", 2: "restore", 3: "mark"}


def to_frame(F, lo_tick: int, m_end_idx: float, symbol: str, date: str) -> pl.DataFrame:
    """Engine output (book-index prices) -> fills frame with CNY prices."""
    df = pl.DataFrame(F, schema=list(FILL_COLS), orient="row")
    idx_to_px = lambda c: pl.when(pl.col(c) >= 0).then((pl.col(c) + lo_tick) / 100.0)  # noqa: E731
    return df.with_columns(
        pl.lit(symbol).alias("symbol"), pl.lit(date).alias("date"),
        pl.col("trip", "leg", "side", "qty", "passive", "ahead_at_entry").cast(pl.Int64),
        pl.col("x_time", "t_decision").cast(pl.Int64),
        idx_to_px("price").alias("price"), idx_to_px("m0").alias("m0"), idx_to_px("m1").alias("m1"),
        pl.lit((m_end_idx + lo_tick) / 100.0).alias("m_end"),
    ).with_columns(pl.col("leg").replace_strict(LEG_NAMES).alias("leg_name"))


def with_fees(fills: pl.DataFrame, cost: CostModel | None, date: str) -> pl.DataFrame:
    """Add per-fill ``fee`` (CNY): order-level commission minimum spread pro rata over its fills."""
    if cost is None or fills.height == 0:
        return fills.with_columns(pl.lit(0.0).alias("fee"))
    stamp = stamp_duty_for(f"{date[:4]}-{date[4:6]}-{date[6:]}")
    notional = pl.col("qty") * pl.col("price")
    order = ["symbol", "trip", "leg"]
    return (
        fills.with_columns(notional.alias("_n"))
        .with_columns(pl.col("_n").sum().over(order).alias("_order_n"))
        .with_columns(
            pl.when(pl.col("leg") == 3).then(0.0).otherwise(
                # commission: order-level max(rate*notional, minimum), allocated by notional share
                pl.max_horizontal(pl.col("_order_n") * cost.commission, pl.lit(cost.min_commission))
                * pl.col("_n") / pl.col("_order_n")
                + pl.col("_n") * cost.transfer_fee
                + pl.when(pl.col("side") < 0).then(pl.col("_n") * stamp).otherwise(0.0)
            ).alias("fee")
        )
        .drop("_n", "_order_n")
    )


def decompose(fills: pl.DataFrame) -> pl.DataFrame:
    """Per-fill components; their sum equals ``total`` exactly."""
    sq = pl.col("side") * pl.col("qty")
    return fills.with_columns(
        (sq * (pl.col("m_end") - pl.col("m0"))).alias("signal"),
        (-sq * (pl.col("m1") - pl.col("m0"))).alias("latency"),
        (-sq * (pl.col("price") - pl.col("m1"))).alias("spread"),
        (-pl.col("fee")).alias("fees"),
        (sq * (pl.col("m_end") - pl.col("price")) - pl.col("fee")).alias("total"),
    )


def summarize(fills: pl.DataFrame) -> dict[str, float]:
    """Component sums for one run; restore legs collapsed into their own ``restore`` bucket."""
    if fills.height == 0:
        return dict.fromkeys(("signal", "latency", "spread", "fees", "restore", "total"), 0.0)
    main = fills.filter(pl.col("leg") != 2)
    rest = fills.filter(pl.col("leg") == 2)
    out = {c: float(main[c].sum()) for c in ("signal", "latency", "spread", "fees")}
    out["restore"] = float(rest["total"].sum()) if rest.height else 0.0
    out["total"] = float(fills["total"].sum())
    return out
