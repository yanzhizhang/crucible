"""The toy signal driving the learning backtest: 1-minute order-flow imbalance, momentum.

This is deliberately simple -- the point of stage 7 is to see what latency, queueing, costs and
T+1 do to *any* signal, not to have a good one. It is swapped for the reproduced PM model
predictions once stages 3-6 land.

For each stock and each 1-minute bar, ``imb = sum(buy_vol - sell_vol) / sum(volume)`` over the
last ``window`` bars (aggressor-signed, from the trade stream). When ``|imb| > threshold`` the
intent is to trade in the direction of the flow. The decision time is the bar's point-in-time
``available_ts`` (when the capture had actually seen the whole bar) -- never the bar label.
"""

from __future__ import annotations

import polars as pl

_LOCAL = 8 * 3600 * 10**9


def intents(bars: pl.DataFrame, *, window: int = 3, threshold: float = 0.35,
            start: str = "09:35", end: str = "14:50") -> pl.DataFrame:
    """(symbol, t_local_ns, side) intents from 1-minute bars, sorted by symbol then time."""
    b = bars.sort("symbol", "ts").with_columns(
        (pl.col("buy_volume") - pl.col("sell_volume")).rolling_sum(window).over("symbol").alias("_net"),
        pl.col("volume").rolling_sum(window).over("symbol").alias("_vol"),
    ).with_columns((pl.col("_net") / pl.col("_vol")).alias("imb"))
    hhmm = pl.col("ts").dt.strftime("%H:%M")
    return (
        b.filter((hhmm >= start) & (hhmm <= end) & (pl.col("_vol") > 0)
                 & (pl.col("imb").abs() > threshold) & pl.col("available_ts").is_not_null())
        .select(
            "symbol",
            (pl.col("available_ts").cast(pl.Datetime("ns")).dt.epoch("ns") - _LOCAL).alias("t_local"),
            pl.col("imb").sign().cast(pl.Int64).alias("side"),
            "imb",
        )
        .sort("symbol", "t_local")
    )
