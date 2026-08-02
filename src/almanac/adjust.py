"""Corporate-action price adjustment.

Which direction you adjust in is a point-in-time question, not a style choice.

**Backward (后复权, hfq)** anchors the *first* observation and scales later
prices up. The adjusted value at time ``t`` depends only on actions at or
before ``t``, so re-running the pipeline next year reproduces today's numbers
exactly. This is the only adjustment crucible uses for features and labels.

**Forward (前复权, qfq)** anchors the *latest* observation and scales earlier
prices down. Every new dividend rewrites the entire history. It is convenient
for charting and actively dangerous for research: a factor computed on qfq
prices in January is not the same factor recomputed in July, and any stored
result silently disagrees with a fresh run. :func:`adjust` supports it but
refuses to be the default and says so.

Whichever you use, the **unadjusted price is kept**. Margin requirements, lot
sizing, stamp duty and price limits are all defined on the traded price. Sizing
a position off a backward-adjusted price that has drifted 3x above the real one
buys three times the intended exposure.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import polars as pl

from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = ["event_factors", "cumulative_factors", "adjust"]

DEFAULT_PRICE_COLS = ("open", "high", "low", "close", "vwap", "prev_close")


def event_factors(actions: Frame) -> pl.DataFrame:
    """Per-event price ratio from raw corporate actions.

    Expected columns: ``ts`` (ex-date), ``symbol``, ``prev_close``, and any of
    ``cash_div`` (cash per share), ``split_ratio`` (bonus + split shares issued
    per existing share), ``rights_ratio`` and ``rights_price``.

    The theoretical ex-price follows the CSRC formula::

        ex_price = (prev_close - cash_div + rights_ratio * rights_price)
                   / (1 + split_ratio + rights_ratio)

    and the event factor is ``ex_price / prev_close`` -- the multiplicative
    gap the quoted price jumps across on the ex-date, which is precisely the
    part of the return that is *not* a return.
    """
    lf = to_polars(actions)
    require_columns(lf, (TS, SYMBOL, "prev_close"), where="event_factors")

    def col(name: str) -> pl.Expr:
        return pl.col(name).fill_null(0.0) if name in lf.columns else pl.lit(0.0)

    ex_price = (pl.col("prev_close") - col("cash_div") + col("rights_ratio") * col("rights_price")) / (
        1.0 + col("split_ratio") + col("rights_ratio")
    )
    return (
        lf.with_columns(factor=(ex_price / pl.col("prev_close")))
        .select(TS, SYMBOL, "factor")
        .filter(pl.col("factor").is_not_null() & (pl.col("factor") > 0))
        .sort([SYMBOL, TS], maintain_order=True)
    )


def cumulative_factors(
    events: Frame,
    grid: Frame,
    *,
    mode: str = "backward",
) -> pl.DataFrame:
    """Expand per-event factors onto a full ``(ts, symbol)`` grid.

    Parameters
    ----------
    events:
        Output of :func:`event_factors`, or any ``ts, symbol, factor`` frame.
    grid:
        The ``(ts, symbol)`` rows to produce factors for -- typically the bar
        frame about to be adjusted.
    mode:
        ``"backward"`` (default, PIT-safe) or ``"forward"``.

    Returns
    -------
    ``ts, symbol, adj_factor`` where the adjusted price is
    ``raw_price * adj_factor``.

    Point-in-time contract
    ----------------------
    In ``"backward"`` mode the factor at ``t`` is a cumulative product over
    events with ex-date ``<= t`` only. In ``"forward"`` mode it reads events
    strictly after ``t``, which is future information by construction -- do not
    use it to build features.
    """
    if mode not in ("backward", "forward"):
        raise ValueError(f"mode must be 'backward' or 'forward', got {mode!r}")

    ev = to_polars(events)
    require_columns(ev, (TS, SYMBOL, "factor"), where="cumulative_factors")
    g = to_polars(grid).select(TS, SYMBOL).unique().sort([SYMBOL, TS], maintain_order=True)

    joined = (
        g.join(ev.select(TS, SYMBOL, "factor"), on=[TS, SYMBOL], how="left")
        .with_columns(pl.col("factor").fill_null(1.0))
        .sort([SYMBOL, TS], maintain_order=True)
    )

    if mode == "backward":
        # Anchor the earliest bar at its raw price; scale forward by 1/f.
        adj = (1.0 / pl.col("factor")).cum_prod().over(SYMBOL)
    else:
        # Anchor the latest bar; each earlier bar carries the product of all
        # factors strictly after it. Reverse cum_prod, then shift off self.
        adj = (
            pl.col("factor")
            .reverse()
            .cum_prod()
            .reverse()
            .shift(-1)
            .fill_null(1.0)
            .over(SYMBOL)
        )

    return joined.with_columns(adj_factor=adj).select(TS, SYMBOL, "adj_factor")


def adjust(
    bars: Frame,
    factors: Frame,
    *,
    mode: str = "backward",
    price_cols: Sequence[str] = DEFAULT_PRICE_COLS,
    volume_col: str | None = "volume",
    keep_raw: bool = True,
    raw_suffix: str = "_raw",
) -> Frame:
    """Apply adjustment factors to a bar frame.

    Price columns are multiplied by ``adj_factor``; ``volume_col`` is divided
    by it, since a 2-for-1 split halves the price and doubles the share count
    and only the product -- turnover -- is invariant.

    Parameters
    ----------
    keep_raw:
        Retain the unadjusted columns under ``raw_suffix``. On by default and
        you should leave it on: cost models, lot rounding and limit bands are
        all defined on traded prices.

    Warns
    -----
    UserWarning
        When ``mode="forward"``, because forward adjustment is not
        reproducible across data vintages.
    """
    if mode == "forward":
        warnings.warn(
            "forward (qfq) adjustment is not point-in-time: every future dividend "
            "rewrites this history, so results are not reproducible across data "
            "vintages. Use mode='backward' for features and labels.",
            UserWarning,
            stacklevel=2,
        )

    want = flavor(bars)
    lf = to_polars(bars)
    require_columns(lf, (TS, SYMBOL), where="adjust")

    fac = to_polars(factors)
    if "adj_factor" not in fac.columns:
        fac = cumulative_factors(fac, lf, mode=mode)

    out = lf.join(fac.select(TS, SYMBOL, "adj_factor"), on=[TS, SYMBOL], how="left").with_columns(
        pl.col("adj_factor").fill_null(1.0)
    )

    present = [c for c in price_cols if c in lf.columns]
    keep: list[pl.Expr] = []
    if keep_raw:
        keep = [pl.col(c).alias(f"{c}{raw_suffix}") for c in present]
        if volume_col and volume_col in lf.columns:
            keep.append(pl.col(volume_col).alias(f"{volume_col}{raw_suffix}"))

    scaled: list[pl.Expr] = [(pl.col(c) * pl.col("adj_factor")).alias(c) for c in present]
    if volume_col and volume_col in lf.columns:
        scaled.append((pl.col(volume_col) / pl.col("adj_factor")).alias(volume_col))

    return restore(
        out.with_columns(keep).with_columns(scaled).sort([TS, SYMBOL], maintain_order=True), want
    )
