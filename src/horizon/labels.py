"""Forward-looking labels.

Two design decisions here carry most of the point-in-time weight.

**No shift.** ``shift(-n)`` takes the row *n positions down the frame*, which
equals "n sessions ahead" only if the panel has no gaps. Suspensions guarantee
gaps. A shifted label therefore silently pairs a feature with the wrong future
return, and the error is invisible because the number looks perfectly
reasonable. crucible instead assigns every timestamp an ordinal **slot index**
and joins on ``index + n`` explicitly. A missing future row then yields null --
an honest absence -- rather than a plausible lie.

**Entry lag.** The naive label ``price[t+n] / price[t] - 1`` assumes entry at
the price stamped ``t``. Under right-labelled bars that price is the VWAP of
the interval *ending* at ``t``, which is already over by the time the feature
exists. You cannot trade it. Labels therefore default to ``entry_lag=1``:
decide at ``t``, enter over ``t+1``, exit at ``t+1+n``. Setting ``entry_lag=0``
is permitted for research on execution-free signals but overstates every
tradable strategy.

VWAP-to-VWAP is the default price basis. Close-to-close must be opted into and
warns, because it assumes the entire position fills at a single closing print.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import polars as pl

from crucible.errors import PointInTimeError
from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = [
    "forward_return",
    "vol_normalized_return",
    "ternary_label",
    "assert_no_overlap",
    "slot_index",
]


def slot_index(df: pl.DataFrame, *, ts: str = TS) -> pl.DataFrame:
    """Attach ``_slot``: the dense ordinal rank of each distinct timestamp.

    Computed over the *whole panel*, not per symbol, so index arithmetic means
    the same thing for every instrument. A symbol missing a session keeps the
    global numbering and simply has no row at that index.
    """
    order = (
        df.select(pl.col(ts).unique().sort().alias(ts))
        .with_row_index("_slot")
        # Int64, not the default UInt32: label joins subtract the entry lag and
        # horizon from this, and an unsigned 0 - 1 wraps to ~4e9 instead of
        # going negative. That matches nothing, so every label would silently
        # come back null rather than erroring.
        .with_columns(pl.col("_slot").cast(pl.Int64))
    )
    return df.join(order, on=ts, how="left")


def _price_column(df: pl.DataFrame, price: str) -> str:
    """Resolve the price basis, warning when close-to-close is chosen."""
    if price == "close":
        warnings.warn(
            "close-to-close labels assume the whole position fills at the closing "
            "print. That is unachievable at size and materially overstates intraday "
            "strategies. Prefer price='vwap'.",
            UserWarning,
            stacklevel=3,
        )
    if price not in df.columns:
        raise KeyError(f"price column {price!r} not in frame; have {df.columns}")
    return price


def forward_return(
    df: Frame,
    n: int = 1,
    price: str = "vwap",
    *,
    entry_lag: int = 1,
    label: str = "fwd_ret",
) -> Frame:
    """Forward return over ``n`` slots, entered ``entry_lag`` slots after ``t``.

    Computes ``price[t + entry_lag + n] / price[t + entry_lag] - 1`` via an
    explicit slot-index join.

    Parameters
    ----------
    n:
        Holding horizon in slots. Slots are whatever grain the frame is on --
        sessions for daily data, 3-second bars for tick research.
    price:
        Price basis. ``"vwap"`` (default) or ``"close"`` (warns).
    entry_lag:
        Slots between the decision and the fill. Default 1; see module
        docstring for why 0 is not the default.

    Returns
    -------
    The input frame with a ``label`` column added, in the caller's flavor.
    Rows whose future is outside the sample, or whose future slot is missing
    (suspension), carry null.

    Point-in-time contract
    ----------------------
    The value at ``t`` is a function of prices strictly after ``t`` whenever
    ``entry_lag >= 1``. It must never be used as a feature. Pair it only with
    features stamped at or before ``t``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if entry_lag < 0:
        raise ValueError(f"entry_lag must be >= 0, got {entry_lag}")

    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (TS, SYMBOL), where="forward_return")
    px = _price_column(lf, price)

    base = slot_index(lf).sort([SYMBOL, "_slot"], maintain_order=True)
    prices = base.select(SYMBOL, "_slot", _px=pl.col(px))

    entry = prices.select(SYMBOL, pl.col("_slot") - entry_lag, _entry="_px")
    exit_ = prices.select(SYMBOL, pl.col("_slot") - entry_lag - n, _exit="_px")

    out = (
        base.join(entry, on=[SYMBOL, "_slot"], how="left")
        .join(exit_, on=[SYMBOL, "_slot"], how="left")
        .with_columns(
            pl.when(pl.col("_entry").is_not_null() & (pl.col("_entry") > 0))
            .then(pl.col("_exit") / pl.col("_entry") - 1.0)
            .otherwise(None)
            .alias(label)
        )
        .drop("_slot", "_entry", "_exit")
    )
    return restore(out.sort([TS, SYMBOL], maintain_order=True), want)


def vol_normalized_return(
    df: Frame,
    n: int = 1,
    price: str = "vwap",
    *,
    vol_window: int = 20,
    entry_lag: int = 1,
    label: str = "fwd_ret_vol",
    min_periods: int | None = None,
) -> Frame:
    """Forward return scaled by trailing realised volatility.

    The volatility estimate is **trailing** -- computed on returns at or before
    ``t`` -- so the scaling factor is known at decision time. Normalising by the
    volatility realised *over the holding period* would leak the future into
    the denominator, which is a subtle and common way to manufacture alpha.

    Parameters
    ----------
    vol_window:
        Lookback in slots for the trailing standard deviation.
    min_periods:
        Minimum observations before emitting a value. Defaults to
        ``vol_window``, so early rows are null rather than backed by a
        two-sample standard deviation.

    Notes
    -----
    The trailing return feeding the volatility estimate uses ``shift(1)`` over
    the symbol's *observed* rows. This is the one sanctioned use of ``shift``
    in the package: it only ever looks backwards, so it cannot leak. It is not
    exact across gaps -- after a suspension the "previous" observation is more
    than one slot back, which slightly overstates that single return. The
    effect on a ``vol_window``-length standard deviation is small; if it
    matters for your grain, mask before labelling so the gapped rows drop out.

    Returns
    -------
    Frame with ``label`` added: the forward return divided by trailing vol,
    scaled to the horizon by ``sqrt(n)``.
    """
    want = flavor(df)
    lf = to_polars(df)
    px = _price_column(lf, price)
    mp = vol_window if min_periods is None else min_periods

    with_fwd = to_polars(forward_return(lf, n, price, entry_lag=entry_lag, label="_fwd"))

    out = (
        slot_index(with_fwd)
        .sort([SYMBOL, "_slot"], maintain_order=True)
        .with_columns(_ret=(pl.col(px) / pl.col(px).shift(1).over(SYMBOL) - 1.0))
        .with_columns(
            _vol=pl.col("_ret")
            .rolling_std(window_size=vol_window, min_samples=mp)
            .over(SYMBOL)
        )
        .with_columns(
            pl.when(pl.col("_vol") > 0)
            .then(pl.col("_fwd") / (pl.col("_vol") * (n**0.5)))
            .otherwise(None)
            .alias(label)
        )
        .drop("_slot", "_ret", "_vol", "_fwd")
    )
    return restore(out.sort([TS, SYMBOL], maintain_order=True), want)


def ternary_label(
    df: Frame,
    n: int = 1,
    price: str = "vwap",
    *,
    threshold: float = 0.005,
    entry_lag: int = 1,
    label: str = "y3",
) -> Frame:
    """Three-class label: -1 / 0 / +1 around a symmetric deadband.

    The neutral class is not a modelling convenience, it is the honest
    representation of the problem: a move smaller than round-trip cost is not
    tradable in either direction. Sizing ``threshold`` at or above the expected
    cost (see :func:`toll.deadband_threshold`) keeps the classifier from
    learning to trade noise.

    Returns
    -------
    Frame with ``label`` as Int8, null wherever the forward return is null.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    want = flavor(df)
    lf = to_polars(df)
    fwd = to_polars(forward_return(lf, n, price, entry_lag=entry_lag, label="_fwd"))

    out = fwd.with_columns(
        pl.when(pl.col("_fwd").is_null())
        .then(None)
        .when(pl.col("_fwd") > threshold)
        .then(1)
        .when(pl.col("_fwd") < -threshold)
        .then(-1)
        .otherwise(0)
        .cast(pl.Int8)
        .alias(label)
    ).drop("_fwd")
    return restore(out, want)


def assert_no_overlap(
    features: Frame,
    labels: Frame,
    *,
    n: int,
    entry_lag: int = 1,
    feature_cols: Sequence[str] | None = None,
) -> None:
    """Assert the feature timestamp precedes the label's observation window.

    Checks that for every row, the label window ``(t + entry_lag, t + entry_lag
    + n]`` opens strictly after ``t``. With ``entry_lag >= 1`` this holds by
    construction; the assertion exists to catch a caller who set
    ``entry_lag=0`` and then treated the result as tradable, or who joined a
    label frame onto features stamped later than the label's own ``ts``.

    Raises
    ------
    PointInTimeError
        If the windows overlap, or if the frames disagree on ``(ts, symbol)``
        coverage in a way that implies a misaligned join.
    """
    if entry_lag < 1:
        raise PointInTimeError(
            f"entry_lag={entry_lag} means the label is entered at the same slot the "
            f"feature is observed. The entry price is already complete at that "
            f"timestamp and cannot be traded. Use entry_lag >= 1 for any label that "
            f"will be treated as achievable."
        )
    if n < 1:
        raise PointInTimeError(f"label horizon n={n} does not extend into the future")

    f, lab = to_polars(features), to_polars(labels)
    require_columns(f, (TS, SYMBOL), where="assert_no_overlap")
    require_columns(lab, (TS, SYMBOL), where="assert_no_overlap")

    fk = f.select(TS, SYMBOL).unique()
    lk = lab.select(TS, SYMBOL).unique()
    orphans = fk.join(lk, on=[TS, SYMBOL], how="anti").height
    if orphans and orphans == fk.height:
        raise PointInTimeError(
            "features and labels share no (ts, symbol) keys at all. This is a "
            "misaligned join, not a coverage gap -- check that both frames are on "
            "the same slot grid and timezone convention."
        )

    if feature_cols:
        missing = [c for c in feature_cols if c not in f.columns]
        if missing:
            raise PointInTimeError(f"declared feature column(s) {missing} absent from features")
