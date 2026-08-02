"""Tradability masks.

A masked observation is one where a price exists in the data but *no trade
could have happened at it*. The distinction matters more than it sounds: a
suspended stock keeps printing its last close in most vendor feeds, so a naive
return series shows a flat 0% for the suspension and then a jump on resumption.
Every factor evaluated on that series inherits a fake mean-reversion signal.

The rule crucible enforces is that masks apply to **features and labels
together**. Masking only the feature leaves the label carrying an unreachable
return; masking only the label leaves the model training on a state it could
never have acted from. :func:`apply_masks` therefore takes both.

Masked values become null rather than being dropped, so the ``(ts, symbol)``
grid stays rectangular and downstream cross-sectional operations still see a
consistent universe with visible holes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = ["Masks", "FLAGS", "build_masks", "apply_masks", "limit_pct", "limit_prices"]

FLAGS = ("st", "suspended", "limit_up", "limit_down", "newly_listed", "one_word_board")
"""Mask flag columns, in reporting order."""

#: Flags that make an observation untradeable in the direction of a *buy*.
_BLOCKS_BUY = ("suspended", "limit_up", "newly_listed", "one_word_board")

#: Flags that make an observation untradeable in the direction of a *sell*.
_BLOCKS_SELL = ("suspended", "limit_down", "newly_listed", "one_word_board")


@dataclass(frozen=True)
class Masks:
    """Point-in-time tradability flags on a ``(ts, symbol)`` grid.

    ``frame`` carries :data:`FLAGS` as boolean columns. ``True`` means the
    condition holds, i.e. the observation is impaired.
    """

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        require_columns(self.frame, (TS, SYMBOL, *FLAGS), where="Masks")

    def tradable(self, side: str = "both") -> pl.DataFrame:
        """``(ts, symbol, tradable)`` for the given side.

        ``side`` is ``"buy"``, ``"sell"`` or ``"both"``. Sides differ because a
        limit-up name can be sold but not bought -- collapsing them into one
        flag throws away half the tradable universe on a strong day.
        """
        blocking = {
            "buy": _BLOCKS_BUY,
            "sell": _BLOCKS_SELL,
            # sorted(), not set(): set iteration order over strings varies with
            # the hash seed, and determinism is not negotiable here.
            "both": tuple(sorted(set(_BLOCKS_BUY) | set(_BLOCKS_SELL))),
        }[side]
        impaired = pl.any_horizontal([pl.col(f) for f in blocking])
        return self.frame.select(TS, SYMBOL, pl.lit(True).and_(~impaired).alias("tradable"))

    def coverage(self) -> pl.DataFrame:
        """Per-flag hit counts by timestamp.

        Worth eyeballing before trusting any backtest: a universe that is 40%
        suspended on a given day is not a universe.
        """
        return (
            self.frame.group_by(TS)
            .agg(
                pl.len().alias("n"),
                *[pl.col(f).sum().alias(f) for f in FLAGS],
            )
            .sort(TS)
        )

    def __len__(self) -> int:
        return self.frame.height


def limit_pct(symbol: str, *, is_st: bool = False) -> float:
    """Daily price-limit band for an A-share symbol, as a fraction.

    Board rules as of the 2020 ChiNext reform:

    * ``688xxx`` STAR Market and ``300/301xxx`` ChiNext -- 20%, and ST status
      does not narrow it on these boards.
    * ``4xxxxx``/``8xxxxx`` Beijing Stock Exchange -- 30%.
    * everything else (SH/SZ main board) -- 10%, narrowed to 5% under ST.

    Newly listed names are exempt from limits on day one, which is why
    ``newly_listed`` is a separate flag rather than an adjustment here.
    """
    s = str(symbol)
    if s.startswith(("688", "689")) or s.startswith(("300", "301")):
        return 0.20
    if s.startswith(("4", "8")):
        return 0.30
    return 0.05 if is_st else 0.10


def limit_prices(prev_close: pl.Expr, symbol: pl.Expr, is_st: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """Upper and lower limit price expressions.

    Exchanges round the limit to the 0.01 tick using round-half-up. Comparing
    an unrounded limit against a traded price produces off-by-one-tick
    misclassification on roughly one name in a hundred, which is enough to
    corrupt a limit-up mask.
    """
    star = symbol.str.starts_with("688") | symbol.str.starts_with("689")
    chinext = symbol.str.starts_with("300") | symbol.str.starts_with("301")
    bse = symbol.str.starts_with("4") | symbol.str.starts_with("8")
    pct = (
        pl.when(star | chinext)
        .then(0.20)
        .when(bse)
        .then(0.30)
        .when(is_st)
        .then(0.05)
        .otherwise(0.10)
    )
    up = ((prev_close * (1 + pct)) * 100 + 0.5).floor() / 100
    down = ((prev_close * (1 - pct)) * 100 + 0.5).floor() / 100
    return up, down


def build_masks(
    daily: Frame,
    *,
    list_date_col: str = "list_date",
    min_sessions: int = 60,
    tol: float = 1e-6,
) -> Masks:
    """Derive :class:`Masks` from a daily bar frame.

    Expected columns: ``ts``, ``symbol``, ``open``, ``high``, ``low``,
    ``close``, ``prev_close``, ``volume``, and optionally ``is_st`` and
    ``list_date``. Prices must be **unadjusted** -- limit bands are defined on
    the traded price, so feeding an adjusted series misclassifies every limit
    day before the most recent corporate action.

    Flags
    -----
    suspended
        Zero volume on a scheduled session. Vendor feeds carry the stale prior
        close on these rows, which is exactly the stale-price trap.
    limit_up / limit_down
        Close sits at the rounded limit band.
    one_word_board
        Opened at the limit and never left it (``high == low``). Untradeable
        at the open in the limit direction, so a strategy cannot enter.
    newly_listed
        Fewer than ``min_sessions`` sessions of history. Early sessions have no
        price limit and abnormal turnover; including them contaminates any
        reversal factor.

    Point-in-time contract
    ----------------------
    Every flag reads only row ``t``'s own fields plus ``prev_close``, which is
    known at ``t``. No forward information enters.
    """
    lf = to_polars(daily)
    require_columns(
        lf,
        (TS, SYMBOL, "open", "high", "low", "close", "prev_close", "volume"),
        where="build_masks",
    )

    is_st = pl.col("is_st") if "is_st" in lf.columns else pl.lit(False)
    up, down = limit_prices(pl.col("prev_close"), pl.col(SYMBOL), is_st)

    out = lf.sort([SYMBOL, TS], maintain_order=True).with_columns(
        _limit_up=(pl.col("close") >= up - tol),
        _limit_down=(pl.col("close") <= down + tol),
        _flat=(pl.col("high") - pl.col("low")).abs() <= tol,
        _suspended=(pl.col("volume").fill_null(0) <= 0),
        _st=is_st.cast(pl.Boolean),
    )

    n_sessions = int(lf[TS].n_unique())
    if list_date_col in lf.columns:
        # Sessions since listing, approximated as ~5 sessions per 7 calendar
        # days. Counting true sessions would mean importing a calendar here,
        # and masks must stay usable on any grain without one.
        span = math.ceil(min_sessions * 7 / 5)
        listed = pl.col(list_date_col).cast(pl.Datetime("ns"))
        newly = (pl.col(TS) < listed + pl.duration(days=span)) | (pl.col(TS) < listed)
    elif n_sessions > min_sessions:
        # Position within the sample. Under-flags names that listed before the
        # sample starts, which is the safe direction -- it never masks a
        # genuinely seasoned name.
        newly = pl.col(TS).rank("ordinal").over(SYMBOL) <= min_sessions
    else:
        # Sample shorter than the seasoning window and no listing dates: a new
        # listing is indistinguishable from a short sample. Flagging by rank
        # here would mark the ENTIRE panel newly-listed and mask everything,
        # which looks like a working mask while silently deleting the universe.
        # Flag nothing; supply `list_date` if seasoning matters on this sample.
        newly = pl.lit(False)

    out = out.with_columns(
        st=pl.col("_st"),
        suspended=pl.col("_suspended"),
        limit_up=pl.col("_limit_up") & ~pl.col("_suspended"),
        limit_down=pl.col("_limit_down") & ~pl.col("_suspended"),
        newly_listed=newly.fill_null(False),
        one_word_board=pl.col("_flat")
        & (pl.col("_limit_up") | pl.col("_limit_down"))
        & ~pl.col("_suspended"),
    )
    return Masks(out.select(TS, SYMBOL, *FLAGS).sort([TS, SYMBOL], maintain_order=True))


def apply_masks(
    df: Frame,
    masks: Masks,
    *,
    columns: Sequence[str] | None = None,
    side: str = "both",
    drop: bool = False,
) -> Frame:
    """Null out impaired observations in ``df``.

    Parameters
    ----------
    columns:
        Value columns to mask. Defaults to every column except ``ts`` and
        ``symbol``. **Pass features and labels in the same call** -- masking
        them in separate passes is how a train set ends up with a label the
        feature row was never allowed to act on.
    side:
        Which direction of tradability to require; see :meth:`Masks.tradable`.
    drop:
        Remove masked rows entirely instead of nulling them. Off by default:
        nulling keeps the ``(ts, symbol)`` grid rectangular so cross-sectional
        ranks and z-scores still align across the panel.

    Returns
    -------
    The frame in the caller's flavor, with masked entries set to null.

    Notes
    -----
    Symbols absent from ``masks`` are treated as **untradeable**, not as
    tradable-by-default. An unknown name is far more often a coverage gap than
    a genuinely clean one, and defaulting to tradable would quietly readmit
    exactly the rows the mask exists to exclude.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (TS, SYMBOL), where="apply_masks")

    value_cols = (
        list(columns) if columns is not None else [c for c in lf.columns if c not in (TS, SYMBOL)]
    )
    unknown = [c for c in value_cols if c not in lf.columns]
    if unknown:
        raise KeyError(f"apply_masks: column(s) {unknown} not in frame")

    joined = lf.join(masks.tradable(side), on=[TS, SYMBOL], how="left")
    ok = pl.col("tradable").fill_null(False)

    if drop:
        out = joined.filter(ok).drop("tradable")
    else:
        out = joined.with_columns(
            [pl.when(ok).then(pl.col(c)).otherwise(None).alias(c) for c in value_cols]
        ).drop("tradable")

    return restore(out.sort([TS, SYMBOL], maintain_order=True), want)
