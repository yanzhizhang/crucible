"""Point-in-time universe resolution.

The single most common backtest lie in A-share research is running history
against *today's* CSI 300 constituents. Every name currently in the index got
there by performing well; the ones that fell out are missing. The resulting
backtest earns a large, entirely fake alpha, and it is invisible because
nothing errors.

crucible makes that failure impossible to reach by accident: membership must
carry effective dates, and a membership table without them is rejected at
construction rather than silently treated as always-valid.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from crucible.errors import UniverseError
from crucible.frames import SYMBOL, Frame, require_columns, to_polars

__all__ = ["Membership", "Universe", "universe"]

_MEMBERSHIP_COLS = ("index_code", SYMBOL, "start_date", "end_date")


@dataclass(frozen=True)
class Membership:
    """Index constituent intervals.

    Columns: ``index_code``, ``symbol``, ``start_date``, ``end_date``. A null
    ``end_date`` means still a member as of the data vendor's snapshot.

    Intervals are half-open ``[start_date, end_date)``: a name removed
    effective 2023-06-12 is a member on 2023-06-09 and not on 2023-06-12.
    """

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        require_columns(self.frame, _MEMBERSHIP_COLS, where="Membership")
        if self.frame["start_date"].null_count() > 0:
            raise UniverseError(
                "membership has null start_date; a constituent without an effective date "
                "cannot be resolved point-in-time. Reject the vendor file rather than "
                "assuming the name was always a member."
            )

    def at(self, date: dt.date | str) -> pl.DataFrame:
        """Rows in effect on ``date``."""
        d = _as_date(date)
        return self.frame.filter(
            (pl.col("start_date") <= d)
            & (pl.col("end_date").is_null() | (pl.col("end_date") > d))
        )

    @classmethod
    def from_frame(cls, df: Frame) -> Membership:
        """Build from any frame flavor, normalising date columns to ``date``."""
        lf = to_polars(df)
        require_columns(lf, _MEMBERSHIP_COLS, where="Membership.from_frame")
        return cls(
            lf.with_columns(
                pl.col("start_date").cast(pl.Date),
                pl.col("end_date").cast(pl.Date),
            )
        )


@dataclass(frozen=True)
class Universe:
    """Resolves a universe rule to a symbol set on a given date.

    Parameters
    ----------
    membership:
        Index constituent intervals, or ``None`` if only ``"all"`` and
        liquidity rules are needed.
    listings:
        Optional ``symbol``, ``list_date``, ``delist_date`` frame. Supplying it
        is what removes survivorship bias from the ``"all"`` rule -- without
        it, ``"all"`` means "every symbol the file happens to contain", which
        is exactly the snapshot bias this module exists to prevent.
    """

    membership: Membership | None = None
    listings: pl.DataFrame | None = None

    def at(self, date: dt.date | str, rule: str = "all") -> set[str]:
        """Symbols in the universe on ``date`` under ``rule``.

        ``rule`` is either ``"all"`` or an index code present in
        ``membership`` (e.g. ``"000300.SH"``, ``"000905.SH"``, ``"000852.SH"``).

        Point-in-time contract
        ----------------------
        Reads only intervals whose ``start_date <= date``. No row dated after
        ``date`` can influence the result.
        """
        d = _as_date(date)
        if rule == "all":
            return self._listed_on(d)

        if self.membership is None:
            raise UniverseError(
                f"rule {rule!r} needs a Membership table, but none was supplied. "
                "Pass one with effective dates; do not substitute current constituents."
            )
        members = set(self.membership.at(d).filter(pl.col("index_code") == rule)[SYMBOL].to_list())
        if not members:
            raise UniverseError(
                f"no constituents for index {rule!r} on {d}. Either the index code is "
                f"wrong or the membership file does not cover this date -- both are bugs, "
                f"not an empty universe."
            )
        listed = self._listed_on(d)
        return members & listed if listed else members

    def _listed_on(self, d: dt.date) -> set[str]:
        if self.listings is None:
            return set()
        lf = self.listings
        cond = pl.col("list_date") <= d
        if "delist_date" in lf.columns:
            cond = cond & (pl.col("delist_date").is_null() | (pl.col("delist_date") > d))
        return set(lf.filter(cond)[SYMBOL].to_list())

    def __repr__(self) -> str:
        n = 0 if self.listings is None else self.listings.height
        idx = 0 if self.membership is None else self.membership.frame["index_code"].n_unique()
        return f"Universe(listings={n}, indices={idx})"


def universe(date: dt.date | str, rule: str, registry: Universe) -> set[str]:
    """Point-in-time universe as a plain set.

    Thin free-function form of :meth:`Universe.at`, matching the signature the
    build contract specifies.
    """
    return registry.at(date, rule)


def _as_date(d: dt.date | str) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))
