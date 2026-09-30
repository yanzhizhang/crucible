"""Point-in-time industry classification.

Industry is reference data that *changes*, and every way it changes breaks a
naive join:

* an issuer is reclassified (``机械设备`` -> ``汽车``) and the whole history
  retroactively adopts the new label;
* the vendor renames or renumbers a level, so two years of a study silently
  straddle two definitions;
* a name is absent from an old snapshot and present in a new one, and the
  neutralisation regression quietly drops it.

None of those raise anything. The residual just gets a little cleaner and the
backtest a little better, which is the worst possible failure mode.

So this module treats industry exactly the way :mod:`almanac.universe` treats
index membership: intervals with effective dates, or a refusal. A vendor table
without ``start_date`` cannot be loaded, and a lookup on a date the table does
not cover raises rather than returning today's labels.

Schemes
-------
``scheme`` names the classification system -- ``"citic"``, ``"wind"``,
``"sw"``, ``"em"``. It is part of the key everywhere, because mixing two
systems in one panel produces a column that looks fine and means nothing.
``level`` is 1 for the coarsest tier; CITIC runs 1..3, Wind 1..4, the
Eastmoney stand-in has only level 1.

Building history from snapshots
-------------------------------
Licensed feeds ship the change history directly. Free endpoints ship only
"as of now", so :func:`classification_from_snapshots` folds a series of dated
snapshots into intervals: a symbol keeps an industry until a later snapshot
disagrees, and the change is dated at the snapshot that first saw it. That
dating is an upper bound on the true change date -- the reclassification
happened somewhere in the gap between two snapshots -- so snapshot cadence is
the resolution of the history. Say so in the tearsheet; do not interpolate.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import polars as pl

from crucible.errors import UniverseError
from crucible.frames import SYMBOL, Frame, require_columns, to_polars

__all__ = [
    "INDUSTRY",
    "Classification",
    "classification_from_snapshots",
]

INDUSTRY = "industry"
"""Canonical column name for the attached label. :func:`forge.ops.neutralize`
and :func:`ballast.portfolio` both group on it."""

_CLASSIFICATION_COLS = (
    "scheme",
    "level",
    SYMBOL,
    "industry_code",
    "industry_name",
    "start_date",
    "end_date",
)

_SNAPSHOT_COLS = ("asof", "scheme", "level", SYMBOL, "industry_code", "industry_name")


@dataclass(frozen=True)
class Classification:
    """Effective-dated industry membership.

    Columns: ``scheme``, ``level``, ``symbol``, ``industry_code``,
    ``industry_name``, ``start_date``, ``end_date``. Intervals are half-open
    ``[start_date, end_date)``, matching :class:`almanac.universe.Membership`,
    so a reclassification effective 2023-06-12 takes effect on that date and
    not the day before.

    A null ``end_date`` means "still current as of the last snapshot", not
    "forever". Reading past the last snapshot is a point-in-time question the
    data cannot answer, which is why :meth:`at` checks coverage.
    """

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        require_columns(self.frame, _CLASSIFICATION_COLS, where="Classification")
        if self.frame["start_date"].null_count() > 0:
            raise UniverseError(
                "classification has null start_date; an industry label without an "
                "effective date would be applied to all of history, which is exactly "
                "the retroactive-reclassification bias this type exists to prevent."
            )
        dupes = (
            self.frame.group_by("scheme", "level", SYMBOL, "start_date")
            .len()
            .filter(pl.col("len") > 1)
        )
        if dupes.height:
            raise UniverseError(
                f"{dupes.height} (scheme, level, symbol, start_date) keys appear more "
                "than once. An industry map must be single-valued, or every join "
                "against it silently duplicates rows."
            )

    @property
    def schemes(self) -> list[str]:
        return sorted(self.frame["scheme"].unique().to_list())

    def coverage(self, scheme: str, level: int = 1) -> tuple[dt.date, dt.date | None]:
        """Earliest ``start_date`` and latest ``end_date`` for a scheme/level.

        A ``None`` upper bound means at least one interval is still open.
        """
        lf = self._select(scheme, level)
        ends = lf["end_date"]
        start = _min_date(lf)
        assert start is not None  # non-null start_date is enforced in __post_init__
        return start, None if ends.null_count() else ends.max()  # type: ignore[return-value]

    def at(self, date: dt.date | str, scheme: str, level: int = 1) -> pl.DataFrame:
        """Labels in effect on ``date``: ``symbol``, ``industry_code``, ``industry``.

        Point-in-time contract
        ----------------------
        Only intervals with ``start_date <= date`` are read. Raises if ``date``
        precedes the first snapshot, because the honest answer there is "not
        known", and returning the earliest known labels instead would back-cast
        a classification onto a period it was never in force for.
        """
        d = _as_date(date)
        lf = self._select(scheme, level)
        first = _min_date(lf)
        if first is not None and d < first:
            raise UniverseError(
                f"no {scheme} level-{level} classification on {d}; the earliest snapshot "
                f"is {first}. Extending it backwards would apply today's industry labels "
                "to history. Either shorten the sample or obtain a vendor change history."
            )
        out = lf.filter(
            (pl.col("start_date") <= d) & (pl.col("end_date").is_null() | (pl.col("end_date") > d))
        )
        if out.is_empty():
            raise UniverseError(
                f"{scheme} level-{level} classification is empty on {d}; the table has "
                "rows but none in effect, which means a gap in the snapshot history."
            )
        return out.select(
            SYMBOL,
            "industry_code",
            pl.col("industry_name").alias(INDUSTRY),
        )

    def attach(
        self,
        panel: Frame,
        scheme: str,
        level: int = 1,
        *,
        column: str = INDUSTRY,
        ts: str = "ts",
    ) -> pl.DataFrame:
        """Left-join the label in force at each row's own timestamp.

        This is an as-of join on ``(symbol, date)``, not a broadcast of one
        snapshot. Symbols with no interval covering their timestamp get a
        **null** label rather than being dropped, keeping the panel
        rectangular so cross-sectional operations still align -- a dropped row
        would quietly shrink one date's cross-section and bias its
        standardisation.
        """
        lf = to_polars(panel)
        require_columns(lf, (ts, SYMBOL), where="Classification.attach")
        rows = self._select(scheme, level).select(
            SYMBOL,
            "industry_code",
            pl.col("industry_name").alias(column),
            "start_date",
            "end_date",
        )
        d = pl.col(ts).cast(pl.Date)
        return (
            lf.with_columns(d.alias("_d"))
            .join(rows, on=SYMBOL, how="left")
            .filter(
                pl.col("start_date").is_null()
                | (
                    (pl.col("start_date") <= pl.col("_d"))
                    & (pl.col("end_date").is_null() | (pl.col("end_date") > pl.col("_d")))
                )
            )
            .drop("_d", "start_date", "end_date")
        )

    def _select(self, scheme: str, level: int) -> pl.DataFrame:
        lf = self.frame.filter((pl.col("scheme") == scheme) & (pl.col("level") == level))
        if lf.is_empty():
            raise UniverseError(
                f"no rows for scheme {scheme!r} level {level}. Available: "
                f"{sorted(set(zip(self.frame['scheme'], self.frame['level'], strict=True)))}"
            )
        return lf

    @classmethod
    def from_frame(cls, df: Frame) -> Classification:
        """Build from a vendor change history that already carries dates."""
        lf = to_polars(df)
        require_columns(lf, _CLASSIFICATION_COLS, where="Classification.from_frame")
        return cls(
            lf.with_columns(
                pl.col(SYMBOL).cast(pl.String),
                pl.col("scheme").cast(pl.String),
                pl.col("level").cast(pl.Int32),
                pl.col("industry_code").cast(pl.String),
                pl.col("start_date").cast(pl.Date),
                pl.col("end_date").cast(pl.Date),
            ).sort("scheme", "level", SYMBOL, "start_date")
        )

    def __repr__(self) -> str:
        n_sym = self.frame[SYMBOL].n_unique()
        return (
            f"Classification(schemes={self.schemes}, symbols={n_sym}, "
            f"intervals={self.frame.height})"
        )


def classification_from_snapshots(snapshots: Frame) -> Classification:
    """Fold dated snapshots into effective-dated intervals.

    ``snapshots`` is long: ``asof``, ``scheme``, ``level``, ``symbol``,
    ``industry_code``, ``industry_name`` -- one row per symbol per level per
    observation date, as written by ``research/fetch_industry.py``.

    A symbol's interval starts at the ``asof`` where its label was first seen
    (or first seen *again* after changing) and ends at the ``asof`` where a
    different label appeared. The last interval is left open.

    Two properties worth being explicit about, because both are limitations
    rather than features:

    * **Change dates are snapshot dates.** The reclassification happened
      somewhere in the preceding gap; this dates it at the observation. It is
      never *early*, which is the safe direction for point-in-time.
    * **A symbol missing from a snapshot ends its interval there.** Delisting
      and a truncated fetch look identical from here, so the fetcher refuses to
      write a short snapshot rather than let this function read a gap as a
      market-wide exit.
    """
    lf = to_polars(snapshots)
    require_columns(lf, _SNAPSHOT_COLS, where="classification_from_snapshots")
    lf = lf.with_columns(
        pl.col("asof").cast(pl.Date),
        pl.col("scheme").cast(pl.String),
        pl.col("level").cast(pl.Int32),
        pl.col(SYMBOL).cast(pl.String),
        pl.col("industry_code").cast(pl.String),
        pl.col("industry_name").cast(pl.String),
    )

    dupes = lf.group_by("asof", "scheme", "level", SYMBOL).len().filter(pl.col("len") > 1)
    if dupes.height:
        raise UniverseError(
            f"{dupes.height} symbols carry two industries within one snapshot. "
            "A classification level must be single-valued per symbol; fix the source "
            "rather than picking one arbitrarily."
        )

    key = ["scheme", "level", SYMBOL]
    lf = lf.sort([*key, "asof"])
    # A new interval opens whenever the code differs from the previous snapshot
    # for this symbol -- including the first observation, where prev is null.
    lf = lf.with_columns(
        (pl.col("industry_code") != pl.col("industry_code").shift(1).over(key))
        .fill_null(True)
        .alias("_new")
    ).with_columns(pl.col("_new").cum_sum().over(key).alias("_run"))

    runs = lf.group_by([*key, "_run"]).agg(
        pl.col("industry_code").first(),
        pl.col("industry_name").last(),
        pl.col("asof").min().alias("start_date"),
        pl.col("asof").max().alias("_last_seen"),
    )

    # The end of a run is the start of the next one for the same symbol. The
    # final run stays open. Note the gap semantics: a symbol absent from later
    # snapshots keeps an open interval, because "vanished from a free endpoint"
    # is not evidence of delisting -- listings, not this table, own that.
    runs = runs.sort([*key, "start_date"]).with_columns(
        pl.col("start_date").shift(-1).over(key).alias("end_date")
    )

    return Classification.from_frame(
        runs.select(
            "scheme", "level", SYMBOL, "industry_code", "industry_name", "start_date", "end_date"
        )
    )


def _min_date(frame: pl.DataFrame) -> dt.date | None:
    """Earliest ``start_date``, typed. ``Series.min`` is declared far wider."""
    value = frame["start_date"].min()
    return value if isinstance(value, dt.date) else None


def _as_date(d: dt.date | str) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))
