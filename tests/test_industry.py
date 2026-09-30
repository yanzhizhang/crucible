"""Industry classification: effective dating, and the reclassification trap.

The failure this guards is silent by construction -- a backwards-applied
industry label makes neutralisation *look* better -- so every assertion here is
about a refusal or an exact interval boundary, not about a number being close.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from almanac.industry import Classification, classification_from_snapshots
from crucible.errors import UniverseError


def snapshots(rows: list[tuple[str, str, str, str]]) -> pl.DataFrame:
    """``(asof, symbol, industry_code, industry_name)`` -> snapshot frame."""
    return pl.DataFrame(
        {
            "asof": [dt.date.fromisoformat(r[0]) for r in rows],
            "scheme": ["em"] * len(rows),
            "level": [1] * len(rows),
            "symbol": [r[1] for r in rows],
            "industry_code": [r[2] for r in rows],
            "industry_name": [r[3] for r in rows],
        }
    )


TWO_SNAPS = snapshots(
    [
        ("2024-01-31", "000001", "BK0475", "银行"),
        ("2024-01-31", "600519", "BK0438", "酿酒行业"),
        ("2024-01-31", "300750", "BK0459", "电池"),
        # 300750 reclassified by the February observation; the other two hold.
        ("2024-02-29", "000001", "BK0475", "银行"),
        ("2024-02-29", "600519", "BK0438", "酿酒行业"),
        ("2024-02-29", "300750", "BK1037", "汽车零部件"),
    ]
)


# --------------------------------------------------------------------------
# snapshots -> intervals
# --------------------------------------------------------------------------


def test_unchanged_symbol_gets_one_open_interval() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    rows = cls.frame.filter(pl.col("symbol") == "000001")
    assert rows.height == 1
    assert rows["start_date"][0] == dt.date(2024, 1, 31)
    assert rows["end_date"][0] is None


def test_reclassification_splits_into_adjacent_half_open_intervals() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    rows = cls.frame.filter(pl.col("symbol") == "300750").sort("start_date")
    assert rows["industry_code"].to_list() == ["BK0459", "BK1037"]
    assert rows["start_date"].to_list() == [dt.date(2024, 1, 31), dt.date(2024, 2, 29)]
    # The old interval ends exactly where the new one starts: no overlap, no gap.
    assert rows["end_date"][0] == rows["start_date"][1]
    assert rows["end_date"][1] is None


def test_label_does_not_leak_backwards_across_the_change() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    before = cls.at("2024-02-28", "em")
    on = cls.at("2024-02-29", "em")
    assert before.filter(pl.col("symbol") == "300750")["industry"][0] == "电池"
    assert on.filter(pl.col("symbol") == "300750")["industry"][0] == "汽车零部件"


def test_change_back_and_forth_yields_three_intervals() -> None:
    cls = classification_from_snapshots(
        snapshots(
            [
                ("2024-01-31", "300750", "BK0459", "电池"),
                ("2024-02-29", "300750", "BK1037", "汽车零部件"),
                ("2024-03-29", "300750", "BK0459", "电池"),
            ]
        )
    )
    assert cls.frame.height == 3


def test_two_industries_in_one_snapshot_is_rejected() -> None:
    dupe = snapshots(
        [
            ("2024-01-31", "000001", "BK0475", "银行"),
            ("2024-01-31", "000001", "BK0473", "证券"),
        ]
    )
    with pytest.raises(UniverseError, match="single-valued"):
        classification_from_snapshots(dupe)


# --------------------------------------------------------------------------
# point-in-time lookups
# --------------------------------------------------------------------------


def test_lookup_before_first_snapshot_raises_rather_than_back_casting() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    with pytest.raises(UniverseError, match="earliest snapshot"):
        cls.at("2023-12-31", "em")


def test_unknown_scheme_or_level_raises() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    with pytest.raises(UniverseError, match="no rows for scheme"):
        cls.at("2024-02-29", "citic")
    with pytest.raises(UniverseError, match="no rows for scheme"):
        cls.at("2024-02-29", "em", level=3)


def test_null_start_date_is_refused() -> None:
    frame = pl.DataFrame(
        {
            "scheme": ["em"],
            "level": [1],
            "symbol": ["000001"],
            "industry_code": ["BK0475"],
            "industry_name": ["银行"],
            "start_date": [None],
            "end_date": [None],
        },
        schema_overrides={"start_date": pl.Date, "end_date": pl.Date},
    )
    with pytest.raises(UniverseError, match="null start_date"):
        Classification.from_frame(frame)


def test_overlapping_intervals_for_one_symbol_are_refused() -> None:
    frame = pl.DataFrame(
        {
            "scheme": ["em", "em"],
            "level": [1, 1],
            "symbol": ["000001", "000001"],
            "industry_code": ["BK0475", "BK0473"],
            "industry_name": ["银行", "证券"],
            "start_date": [dt.date(2024, 1, 31)] * 2,
            "end_date": [None, None],
        },
        schema_overrides={"end_date": pl.Date},
    )
    with pytest.raises(UniverseError, match="more than once"):
        Classification.from_frame(frame)


# --------------------------------------------------------------------------
# attaching to a panel
# --------------------------------------------------------------------------


def panel() -> pl.DataFrame:
    dates = [dt.date(2024, 2, 28), dt.date(2024, 3, 1)]
    return pl.DataFrame(
        {
            "ts": [dt.datetime.combine(d, dt.time()) for d in dates for _ in range(2)],
            "symbol": ["300750", "999999"] * 2,
            "factor": [0.1, 0.2, 0.3, 0.4],
        }
    )


def test_attach_uses_the_label_in_force_at_each_row_timestamp() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    out = cls.attach(panel(), "em").sort("ts", "symbol")
    got = dict(zip(out["ts"].to_list(), out["industry"].to_list(), strict=False))
    assert out.filter(pl.col("symbol") == "300750").sort("ts")["industry"].to_list() == [
        "电池",
        "汽车零部件",
    ]
    assert got  # both timestamps present


def test_attach_keeps_unclassified_symbols_as_null_rows() -> None:
    """Dropping them would shrink one date's cross-section and bias standardisation."""
    cls = classification_from_snapshots(TWO_SNAPS)
    out = cls.attach(panel(), "em")
    assert out.height == 4
    unknown = out.filter(pl.col("symbol") == "999999")
    assert unknown.height == 2
    assert unknown["industry"].null_count() == 2


def test_attach_never_duplicates_a_panel_row() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    p = panel()
    assert cls.attach(p, "em").height == p.height


def test_coverage_reports_open_ended_history() -> None:
    cls = classification_from_snapshots(TWO_SNAPS)
    start, end = cls.coverage("em")
    assert start == dt.date(2024, 1, 31)
    assert end is None
