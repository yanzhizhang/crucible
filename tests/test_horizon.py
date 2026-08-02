"""Step 2 gate: labels are point-in-time and gap-safe."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from crucible.errors import PointInTimeError
from horizon.labels import (
    assert_no_overlap,
    forward_return,
    slot_index,
    ternary_label,
    vol_normalized_return,
)


def _panel(symbol_slots: dict[str, list[int]], price: float = 10.0) -> pl.DataFrame:
    """Long frame where symbol -> the slot numbers it actually has rows for."""
    rows = []
    for sym, slots in symbol_slots.items():
        for s in slots:
            rows.append(
                {
                    "ts": dt.datetime(2024, 1, 1, 15, 0) + dt.timedelta(days=s),
                    "symbol": sym,
                    "vwap": price * (1.0 + 0.01 * s),
                }
            )
    return pl.DataFrame(rows).sort(["ts", "symbol"])


def test_slot_index_is_global_not_per_symbol() -> None:
    """Index arithmetic must mean the same thing for every instrument."""
    df = _panel({"A": [0, 1, 2, 3], "B": [0, 1, 3]})
    out = slot_index(df)
    b3 = out.filter((pl.col("symbol") == "B") & (pl.col("_slot") == 3))
    assert b3.height == 1  # B keeps global slot 3 despite missing slot 2


def test_forward_return_basic_arithmetic() -> None:
    """entry at t+1, exit at t+1+n."""
    df = _panel({"A": [0, 1, 2, 3]})
    out = forward_return(df, n=1, price="vwap", entry_lag=1)

    # Row at slot 0: entry = vwap[1] = 10.10, exit = vwap[2] = 10.20
    got = out.sort("ts")["fwd_ret"][0]
    assert got == pytest.approx(10.20 / 10.10 - 1.0)


def test_forward_return_is_null_across_a_gap() -> None:
    """THE no-shift test.

    B is missing slot 2. Its label at slot 1 needs the price at slot 2 and must
    therefore be null. ``shift(-1)`` would instead hand back slot 3's price --
    a wrong answer that looks entirely reasonable.
    """
    df = _panel({"A": [0, 1, 2, 3], "B": [0, 1, 3]})
    out = forward_return(df, n=1, price="vwap", entry_lag=0).sort(["symbol", "ts"])

    b = out.filter(pl.col("symbol") == "B").sort("ts")
    # B rows are slots 0, 1, 3. The label at slot 1 needs slot 2 -> null.
    assert b["fwd_ret"][1] is None

    a = out.filter(pl.col("symbol") == "A").sort("ts")
    assert a["fwd_ret"][1] is not None  # A has slot 2, so its label exists


def test_forward_return_tail_is_null() -> None:
    """The last rows have no future and must not wrap or extrapolate."""
    df = _panel({"A": [0, 1, 2, 3]})
    out = forward_return(df, n=1, entry_lag=1).sort("ts")
    assert out["fwd_ret"][-1] is None
    assert out["fwd_ret"][-2] is None  # needs slots 3 and 4; 4 does not exist


def test_entry_lag_shifts_the_window() -> None:
    df = _panel({"A": [0, 1, 2, 3]})
    lagged = forward_return(df, n=1, entry_lag=1).sort("ts")["fwd_ret"][0]
    immediate = forward_return(df, n=1, entry_lag=0).sort("ts")["fwd_ret"][0]

    assert immediate == pytest.approx(10.10 / 10.00 - 1.0)
    assert lagged == pytest.approx(10.20 / 10.10 - 1.0)
    assert lagged != immediate


def test_close_to_close_warns() -> None:
    df = _panel({"A": [0, 1, 2]}).rename({"vwap": "close"})
    with pytest.warns(UserWarning, match="closing print"):
        forward_return(df, n=1, price="close")


def test_vwap_default_does_not_warn() -> None:
    df = _panel({"A": [0, 1, 2]})
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        forward_return(df, n=1)


def test_ternary_deadband() -> None:
    df = _panel({"A": [0, 1, 2, 3]})
    out = ternary_label(df, n=1, threshold=0.5, entry_lag=0).sort("ts")
    # Moves are ~1%, far inside a 50% deadband -> all neutral.
    labels = [v for v in out["y3"].to_list() if v is not None]
    assert labels and set(labels) == {0}

    tight = ternary_label(df, n=1, threshold=0.0001, entry_lag=0).sort("ts")
    tight_labels = [v for v in tight["y3"].to_list() if v is not None]
    assert set(tight_labels) == {1}  # prices rise monotonically


def test_ternary_rejects_negative_threshold() -> None:
    df = _panel({"A": [0, 1, 2]})
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        ternary_label(df, threshold=-0.01)


def test_vol_normalized_uses_trailing_vol(store, conn) -> None:
    """Early rows are null because trailing vol is not yet estimable."""
    from quarry.loaders import load_daily

    daily = load_daily(conn)
    out = vol_normalized_return(daily, n=1, price="vwap", vol_window=5)
    first_sym = out.filter(pl.col("symbol") == store.symbols[0]).sort("ts")
    assert first_sym["fwd_ret_vol"][0] is None
    assert first_sym["fwd_ret_vol"].drop_nulls().len() > 0


def test_forward_return_rejects_bad_horizon() -> None:
    df = _panel({"A": [0, 1]})
    with pytest.raises(ValueError, match="n must be >= 1"):
        forward_return(df, n=0)


def test_assert_no_overlap_rejects_zero_entry_lag() -> None:
    df = _panel({"A": [0, 1, 2]})
    with pytest.raises(PointInTimeError, match="cannot be traded"):
        assert_no_overlap(df, df, n=1, entry_lag=0)


def test_assert_no_overlap_catches_disjoint_keys() -> None:
    a = _panel({"A": [0, 1, 2]})
    b = _panel({"Z": [90, 91, 92]})
    with pytest.raises(PointInTimeError, match="share no \\(ts, symbol\\) keys"):
        assert_no_overlap(a, b, n=1, entry_lag=1)


def test_assert_no_overlap_passes_on_valid_pairing() -> None:
    df = _panel({"A": [0, 1, 2]})
    assert_no_overlap(df, forward_return(df, n=1), n=1, entry_lag=1)


def test_pandas_in_pandas_out() -> None:
    """The frame bridge returns the flavor it was given."""
    import pandas as pd

    df = _panel({"A": [0, 1, 2]}).to_pandas()
    out = forward_return(df, n=1)
    assert isinstance(out, pd.DataFrame)
    assert "fwd_ret" in out.columns
