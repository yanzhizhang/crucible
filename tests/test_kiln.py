"""Step 4 gate: the leakage canary must be caught, not silently passed."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from crucible.errors import LeakageError
from horizon.labels import forward_return
from kiln.models import (
    assert_no_feature_leakage,
    noise_benchmark,
    phase_randomize,
    walk_forward,
)
from kiln.splits import CombinatorialPurgedCV, PurgedWalkForward, assert_no_leakage
from quarry.loaders import load_daily, load_factor_frame


@pytest.fixture(scope="module")
def panel(conn) -> pl.DataFrame:
    daily = load_daily(conn)
    ff = load_factor_frame(conn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lab = forward_return(daily, n=1, price="close", entry_lag=0, label="y")
    return lab.join(ff, on=["ts", "symbol"], how="inner").drop_nulls("y")


def _stamps(n_periods: int, n_names: int = 10) -> np.ndarray:
    return np.repeat(np.arange(n_periods), n_names)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_cross_sections_are_never_split() -> None:
    """All names at one timestamp are one sample."""
    ts = _stamps(60)
    cv = PurgedWalkForward(train_span=20, test_span=5, embargo=2, label_horizon=1)
    for tr, te in cv.split(ts):
        assert not set(ts[tr]) & set(ts[te])


def test_purge_removes_overlapping_labels() -> None:
    """No training timestamp may sit within label_horizon of the test start."""
    ts = _stamps(80)
    cv = PurgedWalkForward(train_span=30, test_span=10, embargo=0, label_horizon=5)
    for tr, te in cv.split(ts):
        te_lo = ts[te].min()
        assert (ts[tr] < te_lo - 5).all() or (ts[tr] >= ts[te].max()).all()


def test_embargo_is_respected() -> None:
    ts = _stamps(80)
    cv = PurgedWalkForward(
        train_span=30, test_span=10, embargo=3, label_horizon=1, expanding=True
    )
    for tr, te in cv.split(ts):
        assert_no_leakage(ts, tr, te, label_horizon=1, embargo=3)


def test_walk_forward_is_deterministic() -> None:
    ts = _stamps(60)
    cv = PurgedWalkForward(train_span=20, test_span=5)
    a = [(tr.tolist(), te.tolist()) for tr, te in cv.split(ts)]
    b = [(tr.tolist(), te.tolist()) for tr, te in cv.split(ts)]
    assert a == b


def test_short_sample_raises_rather_than_yielding_nothing() -> None:
    """An empty iterator would read as a clean validation."""
    ts = _stamps(10)
    cv = PurgedWalkForward(train_span=20, test_span=5)
    with pytest.raises(ValueError, match="unique timestamps"):
        list(cv.split(ts))


def test_split_parameter_validation() -> None:
    with pytest.raises(ValueError, match="train_span must be >= 1"):
        PurgedWalkForward(train_span=0, test_span=5)
    with pytest.raises(ValueError, match="embargo must be >= 0"):
        PurgedWalkForward(train_span=10, test_span=5, embargo=-1)


def test_expanding_window_grows() -> None:
    ts = _stamps(80)
    cv = PurgedWalkForward(train_span=20, test_span=10, expanding=True)
    sizes = [len(tr) for tr, _ in cv.split(ts)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_combinatorial_cv_yields_many_paths() -> None:
    ts = _stamps(60)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=1, label_horizon=1)
    folds = list(cv.split(ts))
    assert len(folds) == cv.n_splits() == 15
    for tr, te in folds:
        assert not set(ts[tr]) & set(ts[te])


def test_combinatorial_cv_validates_group_counts() -> None:
    with pytest.raises(ValueError, match="n_test_groups must be in"):
        CombinatorialPurgedCV(n_groups=4, n_test_groups=4)


# --------------------------------------------------------------------------
# leakage assertions
# --------------------------------------------------------------------------


def test_assert_no_leakage_catches_a_shared_timestamp() -> None:
    ts = _stamps(20)
    train = np.flatnonzero(ts < 12)
    test = np.flatnonzero(ts >= 10)  # overlaps 10 and 11
    with pytest.raises(LeakageError, match="both train and test"):
        assert_no_leakage(ts, train, test)


def test_assert_no_leakage_catches_label_overlap() -> None:
    ts = _stamps(30)
    train = np.flatnonzero(ts < 15)
    test = np.flatnonzero((ts >= 15) & (ts < 20))
    # A 5-period label at t=14 observes through t=19, inside the test window.
    with pytest.raises(LeakageError, match="label_horizon"):
        assert_no_leakage(ts, train, test, label_horizon=5)


def test_assert_no_leakage_catches_embargo_breach() -> None:
    """Isolate the embargo check: the purge zone is already excluded.

    Training stops at t=8 so nothing sits within label_horizon=1 of the test
    start, which would otherwise trip the label-overlap check first and mask
    the embargo violation being tested.
    """
    ts = _stamps(40)
    test = np.flatnonzero((ts >= 10) & (ts < 15))
    train = np.flatnonzero((ts < 9) | ((ts >= 15) & (ts < 20)))
    with pytest.raises(LeakageError, match="embargo"):
        assert_no_leakage(ts, train, test, label_horizon=1, embargo=3)


def test_assert_no_leakage_passes_a_clean_split() -> None:
    ts = _stamps(40)
    train = np.flatnonzero(ts < 10)
    test = np.flatnonzero((ts >= 15) & (ts < 20))
    assert_no_leakage(ts, train, test, label_horizon=3, embargo=2)


def test_empty_side_is_not_a_pass() -> None:
    ts = _stamps(20)
    with pytest.raises(LeakageError, match="empty side"):
        assert_no_leakage(ts, np.array([], dtype=int), np.flatnonzero(ts < 5))


# --------------------------------------------------------------------------
# THE STEP 4 GATE -- the leakage canary
# --------------------------------------------------------------------------


def test_leakage_canary_is_caught_by_explicit_assertion(panel) -> None:
    """THE STEP 4 GATE.

    A "feature" that IS the label scores near-perfectly. The point is that this
    must be caught by an assertion that names it, not pass silently while
    producing a spectacular backtest.
    """
    poisoned = panel.with_columns(cheat=pl.col("y"))

    # It really does look perfect -- that is why silent failure is so dangerous.
    from assay.cross_sectional import ic

    assert ic(poisoned, "cheat", "y").mean > 0.99

    with pytest.raises(LeakageError, match="is the label, or derived from it"):
        assert_no_feature_leakage(poisoned, ["cheat"], "y")


def test_leakage_canary_survives_a_light_disguise(panel) -> None:
    """An affine transform of the label is still the label."""
    poisoned = panel.with_columns(cheat=pl.col("y") * -3.0 + 0.01)
    with pytest.raises(LeakageError, match="cheat"):
        assert_no_feature_leakage(poisoned, ["cheat"], "y")


def test_honest_features_pass_the_canary(panel) -> None:
    assert_no_feature_leakage(panel, ["alpha_strong", "alpha_mid", "alpha_none"], "y")


def test_canary_threshold_is_validated(panel) -> None:
    with pytest.raises(ValueError, match="max_abs_ic must be in"):
        assert_no_feature_leakage(panel, ["alpha_strong"], "y", max_abs_ic=1.5)


# --------------------------------------------------------------------------
# noise benchmark
# --------------------------------------------------------------------------


def test_phase_randomize_preserves_the_power_spectrum() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=512))
    y = phase_randomize(x, rng)

    px = np.abs(np.fft.rfft(x - x.mean()))
    py = np.abs(np.fft.rfft(y - y.mean()))
    np.testing.assert_allclose(px, py, rtol=1e-8, atol=1e-8)


def test_phase_randomize_returns_real_values() -> None:
    rng = np.random.default_rng(1)
    y = phase_randomize(rng.normal(size=256), rng)
    assert np.isrealobj(y)
    assert np.isfinite(y).all()


def test_noise_benchmark_rejects_a_null_strategy() -> None:
    """Scoring pure autocorrelation must not look significant."""
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=400))

    def score(series: np.ndarray) -> float:
        # A trend-following score that lives entirely off autocorrelation --
        # exactly what phase randomisation preserves.
        d = np.diff(series)
        return float(np.mean(d[1:] * np.sign(d[:-1])))

    r = noise_benchmark(x, score, n_trials=100, seed=3)
    assert not r.is_significant
    assert 0.0 < r.p_value <= 1.0


def test_noise_benchmark_p_value_is_never_zero() -> None:
    """Beating every surrogate reports 1/(n+1), not an impossible zero.

    Identity (``s is x``) cannot detect the real series: the benchmark filters
    non-finite values first and so passes a copy. Compare by content.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(size=128)
    r = noise_benchmark(
        x, lambda s: 1e9 if np.array_equal(s, x) else 0.0, n_trials=50, seed=5
    )
    assert r.p_value == pytest.approx(1 / 51)
    assert r.percentile == pytest.approx(100.0)
    assert r.is_significant


def test_noise_benchmark_needs_enough_data() -> None:
    with pytest.raises(ValueError, match="at least 16 observations"):
        noise_benchmark(np.arange(4.0), lambda s: 0.0)


# --------------------------------------------------------------------------
# walk-forward fitting
# --------------------------------------------------------------------------


def test_walk_forward_produces_oos_predictions(panel) -> None:
    cv = PurgedWalkForward(train_span=8, test_span=3, embargo=1, label_horizon=1)
    res = walk_forward(
        panel,
        ["alpha_strong", "alpha_mid", "alpha_weak", "alpha_none"],
        "y",
        cv,
        num_boost_round=20,
    )
    assert res.n_folds > 0
    assert res.predictions.height > 0
    assert set(res.predictions.columns) >= {"ts", "symbol", "pred", "y", "fold"}


def test_walk_forward_predictions_are_out_of_sample(panel) -> None:
    """No prediction may come from a fold that trained on its timestamp."""
    cv = PurgedWalkForward(train_span=8, test_span=3, embargo=1, label_horizon=1)
    res = walk_forward(panel, ["alpha_strong"], "y", cv, num_boost_round=10)
    per_fold = res.predictions.group_by("fold").agg(pl.col("ts").min().alias("lo"))
    assert per_fold.height == res.n_folds


def test_walk_forward_is_reproducible(panel) -> None:
    cv = PurgedWalkForward(train_span=8, test_span=3, label_horizon=1)
    a = walk_forward(panel, ["alpha_strong"], "y", cv, num_boost_round=15, seed=42)
    b = walk_forward(panel, ["alpha_strong"], "y", cv, num_boost_round=15, seed=42)
    assert a.predictions["pred"].to_list() == pytest.approx(b.predictions["pred"].to_list())
