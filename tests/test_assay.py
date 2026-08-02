"""Step 2 gate: IC must be monotone in the planted signal strength."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest

from almanac.masks import apply_masks, build_masks
from assay.cross_sectional import decay, ic, ic_summary, quantile_returns, turnover
from assay.time_series import (
    calmar,
    conditional_return_by_quantile,
    drawdown_duration,
    max_drawdown,
    monthly_winrate,
    parameter_surface,
    performance,
    rolling_corr,
    sharpe,
)
from horizon.labels import forward_return
from quarry.loaders import load_daily, load_factor_frame
from quarry.synth import PLANTED_FACTORS


@pytest.fixture(scope="module")
def panel(conn) -> pl.DataFrame:
    """Masked factor/label panel aligned the way the fixture planted the signal.

    The synthetic factors were built against the close-to-close return from
    ``t`` to ``t+1``, so the label here uses ``entry_lag=0`` and a close basis
    to match. That pairing is a *diagnostic* choice for validating the
    estimator against known truth -- it is not a tradable label, which is
    exactly why :func:`forward_return` warns about it.
    """
    daily = load_daily(conn)
    factors = load_factor_frame(conn)

    masks = build_masks(daily)
    daily = apply_masks(daily, masks, columns=["close", "vwap", "volume"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        labelled = forward_return(daily, n=1, price="close", entry_lag=0, label="y")

    return labelled.join(factors, on=["ts", "symbol"], how="inner")


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def test_ic_is_monotone_in_planted_signal_strength(panel) -> None:
    """THE STEP 2 GATE.

    Factors were planted as ``target_ic * signal + noise`` with a known ladder
    of strengths. A correct IC estimator must recover that ordering.
    """
    ladder = ["alpha_strong", "alpha_mid", "alpha_weak", "alpha_none"]
    got = [ic(panel, f, "y", min_names=20).mean for f in ladder]

    assert got == sorted(got, reverse=True), dict(zip(ladder, got))
    assert got[0] > 0.15, f"strong factor recovered only IC={got[0]:.3f}"
    assert abs(got[-1]) < 0.08, f"pure-noise factor showed IC={got[-1]:.3f}"


def test_ic_sign_flips_with_the_factor(panel) -> None:
    flipped = panel.with_columns(neg=-pl.col("alpha_strong"))
    assert ic(panel, "alpha_strong", "y").mean == pytest.approx(
        -ic(flipped, "neg", "y").mean, abs=1e-12
    )


def test_ic_carries_sample_counts(panel) -> None:
    """A result must expose the breadth it was computed on."""
    r = ic(panel, "alpha_strong", "y")
    assert r.n_periods > 0
    assert r.mean_breadth > 0
    assert "n" in r.series.columns
    assert r.is_thin  # one month of data -- the flag must say so


def test_thin_cross_sections_are_dropped(panel) -> None:
    """A 4-name cross-section yields a number, not information."""
    r = ic(panel, "alpha_strong", "y", min_names=10_000)
    assert r.n_periods == 0
    assert r.mean == 0.0


def test_spearman_is_robust_to_an_outlier(panel) -> None:
    """Rank IC must barely move when one value is made enormous."""
    spoiled = panel.with_columns(
        alpha_strong=pl.when(pl.int_range(pl.len()) == 0)
        .then(1e12)
        .otherwise(pl.col("alpha_strong"))
    )
    base = ic(panel, "alpha_strong", "y", method="spearman").mean
    hurt = ic(spoiled, "alpha_strong", "y", method="spearman").mean
    assert abs(base - hurt) < 0.05

    p_base = ic(panel, "alpha_strong", "y", method="pearson").mean
    p_hurt = ic(spoiled, "alpha_strong", "y", method="pearson").mean
    assert abs(p_base - p_hurt) > abs(base - hurt)


def test_newey_west_reduces_t_stat_on_autocorrelated_ic() -> None:
    """Overlapping labels inflate the naive t-stat; the correction must bite."""
    rng = np.random.default_rng(0)
    n = 400
    e = rng.normal(0, 1, n)
    # Strongly positively autocorrelated IC series with a positive mean.
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = 0.8 * x[i - 1] + e[i]
    x = x * 0.02 + 0.03

    series = pl.DataFrame({"ts": range(n), "ic": x, "n": [50] * n})
    naive = ic_summary(series).t_stat
    corrected = ic_summary(series, newey_west_lags=10).t_stat
    assert abs(corrected) < abs(naive)


def test_ic_method_validation(panel) -> None:
    with pytest.raises(ValueError, match="spearman.*pearson"):
        ic(panel, "alpha_strong", "y", method="kendall")


# --------------------------------------------------------------------------
# quantiles
# --------------------------------------------------------------------------


def test_quantile_spread_is_positive_for_a_real_factor(panel) -> None:
    q = quantile_returns(panel, "alpha_strong", "y", n=5)
    assert q.spread_mean > 0
    assert q.n_periods > 0


def test_quantile_monotonicity_ranks_factors(panel) -> None:
    """A real factor's buckets should be ordered; noise should not be."""
    strong = quantile_returns(panel, "alpha_strong", "y", n=5)
    noise = quantile_returns(panel, "alpha_none", "y", n=5)
    assert strong.monotonicity > noise.monotonicity
    assert strong.monotonicity > 0.5


def test_top_bucket_beats_bottom(panel) -> None:
    q = quantile_returns(panel, "alpha_strong", "y", n=5)
    means = q.by_bucket.sort("bucket")["mean_ret"].to_list()
    assert means[-1] > means[0]


def test_quantile_buckets_are_within_period(panel) -> None:
    """Every period contributes to every bucket; ranks are not pooled."""
    q = quantile_returns(panel, "alpha_strong", "y", n=5)
    per_period = q.curves.group_by("ts").agg(pl.col("bucket").n_unique().alias("k"))
    assert per_period["k"].min() == 5


def test_quantile_rejects_degenerate_bucket_count(panel) -> None:
    with pytest.raises(ValueError, match="at least 2 buckets"):
        quantile_returns(panel, "alpha_strong", "y", n=1)


# --------------------------------------------------------------------------
# decay and turnover
# --------------------------------------------------------------------------


def test_decay_peaks_at_the_planted_horizon(conn) -> None:
    """Signal was planted at horizon 1, so IC must peak there and fall away."""
    daily = load_daily(conn)
    factors = load_factor_frame(conn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        lab = daily
        for h in (1, 2, 3, 5):
            lab = forward_return(lab, n=h, price="close", entry_lag=0, label=f"y{h}")
    panel = lab.join(factors, on=["ts", "symbol"], how="inner")

    d = decay(panel, "alpha_strong", {1: "y1", 2: "y2", 3: "y3", 5: "y5"})
    assert d.peak_horizon == 1
    curve = d.curve.sort("horizon")["ic"].to_list()
    assert curve[0] > curve[-1]


def test_decay_requires_labels() -> None:
    with pytest.raises(ValueError, match="labels_by_horizon is empty"):
        decay(pl.DataFrame({"ts": [], "symbol": [], "f": []}), "f", {})


def test_turnover_of_a_static_factor_is_zero(panel) -> None:
    """A factor that never changes rank has no turnover."""
    static = panel.with_columns(
        const=pl.col("symbol").str.slice(-2).cast(pl.Int32).cast(pl.Float64)
    )
    t = turnover(static, "const")
    assert t.mean == pytest.approx(0.0, abs=1e-9)


def test_turnover_of_noise_is_high(panel) -> None:
    rng = np.random.default_rng(7)
    noisy = panel.with_columns(
        rand=pl.Series("rand", rng.normal(size=panel.height))
    )
    t = turnover(noisy, "rand")
    assert t.mean > 0.7
    assert t.implied_holding_periods < 2.0


def test_turnover_matches_symbols_not_positions(panel) -> None:
    """Universe size changes must not register as turnover."""
    t_full = turnover(panel, "alpha_strong")
    dropped = panel.filter(
        ~((pl.col("ts") == panel["ts"].max()) & (pl.col("symbol") == panel["symbol"].min()))
    )
    t_short = turnover(dropped, "alpha_strong")
    assert abs(t_full.mean - t_short.mean) < 0.05


# --------------------------------------------------------------------------
# time-series metrics
# --------------------------------------------------------------------------


def test_sharpe_of_constant_positive_returns_is_infinite_free() -> None:
    """Zero variance must not divide by zero."""
    assert sharpe(np.full(100, 0.001)) == 0.0


def test_sharpe_scales_with_annualisation() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(0.0005, 0.01, 1000)
    daily = sharpe(r, periods_per_year=244)
    minute = sharpe(r, periods_per_year=244 * 240)
    # Compare magnitudes: annualisation scales by sqrt(k) regardless of sign,
    # and this sample's mean can land either side of zero.
    assert abs(minute) > abs(daily)
    assert minute / daily == pytest.approx(np.sqrt(240), rel=1e-9)


def test_max_drawdown_is_compounded() -> None:
    r = np.array([0.5, -0.5, 0.5, -0.5])
    mdd = max_drawdown(r)
    assert mdd < 0
    # equity: 1.5, 0.75, 1.125, 0.5625 -> trough vs peak 1.5
    assert mdd == pytest.approx(0.5625 / 1.5 - 1.0)


def test_max_drawdown_of_monotone_gains_is_zero() -> None:
    assert max_drawdown(np.full(50, 0.01)) == pytest.approx(0.0)


def test_drawdown_duration_counts_underwater_periods() -> None:
    r = np.array([0.1, -0.05, -0.05, 0.20, 0.0])
    assert drawdown_duration(r) == 2


def test_calmar_relates_return_to_drawdown() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.001, 0.01, 500)
    c = calmar(r, periods_per_year=244)
    assert np.isfinite(c)


def test_performance_flags_thin_samples() -> None:
    rng = np.random.default_rng(5)
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 3, 1), "1d", eager=True
            ),
        }
    )
    df = df.with_columns(ret=pl.Series("ret", rng.normal(0.001, 0.01, df.height)))
    p = performance(df, periods_per_year=244)
    assert p.is_thin
    assert p.n == df.height
    assert "sharpe" in p.summary()


def test_monthly_winrate_bounds() -> None:
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2023, 1, 1), pl.datetime(2023, 12, 31), "1d", eager=True
            ),
        }
    )
    df = df.with_columns(ret=pl.lit(0.001))
    assert monthly_winrate(df) == 1.0


def test_rolling_corr_is_trailing() -> None:
    """Leading rows are null; a centred window would read the future."""
    df = pl.DataFrame(
        {
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1), pl.datetime(2024, 2, 20), "1d", eager=True
            ),
        }
    )
    rng = np.random.default_rng(11)
    df = df.with_columns(
        a=pl.Series("a", rng.normal(size=df.height)),
        b=pl.Series("b", rng.normal(size=df.height)),
    )
    out = rolling_corr(df, "a", "b", window=10)
    assert out["corr"][:9].null_count() == 9
    assert out["corr"][9] is not None


def test_conditional_return_by_quantile_shape() -> None:
    rng = np.random.default_rng(13)
    n = 500
    sig = rng.normal(size=n)
    df = pl.DataFrame({"sig": sig, "y": 0.3 * sig + rng.normal(0, 1, n)})
    out = conditional_return_by_quantile(df, "sig", "y", n=5)
    assert out.height == 5
    assert out["mean_ret"][4] > out["mean_ret"][0]


# --------------------------------------------------------------------------
# overfitting detection
# --------------------------------------------------------------------------


def test_parameter_surface_detects_an_isolated_peak() -> None:
    """A spike surrounded by nothing is overfit, and must be called out."""

    def spike(window: int, threshold: float) -> float:
        return 5.0 if (window == 20 and threshold == 0.02) else 0.1

    r = parameter_surface(
        {"window": [10, 15, 20, 25, 30], "threshold": [0.01, 0.02, 0.03]}, spike
    )
    assert r.best_score == 5.0
    assert r.is_isolated_peak
    assert r.plateau_score < 0.1


def test_parameter_surface_accepts_a_plateau() -> None:
    """A real edge degrades gently as parameters move."""

    def smooth(window: int, threshold: float) -> float:
        return 2.0 - abs(window - 20) * 0.01 - abs(threshold - 0.02) * 2.0

    r = parameter_surface(
        {"window": [10, 15, 20, 25, 30], "threshold": [0.01, 0.02, 0.03]}, smooth
    )
    assert not r.is_isolated_peak
    assert r.plateau_score > 0.9


def test_parameter_surface_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="grid is empty"):
        parameter_surface({}, lambda: 0.0)
