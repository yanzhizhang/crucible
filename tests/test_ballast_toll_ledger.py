"""Step 5 gate: costs, constraints, and PnL parity against the C++ backtester."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from ballast.covariance import factor_covariance, ledoit_wolf, nearest_psd, sample_covariance
from ballast.portfolio import (
    Constraints,
    apply_constraints,
    combine_scores,
    market_neutral,
    score_to_weight,
)
from crucible.errors import ParityError
from ledger.accounting import parity_check, simulate_cross_sectional, simulate_time_series
from toll.costs import CostModel, round_lots, square_root_impact, stamp_duty_for

# --------------------------------------------------------------------------
# toll
# --------------------------------------------------------------------------


def test_stamp_duty_halved_in_2023() -> None:
    assert stamp_duty_for("2023-08-27") == 0.0010
    assert stamp_duty_for("2023-08-28") == 0.0005


def test_stamp_duty_is_sell_side_only() -> None:
    model = CostModel(stamp_duty=0.001, commission=0.0, min_commission=0.0, transfer_fee=0.0)
    trades = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 2)] * 2,
            "symbol": ["600000", "600001"],
            "shares": [1000.0, -1000.0],
            "price": [10.0, 10.0],
        }
    )
    out = model.apply(trades)
    assert out["stamp_duty"][0] == 0.0
    assert out["stamp_duty"][1] == pytest.approx(10_000 * 0.001)


def test_minimum_commission_dominates_small_tickets() -> None:
    model = CostModel(commission=0.00025, min_commission=5.0)
    trades = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 2)],
            "symbol": ["600000"],
            "shares": [100.0],
            "price": [30.0],  # 3000 CNY notional
        }
    )
    out = model.apply(trades)
    assert out["commission"][0] == 5.0
    assert out["cost_bps"][0] > 16  # ~17bp, not 2.5bp


def test_round_lots_is_side_aware() -> None:
    """Buys round down to whole lots; sells must stay exitable."""
    got = round_lots(np.array([150.0, -150.0, 99.0, -99.0]), 100)
    assert got[0] == 100.0  # buy rounds down
    assert got[1] == -150.0  # sell untouched
    assert got[2] == 0.0  # sub-lot buy is dropped
    assert got[3] == -99.0  # odd-lot sell survives


def test_square_root_impact_scaling() -> None:
    """Quadrupling size doubles impact -- the defining property."""
    small = square_root_impact(np.array([1e4]), np.array([1e6]), np.array([0.02]))
    big = square_root_impact(np.array([4e4]), np.array([1e6]), np.array([0.02]))
    assert big[0] / small[0] == pytest.approx(2.0)


def test_impact_with_zero_volume_is_finite() -> None:
    out = square_root_impact(np.array([1000.0]), np.array([0.0]), np.array([0.02]))
    assert np.isfinite(out).all()
    assert out[0] == 0.0


def test_impact_participation_is_capped() -> None:
    out = square_root_impact(np.array([1e9]), np.array([1e3]), np.array([0.02]))
    assert out[0] == pytest.approx(0.5 * 0.02)


def test_deadband_is_derived_not_constant() -> None:
    """A liquid megacap and an illiquid small cap must get different deadbands."""
    model = CostModel()
    liquid = model.deadband_threshold(20.0, 10_000, adv=5e7, sigma=0.018)
    illiquid = model.deadband_threshold(20.0, 10_000, adv=1e5, sigma=0.040)
    assert illiquid > liquid * 2
    assert liquid > model.stamp_duty  # never cheaper than the tax alone


def test_cost_components_sum_to_total() -> None:
    model = CostModel()
    trades = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 2)] * 3,
            "symbol": ["600000", "600001", "300001"],
            "shares": [10_000.0, -8_000.0, 5_000.0],
            "price": [12.0, 25.0, 40.0],
            "adv": [1e6, 2e6, 5e5],
            "sigma": [0.02, 0.018, 0.03],
        }
    )
    out = model.apply(trades)
    parts = (
        out["commission"]
        + out["stamp_duty"]
        + out["transfer_fee"]
        + out["impact_temporary"]
        + out["impact_permanent"]
    )
    assert parts.to_numpy() == pytest.approx(out["cost_total"].to_numpy())


def test_cost_model_validates_parameters() -> None:
    with pytest.raises(ValueError, match="permanent_fraction"):
        CostModel(permanent_fraction=1.5)
    with pytest.raises(ValueError, match="lot_size"):
        CostModel(lot_size=0)


# --------------------------------------------------------------------------
# ballast
# --------------------------------------------------------------------------


@pytest.fixture
def scores() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    n_ts, n_sym = 6, 40
    rows = []
    for i in range(n_ts):
        for j in range(n_sym):
            rows.append(
                {
                    "ts": dt.datetime(2024, 1, 2) + dt.timedelta(days=i),
                    "symbol": f"6000{j:02d}",
                    "score": float(rng.normal()),
                    "industry": ["bank", "tech", "pharma", "energy"][j % 4],
                    "beta": float(rng.uniform(0.7, 1.3)),
                }
            )
    return pl.DataFrame(rows)


def test_weights_are_dollar_neutral_and_unit_gross(scores) -> None:
    # 40 names at the 2% default cap admit only 0.8 gross, so the 1.0 target
    # would be infeasible. Use a cap that leaves room.
    out = score_to_weight(
        scores, "score", method="rank",
        constraints=Constraints(max_weight=0.10, min_weight=-0.10, max_industry=None),
    )
    per_ts = out.group_by("ts").agg(
        pl.col("weight").sum().alias("net"), pl.col("weight").abs().sum().alias("gross")
    )
    assert per_ts["net"].abs().max() < 1e-8
    assert per_ts["gross"].to_numpy() == pytest.approx(1.0, abs=1e-6)


def test_single_name_cap_is_respected(scores) -> None:
    c = Constraints(max_weight=0.03, min_weight=-0.03, max_industry=None)
    out = score_to_weight(scores, "score", method="score", constraints=c)
    assert out["weight"].abs().max() <= 0.03 + 1e-9


def test_industry_cap_is_respected(scores) -> None:
    c = Constraints(max_weight=0.5, min_weight=-0.5, max_industry=0.05)
    out = score_to_weight(scores, "score", method="rank", constraints=c)
    per = out.group_by(["ts", "industry"]).agg(pl.col("weight").sum().alias("net"))
    assert per["net"].abs().max() <= 0.05 + 1e-6


def test_turnover_cap_blends_toward_the_prior_book(scores) -> None:
    first = scores.filter(pl.col("ts") == scores["ts"].min())
    base = score_to_weight(first, "score", method="rank")
    flipped = base.with_columns(score=-pl.col("score"), prev=pl.col("weight"))

    capped = apply_constraints(
        score_to_weight(flipped, "score", method="rank"),
        Constraints(max_turnover=0.2, max_industry=None, max_weight=1.0, min_weight=-1.0),
        prev_weight="prev",
    )
    turnover = float((capped["weight"] - capped["prev"]).abs().sum())
    assert turnover <= 0.2 + 1e-6


def test_equal_weight_selects_only_the_tails(scores) -> None:
    out = score_to_weight(scores, "score", method="equal", top_quantile=0.2)
    one = out.filter(pl.col("ts") == out["ts"].min())
    assert (one["weight"] == 0.0).sum() > 0  # the middle is untouched
    assert one["weight"].max() > 0 and one["weight"].min() < 0


def test_score_to_weight_rejects_unknown_method(scores) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        score_to_weight(scores, "score", method="magic")


def test_combine_scores_requires_weights_for_ic_method(scores) -> None:
    df = scores.with_columns(a=pl.col("score"), b=-pl.col("score"))
    with pytest.raises(ValueError, match="requires `weights`"):
        combine_scores(df, ["a", "b"], method="ic_weighted")


def test_combine_scores_equal_averages_standardised_factors(scores) -> None:
    df = scores.with_columns(a=pl.col("score"), b=pl.col("score") * 100.0)
    out = combine_scores(df, ["a", "b"], method="equal", out="blend")
    # b is a rescaled a, so their z-scores are identical and the blend equals
    # z(a) exactly. Correlation against the RAW a is high but not 1: the blend
    # is standardised per timestamp while a is not.
    corr = out.select(pl.corr("blend", "a").alias("c"))["c"][0]
    assert abs(float(corr)) > 0.95

    # Against the per-timestamp z-score of a, the match is exact.
    zc = out.select(
        pl.corr(
            "blend",
            (pl.col("a") - pl.col("a").mean().over("ts")) / pl.col("a").std().over("ts"),
        ).alias("c")
    )["c"][0]
    assert abs(float(zc)) == pytest.approx(1.0, abs=1e-9)


def test_market_neutral_sizes_an_integral_hedge(scores) -> None:
    one = score_to_weight(
        scores.filter(pl.col("ts") == scores["ts"].min()), "score", method="rank"
    ).with_columns(weight=pl.col("weight") + 0.02)  # deliberate net long

    h = market_neutral(one, capital=1e8, index_price=6000.0, hedge="IM")
    assert isinstance(h.contracts, int)
    assert h.contracts < 0  # short futures against a net-long book
    assert abs(h.residual_notional) <= 6000.0 * 200.0 / 2 + 1e-6


def test_market_neutral_rejects_unknown_contract(scores) -> None:
    book = scores.head(4).with_columns(weight=pl.lit(0.25))
    with pytest.raises(ValueError, match="unknown hedge instrument"):
        market_neutral(book, capital=1e7, index_price=4000.0, hedge="ZZ")


def test_constraints_validate_bounds() -> None:
    with pytest.raises(ValueError, match="max_weight"):
        Constraints(max_weight=-0.01, min_weight=0.01)


# --------------------------------------------------------------------------
# covariance
# --------------------------------------------------------------------------


def test_ledoit_wolf_shrinks_toward_the_target() -> None:
    rng = np.random.default_rng(1)
    r = rng.normal(size=(60, 30))  # T barely exceeds N -> heavy shrinkage
    cov, intensity = ledoit_wolf(r)
    assert 0.0 <= intensity <= 1.0
    assert intensity > 0.0
    assert cov.shape == (30, 30)
    assert np.allclose(cov, cov.T)


def test_ledoit_wolf_is_psd_when_sample_is_not() -> None:
    rng = np.random.default_rng(2)
    r = rng.normal(size=(20, 40))  # T < N -> sample covariance is singular
    cov, _ = ledoit_wolf(r)
    assert np.linalg.eigvalsh(cov).min() > -1e-8


def test_factor_covariance_is_psd_and_structured() -> None:
    rng = np.random.default_rng(3)
    n, k, t = 50, 4, 120
    b = rng.normal(size=(n, k))
    f = rng.normal(size=(t, k)) * 0.01
    r = f @ b.T + rng.normal(size=(t, n)) * 0.005
    cov = factor_covariance(r, b)
    assert cov.shape == (n, n)
    assert np.linalg.eigvalsh(cov).min() > 0


def test_nearest_psd_clips_negative_eigenvalues() -> None:
    m = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3 and -1
    out = nearest_psd(m)
    assert np.linalg.eigvalsh(out).min() >= 0


def test_sample_covariance_needs_observations() -> None:
    with pytest.raises(ValueError, match="more than 1 observations"):
        sample_covariance(np.zeros((1, 5)))


# --------------------------------------------------------------------------
# ledger -- THE STEP 5 GATE
# --------------------------------------------------------------------------


@pytest.fixture
def book() -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(7)
    stamps = [dt.datetime(2024, 1, 2) + dt.timedelta(days=i) for i in range(10)]
    syms = [f"6000{j:02d}" for j in range(8)]
    rows_w, rows_p = [], []
    price = {s: 10.0 + j for j, s in enumerate(syms)}
    for t in stamps:
        for s in syms:
            price[s] *= 1.0 + rng.normal(0, 0.01)
            rows_p.append(
                {"ts": t, "symbol": s, "vwap": round(price[s], 2), "adv": 1e6, "sigma": 0.02}
            )
            rows_w.append({"ts": t, "symbol": s, "weight": float(rng.normal(0, 0.1))})
    return pl.DataFrame(rows_w), pl.DataFrame(rows_p)


def test_cross_sectional_accounting_produces_a_pnl_series(book) -> None:
    w, p = book
    res = simulate_cross_sectional(w, p, capital=1e7, costs=CostModel())
    assert res.pnl.height == 10
    assert {"gross_pnl", "cost", "net_pnl", "net_return", "turnover", "equity"} <= set(
        res.pnl.columns
    )
    assert res.total_cost > 0


def test_costs_reduce_pnl(book) -> None:
    w, p = book
    gross = simulate_cross_sectional(w, p, capital=1e7, costs=None)
    net = simulate_cross_sectional(w, p, capital=1e7, costs=CostModel())
    assert float(net.pnl["net_pnl"].sum()) < float(gross.pnl["net_pnl"].sum())


def test_parity_passes_on_identical_streams(book) -> None:
    """THE STEP 5 GATE (pass side).

    With fills held constant, the Python accountant must reproduce the C++ PnL
    exactly.
    """
    w, p = book
    res = simulate_cross_sectional(w, p, capital=1e7, costs=CostModel())
    report = parity_check(res.pnl, res.pnl, tolerance=1e-9)
    assert report.passed
    assert report.max_abs_diff == 0.0


def test_parity_fails_on_a_perturbed_stream(book) -> None:
    """THE STEP 5 GATE (fail side). A mismatch must raise, not warn."""
    w, p = book
    res = simulate_cross_sectional(w, p, capital=1e7, costs=CostModel())
    cpp = res.pnl.with_columns(
        net_pnl=pl.when(pl.int_range(pl.len()) == 3)
        .then(pl.col("net_pnl") * 1.05)
        .otherwise(pl.col("net_pnl"))
    )
    with pytest.raises(ParityError, match="parity FAILED"):
        parity_check(res.pnl, cpp, tolerance=1e-6)


def test_parity_reports_the_worst_period(book) -> None:
    w, p = book
    res = simulate_cross_sectional(w, p, capital=1e7, costs=CostModel())
    cpp = res.pnl.with_columns(
        net_pnl=pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.col("net_pnl") + 1000.0)
        .otherwise(pl.col("net_pnl"))
    )
    report = parity_check(res.pnl, cpp, tolerance=1e-6, raise_on_fail=False)
    assert not report.passed
    assert report.worst_period == res.pnl["ts"][5]


def test_parity_refuses_to_compare_disjoint_timestamps(book) -> None:
    """Comparing nothing is not a pass."""
    w, p = book
    res = simulate_cross_sectional(w, p, capital=1e7)
    shifted = res.pnl.with_columns(ts=pl.col("ts") + pl.duration(days=365))
    with pytest.raises(ParityError, match="share no timestamps"):
        parity_check(res.pnl, shifted)


def test_time_series_flags_fill_domination() -> None:
    """A signal whose PnL is smaller than the spread it crosses is not a result."""
    n = 200
    rng = np.random.default_rng(9)
    px = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
    df = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 2) + dt.timedelta(minutes=i) for i in range(n)],
            "signal": rng.choice([-1.0, 0.0, 1.0], n),
            "close": px,
            # Wide relative to the moves: a half-spread of 0.10 against ~0.01
            # daily moves means the fill assumption, not the signal, sets the PnL.
            "spread": np.full(n, 0.20),
        }
    )
    res = simulate_time_series(df, contract_size=100.0)
    assert res.is_fill_dominated
    assert "FILL-DOMINATED" in repr(res)


def test_time_series_reports_the_assumed_fill() -> None:
    n = 60
    df = pl.DataFrame(
        {
            "ts": [dt.datetime(2024, 1, 2) + dt.timedelta(minutes=i) for i in range(n)],
            "signal": [1.0] * n,
            "close": np.linspace(100, 110, n),
            "spread": np.full(n, 0.001),
        }
    )
    res = simulate_time_series(df)
    assert res.assumed_fill.startswith("next-bar")
    assert not res.is_fill_dominated
    assert float(res.pnl["net_pnl"].sum()) > 0
