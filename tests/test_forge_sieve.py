"""Step 3 gate: transforms are leak-free, and a duplicate factor is rejected."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from forge.ops import (
    REGISTRY,
    apply_pipeline,
    ema_ratio,
    get,
    neutralize,
    rank_to_normal,
    rolling_zscore,
    winsorize,
    zscore,
)
from quarry.loaders import load_daily, load_factor_frame
from sieve.screen import FactorPool, correlation_matrix, orthogonalize, screen


@pytest.fixture(scope="module")
def factors(conn) -> pl.DataFrame:
    daily = load_daily(conn)
    ff = load_factor_frame(conn)
    return ff.join(
        daily.select("ts", "symbol", "industry", "market_cap"), on=["ts", "symbol"], how="inner"
    )


# --------------------------------------------------------------------------
# forge
# --------------------------------------------------------------------------


def test_registry_is_populated() -> None:
    assert {"winsorize", "zscore", "neutralize", "rank_to_normal"} <= set(REGISTRY)


def test_get_unknown_transform_lists_options() -> None:
    with pytest.raises(KeyError, match="unknown transform"):
        get("does_not_exist")


def test_zscore_is_within_timestamp(factors) -> None:
    """Each cross-section standardises to mean 0, not the pooled sample."""
    out = zscore(factors, "alpha_strong", out="z")
    per_ts = out.group_by("ts").agg(pl.col("z").mean().alias("m"), pl.col("z").std().alias("s"))
    assert per_ts["m"].abs().max() < 1e-9
    assert (per_ts["s"] - 1.0).abs().max() < 1e-9


def test_zscore_does_not_use_future_cross_sections(factors) -> None:
    """Truncating the sample must not change earlier z-scores."""
    stamps = sorted(factors["ts"].unique().to_list())
    full = zscore(factors, "alpha_strong", out="z").filter(pl.col("ts") == stamps[0])
    prefix = zscore(factors.filter(pl.col("ts") <= stamps[2]), "alpha_strong", out="z").filter(
        pl.col("ts") == stamps[0]
    )
    assert full.sort("symbol")["z"].to_list() == pytest.approx(
        prefix.sort("symbol")["z"].to_list()
    )


def test_winsorize_clips_without_dropping(factors) -> None:
    spoiled = factors.with_columns(
        alpha_strong=pl.when(pl.int_range(pl.len()) == 0)
        .then(1e9)
        .otherwise(pl.col("alpha_strong"))
    )
    out = winsorize(spoiled, "alpha_strong", method="mad", k=3.0)
    assert out.height == spoiled.height  # clipped, not dropped
    assert out["alpha_strong"].max() < 1e9


def test_winsorize_quantile_variant(factors) -> None:
    out = winsorize(factors, "alpha_strong", method="quantile", lower=0.05, upper=0.95)
    assert out.height == factors.height


def test_winsorize_rejects_bad_method(factors) -> None:
    with pytest.raises(ValueError, match="mad.*quantile"):
        winsorize(factors, "alpha_strong", method="sigma")


def test_rank_to_normal_produces_gaussian_shape(factors) -> None:
    out = rank_to_normal(factors, "alpha_strong", out="g")
    vals = out["g"].drop_nulls().to_numpy()
    assert np.isfinite(vals).all()  # Blom offset keeps extremes finite
    assert abs(float(vals.mean())) < 0.1
    assert 0.7 < float(vals.std()) < 1.3


def test_neutralize_removes_industry_and_size(factors) -> None:
    """The residual must be uncorrelated with what it was neutralised against."""
    out = neutralize(factors, "alpha_strong", by=["industry"], size_col="market_cap", out="resid")
    valid = out.filter(pl.col("resid").is_not_null())
    corr = valid.select(
        pl.corr(pl.col("resid"), pl.col("market_cap").log()).alias("c")
    )["c"][0]
    assert abs(float(corr)) < 0.1


def test_neutralize_needs_enough_names(factors) -> None:
    """An under-identified regression must return null, not a zero residual."""
    out = neutralize(factors, "alpha_strong", by=["industry"], min_names=10_000)
    assert out["alpha_strong"].null_count() == out.height


def test_rolling_zscore_is_trailing(factors) -> None:
    out = rolling_zscore(factors, "alpha_strong", window=5, out="rz")
    first = out.filter(pl.col("symbol") == out["symbol"].min()).sort("ts")
    assert first["rz"][0] is None
    assert first["rz"].drop_nulls().len() > 0


def test_ema_ratio_requires_fast_lt_slow(factors) -> None:
    with pytest.raises(ValueError, match="fast span must be shorter"):
        ema_ratio(factors, "alpha_strong", fast=20, slow=5)


def test_ema_ratio_is_scale_free() -> None:
    """Doubling the price must not change the ratio."""
    base = pl.DataFrame(
        {
            "ts": list(range(30)) * 1,
            "symbol": ["A"] * 30,
            "px": np.linspace(10, 20, 30),
        }
    )
    doubled = base.with_columns(px=pl.col("px") * 2)
    a = ema_ratio(base, "px", fast=3, slow=10, out="r")["r"].to_numpy()
    b = ema_ratio(doubled, "px", fast=3, slow=10, out="r")["r"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_pipeline_runs_in_declared_order(factors) -> None:
    out = apply_pipeline(
        factors,
        [
            {"op": "winsorize", "column": "alpha_strong", "method": "mad", "k": 3.0},
            {"op": "neutralize", "column": "alpha_strong", "by": ["industry"], "size_col": "market_cap"},
            {"op": "zscore", "column": "alpha_strong"},
        ],
    )
    per_ts = out.group_by("ts").agg(pl.col("alpha_strong").mean().alias("m"))
    assert per_ts["m"].abs().max() < 1e-8


def test_pipeline_step_without_op_key(factors) -> None:
    with pytest.raises(KeyError, match="no 'op' key"):
        apply_pipeline(factors, [{"column": "alpha_strong"}])


# --------------------------------------------------------------------------
# sieve -- the step 3 gate
# --------------------------------------------------------------------------


def test_correlation_matrix_is_symmetric_with_unit_diagonal(factors) -> None:
    m = correlation_matrix(factors, ["alpha_strong", "alpha_mid", "alpha_none"])
    assert m.height == 3
    assert m["alpha_strong"][0] == pytest.approx(1.0, abs=1e-6)
    assert m["alpha_mid"][0] == pytest.approx(m["alpha_strong"][1], abs=1e-9)


def test_correlation_matrix_needs_two_factors(factors) -> None:
    with pytest.raises(ValueError, match="at least 2 factors"):
        correlation_matrix(factors, ["alpha_strong"])


def test_duplicate_factor_is_rejected(factors) -> None:
    """THE STEP 3 GATE.

    ``dupe_of_strong`` is ``alpha_strong`` plus a whisper of noise. It must not
    enter a pool that already holds the original.
    """
    res = screen(factors, ["alpha_strong"], "dupe_of_strong", threshold=0.7, reject_above=0.95)
    assert res.verdict == "reject"
    assert res.max_abs_corr > 0.95
    assert res.most_correlated == "alpha_strong"
    assert not res.admitted


def test_independent_factor_is_admitted(factors) -> None:
    res = screen(factors, ["alpha_strong"], "alpha_none", threshold=0.7)
    assert res.verdict == "admit"
    assert res.admitted


def test_empty_pool_always_admits(factors) -> None:
    res = screen(factors, [], "alpha_strong")
    assert res.verdict == "admit"


def test_candidate_already_in_pool_is_an_error(factors) -> None:
    with pytest.raises(ValueError, match="already in the pool"):
        screen(factors, ["alpha_strong"], "alpha_strong")


def test_screen_validates_thresholds(factors) -> None:
    with pytest.raises(ValueError, match="threshold <= reject_above"):
        screen(factors, ["alpha_strong"], "alpha_mid", threshold=0.9, reject_above=0.5)


def test_orthogonalize_removes_the_pool_component(factors) -> None:
    out = orthogonalize(factors, "dupe_of_strong", ["alpha_strong"], method="schmidt", out="resid")
    valid = out.filter(pl.col("resid").is_not_null())
    corr = valid.select(pl.corr("resid", "alpha_strong").alias("c"))["c"][0]
    assert abs(float(corr)) < 0.05


def test_symmetric_orthogonalization_touches_the_pool(factors) -> None:
    """Lowdin rewrites incumbents too -- that is the documented trade-off."""
    out = orthogonalize(
        factors, "alpha_mid", ["alpha_strong"], method="symmetric", out="resid"
    )
    changed = (
        out.filter(pl.col("resid").is_not_null())["alpha_strong"].to_numpy()
        != factors["alpha_strong"].to_numpy()[: out.filter(pl.col("resid").is_not_null()).height]
    )
    assert changed.any()


def test_orthogonalize_rejects_unknown_method(factors) -> None:
    with pytest.raises(ValueError, match="schmidt.*symmetric"):
        orthogonalize(factors, "alpha_mid", ["alpha_strong"], method="pca")


def test_factor_pool_admits_logs_and_rejects(factors) -> None:
    pool = FactorPool(threshold=0.7, reject_above=0.95)

    r1, _ = pool.propose(factors, "alpha_strong")
    assert r1.verdict == "admit"
    assert pool.factors == ["alpha_strong"]

    r2, _ = pool.propose(factors, "dupe_of_strong")
    assert r2.verdict == "reject"
    assert "dupe_of_strong" not in pool

    r3, _ = pool.propose(factors, "alpha_none")
    assert r3.verdict == "admit"

    log = pool.admission_log()
    assert log.height == 3  # rejections are logged too
    assert set(log["verdict"].to_list()) == {"admit", "reject"}


def test_admission_log_records_what_it_was_tested_against(factors) -> None:
    pool = FactorPool()
    pool.propose(factors, "alpha_strong")
    pool.propose(factors, "alpha_none")

    second = pool.log[1]
    assert second.tested_against == ("alpha_strong",)
    assert second.n_periods > 0
    assert second.n_names > 0
    assert second.sample_digest
    assert "alpha_none" in second.describe()


def test_pool_orthogonalizes_a_borderline_candidate(factors) -> None:
    """Between the thresholds the candidate is kept, but residualised first."""
    # alpha_strong and alpha_mid share only their planted signal, so their
    # correlation is about 0.40 * 0.15 = 0.06. The threshold has to straddle
    # that to exercise the middle branch.
    pool = FactorPool(threshold=0.03, reject_above=0.99)
    pool.propose(factors, "alpha_strong")
    res, out = pool.propose(factors, "alpha_mid")

    assert res.verdict == "orthogonalize"
    assert "alpha_mid" in pool
    assert pool.log[-1].orthogonalized
    valid = to_pl(out).filter(pl.col("alpha_mid").is_not_null())
    corr = valid.select(pl.corr("alpha_mid", "alpha_strong").alias("c"))["c"][0]
    assert abs(float(corr)) < 0.05


def to_pl(df) -> pl.DataFrame:
    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
