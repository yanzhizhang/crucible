"""Model fitting, walk-forward evaluation and the noise benchmark.

Three things live here and only the first is what people usually mean by
"modelling".

:func:`fit_lgbm` trains a gradient-boosted model with fixed seeds and
determinism forced on. :func:`walk_forward` runs it across purged folds and
collects out-of-sample predictions, asserting the absence of leakage in every
fold rather than trusting the splitter.

:func:`noise_benchmark` is the one that decides whether any of it mattered. It
re-runs the identical pipeline on phase-randomized series -- surrogates with the
*same* power spectrum, and therefore the same autocorrelation and trend
structure, but no genuine predictability -- and reports where the real Sharpe
falls in that distribution. A strategy that is not significant against this is
not a strategy; it is the backtest rediscovering the autocorrelation you fed it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from crucible.determinism import DEFAULT_SEED, rng
from crucible.errors import LeakageError
from crucible.frames import SYMBOL, TS, Frame, require_columns, to_polars
from kiln.splits import assert_no_leakage

__all__ = [
    "FitResult",
    "WalkForwardResult",
    "NoiseBenchmarkResult",
    "fit_lgbm",
    "walk_forward",
    "phase_randomize",
    "noise_benchmark",
    "assert_no_feature_leakage",
]

DEFAULT_LGBM_PARAMS: Mapping[str, Any] = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    # Determinism: LightGBM's default histogram construction is
    # thread-order-dependent and produces slightly different trees run to run.
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}
"""Conservative defaults. ``min_data_in_leaf`` is deliberately high: with 5000
names per cross-section, a small leaf fits a handful of stocks on one day."""


@dataclass(frozen=True)
class FitResult:
    """A fitted model plus what it was fitted on."""

    model: Any
    features: tuple[str, ...]
    importance: pl.DataFrame
    n_train: int
    n_groups: int
    params: Mapping[str, Any]
    seed: int
    best_iteration: int | None = None

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Score a design matrix."""
        return np.asarray(self.model.predict(x), dtype=float)

    def __repr__(self) -> str:
        return (
            f"FitResult({len(self.features)} features, {self.n_train} rows, "
            f"{self.n_groups} cross-sections, seed={self.seed})"
        )


@dataclass(frozen=True)
class WalkForwardResult:
    """Out-of-sample predictions from a purged walk-forward run."""

    predictions: pl.DataFrame
    fold_scores: pl.DataFrame
    models: tuple[FitResult, ...] = field(repr=False, default=())

    @property
    def n_folds(self) -> int:
        """Number of folds actually evaluated."""
        return self.fold_scores.height

    @property
    def mean_ic(self) -> float:
        """Average out-of-sample cross-sectional IC across folds."""
        if not self.fold_scores.height:
            return 0.0
        return float(self.fold_scores["ic"].mean())

    def __repr__(self) -> str:
        return f"WalkForwardResult({self.n_folds} folds, mean OOS IC={self.mean_ic:+.4f})"


@dataclass(frozen=True)
class NoiseBenchmarkResult:
    """Where a real score sits in a phase-randomized null distribution."""

    real_score: float
    null_scores: np.ndarray = field(repr=False)
    percentile: float
    p_value: float
    n_trials: int

    @property
    def is_significant(self) -> bool:
        """Whether the real score beats the null at the 5% level."""
        return self.p_value < 0.05

    def __repr__(self) -> str:
        verdict = "significant" if self.is_significant else "NOT significant vs noise"
        return (
            f"NoiseBenchmarkResult(real={self.real_score:+.3f} "
            f"null_mean={float(np.mean(self.null_scores)):+.3f} "
            f"pctile={self.percentile:.1f} p={self.p_value:.4f} -> {verdict})"
        )


def assert_no_feature_leakage(
    df: Frame,
    features: Sequence[str],
    label: str,
    *,
    max_abs_ic: float = 0.95,
    ts: str = TS,
    min_names: int = 20,
) -> None:
    """Refuse features that are implausibly correlated with the label.

    A cross-sectional IC above ``max_abs_ic`` does not happen in equity
    research. It means the feature *is* the label, or a transform of it -- a
    column accidentally carrying the forward return, a "feature" built from a
    future price, or a join that pulled the target in twice.

    This is a canary, not a proof. It catches the loud failure, which is the
    one that otherwise sails through validation looking like a triumph. Subtle
    leakage still needs :func:`kiln.assert_no_leakage` on the splits.

    Raises
    ------
    LeakageError
        Naming the offending feature and its IC.
    """
    if not 0.0 < max_abs_ic <= 1.0:
        raise ValueError(f"max_abs_ic must be in (0, 1], got {max_abs_ic}")

    from assay.cross_sectional import ic as _ic

    lf = to_polars(df)
    require_columns(lf, (ts, label, *features), where="assert_no_feature_leakage")

    for f in features:
        r = _ic(lf, f, label, method="spearman", min_names=min_names)
        if r.n_periods == 0:
            continue
        if abs(r.mean) >= max_abs_ic:
            raise LeakageError(
                f"feature {f!r} has cross-sectional IC {r.mean:+.4f} against label "
                f"{label!r} over {r.n_periods} periods. An |IC| >= {max_abs_ic} is not "
                f"a discovery -- the feature is the label, or derived from it. Check "
                f"for a column built from a future price or a duplicated join key."
            )


def fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[object] | np.ndarray | None = None,
    params: Mapping[str, Any] | None = None,
    *,
    features: Sequence[str] | None = None,
    num_boost_round: int = 300,
    seed: int = DEFAULT_SEED,
) -> FitResult:
    """Fit a LightGBM model with determinism forced on.

    Parameters
    ----------
    groups:
        Timestamps, one per row. Not passed to LightGBM for a regression
        objective, but recorded so the number of genuine cross-sections is
        visible -- 400k rows across 80 timestamps is 80 samples, not 400k, and
        the fit result says so.
    seed:
        Fixed and threaded into every LightGBM randomness knob.

    Returns
    -------
    FitResult
        With gain-based feature importance.

    Notes
    -----
    ``num_threads=1`` and ``deterministic=True`` are set by default. Multi-thread
    histogram construction in LightGBM is order-dependent, so relaxing either
    trades byte-identical reproducibility for speed. That is a legitimate
    trade, but it must be a deliberate one.
    """
    import lightgbm as lgb

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D, got shape {x.shape}")
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} rows but y has {len(y)}")

    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if not finite.any():
        raise ValueError("no rows with finite features and label")
    x, y = x[finite], y[finite]

    names = list(features) if features is not None else [f"f{i}" for i in range(x.shape[1])]
    merged = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    merged.update({"seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed})

    dataset = lgb.Dataset(x, label=y, feature_name=names, free_raw_data=False)
    booster = lgb.train(merged, dataset, num_boost_round=num_boost_round)

    imp = pl.DataFrame(
        {
            "feature": names,
            "gain": booster.feature_importance("gain").astype(float),
            "split": booster.feature_importance("split").astype(float),
        }
    ).sort("gain", descending=True)

    n_groups = int(len(np.unique(np.asarray(groups)[finite]))) if groups is not None else 0

    return FitResult(
        model=booster,
        features=tuple(names),
        importance=imp,
        n_train=int(len(y)),
        n_groups=n_groups,
        params=merged,
        seed=seed,
    )


def walk_forward(
    df: Frame,
    features: Sequence[str],
    label: str,
    splitter: Any,
    *,
    params: Mapping[str, Any] | None = None,
    num_boost_round: int = 300,
    seed: int = DEFAULT_SEED,
    ts: str = TS,
    check_leakage: bool = True,
    keep_models: bool = False,
) -> WalkForwardResult:
    """Fit and predict across purged walk-forward folds.

    Parameters
    ----------
    splitter:
        Anything with ``.split(ts) -> (train_idx, test_idx)``, i.e.
        :class:`kiln.PurgedWalkForward` or :class:`kiln.CombinatorialPurgedCV`.
    check_leakage:
        Assert every fold is clean before fitting it. On by default and there
        is no good reason to turn it off -- the check costs microseconds and the
        failure it catches costs weeks.

    Returns
    -------
    WalkForwardResult
        Out-of-sample predictions keyed by ``(ts, symbol)``, plus a per-fold IC.

    Raises
    ------
    LeakageError
        From :func:`kiln.assert_no_leakage` when a fold is contaminated.
    """
    lf = to_polars(df).sort([ts, SYMBOL], maintain_order=True)
    require_columns(lf, (ts, SYMBOL, label, *features), where="walk_forward")

    stamps = lf[ts].to_numpy()
    x_all = lf.select(features).to_numpy().astype(float)
    y_all = lf[label].to_numpy().astype(float)

    preds: list[pl.DataFrame] = []
    scores: list[dict[str, object]] = []
    models: list[FitResult] = []

    for fold, (tr, te) in enumerate(splitter.split(stamps)):
        if check_leakage:
            assert_no_leakage(
                stamps,
                tr,
                te,
                label_horizon=getattr(splitter, "label_horizon", 1),
                embargo=getattr(splitter, "embargo", 0),
            )

        ok = np.isfinite(y_all[tr]) & np.isfinite(x_all[tr]).all(axis=1)
        if ok.sum() < 50:
            continue

        fit = fit_lgbm(
            x_all[tr][ok],
            y_all[tr][ok],
            groups=stamps[tr][ok],
            params=params,
            features=features,
            num_boost_round=num_boost_round,
            seed=seed,
        )
        if keep_models:
            models.append(fit)

        yhat = fit.predict(x_all[te])
        block = lf[te].select(ts, SYMBOL, label).with_columns(
            pred=pl.Series("pred", yhat), fold=pl.lit(fold, dtype=pl.Int32)
        )
        preds.append(block)

        from assay.cross_sectional import ic as _ic

        r = _ic(block, "pred", label, method="spearman", min_names=10)
        scores.append(
            {
                "fold": fold,
                "n_train": int(ok.sum()),
                "n_test": int(len(te)),
                "ic": r.mean,
                "icir": r.icir,
                "n_periods": r.n_periods,
            }
        )

    predictions = pl.concat(preds) if preds else pl.DataFrame()
    fold_scores = pl.DataFrame(scores) if scores else pl.DataFrame()
    return WalkForwardResult(predictions, fold_scores, tuple(models))


def phase_randomize(x: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Surrogate series with the same power spectrum but randomized phases.

    The resulting series preserves the autocorrelation structure -- and hence
    trend, momentum and volatility clustering as measured by second-order
    statistics -- while destroying any genuine predictive relationship. That
    makes it the right null: a strategy that beats it is exploiting something
    beyond the linear autocorrelation it was handed.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 4:
        return x.copy()

    spectrum = np.fft.rfft(x - x.mean())
    magnitude = np.abs(spectrum)
    phases = generator.uniform(0, 2 * np.pi, magnitude.size)
    phases[0] = 0.0  # keep the DC term real
    if n % 2 == 0:
        phases[-1] = 0.0  # Nyquist term must stay real too
    surrogate = np.fft.irfft(magnitude * np.exp(1j * phases), n=n)
    return surrogate + x.mean()


def noise_benchmark(
    series: np.ndarray | pl.Series,
    score_fn: Callable[[np.ndarray], float],
    *,
    n_trials: int = 200,
    seed: int = DEFAULT_SEED,
) -> NoiseBenchmarkResult:
    """Score a real series against phase-randomized surrogates.

    Parameters
    ----------
    series:
        The underlying series the pipeline consumes -- typically returns.
    score_fn:
        Runs the **identical** pipeline on a series and returns one score
        (Sharpe, IC, whatever). It must be deterministic and must not peek at
        anything outside its argument, or the null is not comparable.
    n_trials:
        Surrogates to generate. 200 gives a p-value resolution of 0.005.

    Returns
    -------
    NoiseBenchmarkResult
        With a one-sided p-value: the fraction of surrogates scoring at least
        as well as the real series.

    Notes
    -----
    The p-value uses the ``(k + 1) / (n + 1)`` correction, so a real score that
    beats every surrogate reports ``1/(n+1)`` rather than an impossible zero.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    arr = series.to_numpy() if isinstance(series, pl.Series) else np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 16:
        raise ValueError(f"need at least 16 observations for a surrogate test, got {arr.size}")

    real = float(score_fn(arr))
    generator = rng(seed)
    null = np.array([float(score_fn(phase_randomize(arr, generator))) for _ in range(n_trials)])

    k = int((null >= real).sum())
    return NoiseBenchmarkResult(
        real_score=real,
        null_scores=null,
        percentile=float((null < real).mean() * 100.0),
        p_value=(k + 1) / (n_trials + 1),
        n_trials=n_trials,
    )
