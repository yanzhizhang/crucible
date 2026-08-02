"""Score-to-weight conversion, constraints and hedging.

The step from "I have a signal" to "I hold this portfolio" is where most of a
paper factor's edge is lost, and almost none of that loss is visible in the IC.

Three things happen here, in order:

1. **Sizing.** Turn cross-sectional scores into weights.
2. **Constraints.** Cap single names, cap industry tilts, cap turnover. Each
   one costs expected return and each one is worth it, because an unconstrained
   ranking portfolio concentrates in exactly the illiquid, high-idiosyncratic
   names where the score is least reliable.
3. **Hedging.** Neutralise residual market exposure with index futures.

Constraints are applied by projection, not by rejection. A portfolio that
violates a cap is repaired and renormalised rather than discarded, because the
alternative -- skipping the rebalance -- silently changes the strategy.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = [
    "Constraints",
    "score_to_weight",
    "combine_scores",
    "apply_constraints",
    "market_neutral",
    "HedgeResult",
    "INDEX_MULTIPLIERS",
]

INDEX_MULTIPLIERS: Mapping[str, float] = {
    "IF": 300.0,  # CSI 300, 300 CNY per index point
    "IH": 300.0,  # SSE 50
    "IC": 200.0,  # CSI 500
    "IM": 200.0,  # CSI 1000
}
"""CFFEX index-future contract multipliers, CNY per index point."""


@dataclass(frozen=True)
class Constraints:
    """Portfolio constraints.

    Parameters
    ----------
    max_weight, min_weight:
        Per-name bounds. ``min_weight`` should be negative for a long-short
        book and zero for long-only.
    max_industry:
        Cap on the absolute net weight in any one industry.
    max_turnover:
        Cap on one-way turnover per rebalance, as a fraction of gross. Binding
        this is usually more valuable than any refinement to the signal --
        turnover is the one cost lever entirely under your control.
    gross_exposure, net_exposure:
        Target gross (sum of absolute weights) and net (sum of weights).
        ``net_exposure=0`` gives a dollar-neutral book.
    """

    max_weight: float = 0.02
    min_weight: float = -0.02
    max_industry: float | None = 0.10
    max_turnover: float | None = None
    gross_exposure: float = 1.0
    net_exposure: float | None = 0.0

    def __post_init__(self) -> None:
        if self.max_weight <= self.min_weight:
            raise ValueError(
                f"max_weight ({self.max_weight}) must exceed min_weight ({self.min_weight})"
            )
        if self.gross_exposure <= 0:
            raise ValueError(f"gross_exposure must be > 0, got {self.gross_exposure}")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise ValueError(f"max_turnover must be >= 0, got {self.max_turnover}")


def combine_scores(
    df: Frame,
    factors: Sequence[str],
    *,
    weights: Mapping[str, float] | None = None,
    method: str = "equal",
    out: str = "score",
    ts: str = TS,
) -> Frame:
    """Blend several factor columns into one score.

    Parameters
    ----------
    method:
        ``"equal"`` averages the standardised factors. ``"ic_weighted"`` and
        ``"ir_weighted"`` weight each factor by the value supplied in
        ``weights`` (its historical IC or ICIR respectively).
    weights:
        Required for the IC/IR methods. **Must be estimated on data strictly
        before the period being weighted** -- computing IC over the full sample
        and then weighting by it is a look-ahead that reliably manufactures
        alpha.

    Notes
    -----
    Factors are z-scored within each timestamp before blending, so a factor
    with a naturally larger scale does not dominate the average.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (ts, *factors), where="combine_scores")

    if method == "equal":
        w = {f: 1.0 for f in factors}
    elif method in ("ic_weighted", "ir_weighted"):
        if not weights:
            raise ValueError(f"method={method!r} requires `weights` (factor -> IC or ICIR)")
        missing = [f for f in factors if f not in weights]
        if missing:
            raise ValueError(f"no weight supplied for factor(s) {missing}")
        w = {f: float(weights[f]) for f in factors}
    else:
        raise ValueError(f"unknown method {method!r}")

    total = sum(abs(v) for v in w.values())
    if total <= 0:
        raise ValueError("factor weights sum to zero; cannot combine")

    standardized = []
    for f in factors:
        mu = pl.col(f).mean().over(ts)
        sd = pl.col(f).std().over(ts)
        standardized.append(
            pl.when(sd > 0).then((pl.col(f) - mu) / sd * (w[f] / total)).otherwise(0.0)
        )

    res = lf.with_columns(pl.sum_horizontal(standardized).alias(out))
    return restore(res, want)


def score_to_weight(
    df: Frame,
    score: str = "score",
    *,
    method: str = "rank",
    top_quantile: float = 0.2,
    constraints: Constraints | None = None,
    ts: str = TS,
    out: str = "weight",
) -> Frame:
    """Convert cross-sectional scores into portfolio weights.

    Parameters
    ----------
    method:
        ``"equal"``
            Long the top ``top_quantile``, short the bottom, equally weighted.
            Robust and nearly free of estimation error, which is why it is
            still the default in most published factor work.
        ``"rank"``
            Weight proportional to demeaned cross-sectional rank. Uses the
            whole distribution instead of discarding the middle, and is far
            less sensitive to outliers than raw scores.
        ``"score"``
            Weight proportional to the demeaned score itself. Only sensible
            when the score is genuinely cardinal -- an expected return, not a
            ranking.

    constraints:
        Applied after sizing via :func:`apply_constraints`. When omitted the
        weights are only normalised to unit gross.

    Returns
    -------
    Frame with a ``weight`` column, in the caller's flavor.

    Point-in-time contract
    ----------------------
    Purely cross-sectional: weights at ``t`` depend only on scores at ``t``.
    Any historical input (IC estimates for blending, prior weights for turnover
    control) must be supplied by the caller and must be lagged.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (ts, SYMBOL, score), where="score_to_weight")

    valid = pl.col(score).is_not_null() & pl.col(score).is_finite()

    if method == "equal":
        if not 0.0 < top_quantile <= 0.5:
            raise ValueError(f"top_quantile must be in (0, 0.5], got {top_quantile}")
        pct = pl.col(score).rank("average").over(ts) / pl.col(score).count().over(ts)
        raw = (
            pl.when(~valid)
            .then(0.0)
            .when(pct > 1.0 - top_quantile)
            .then(1.0)
            .when(pct <= top_quantile)
            .then(-1.0)
            .otherwise(0.0)
        )
    elif method == "rank":
        r = pl.col(score).rank("average").over(ts)
        raw = pl.when(valid).then(r - r.mean().over(ts)).otherwise(0.0)
    elif method == "score":
        raw = pl.when(valid).then(pl.col(score) - pl.col(score).mean().over(ts)).otherwise(0.0)
    else:
        raise ValueError(f"unknown method {method!r}; use equal, rank or score")

    sized = lf.with_columns(_raw=raw)
    gross = pl.col("_raw").abs().sum().over(ts)
    sized = sized.with_columns(
        pl.when(gross > 0).then(pl.col("_raw") / gross).otherwise(0.0).alias(out)
    ).drop("_raw")

    if constraints is not None:
        sized = to_polars(apply_constraints(sized, constraints, weight=out, ts=ts))

    return restore(sized.sort([ts, SYMBOL], maintain_order=True), want)


def apply_constraints(
    df: Frame,
    constraints: Constraints,
    *,
    weight: str = "weight",
    industry: str = "industry",
    prev_weight: str | None = None,
    ts: str = TS,
    max_iter: int = 50,
) -> Frame:
    """Project weights onto the constraint set, per timestamp.

    Caps, industry limits and net exposure interact -- clipping to a name cap
    changes the net, renormalising the net can re-breach the cap -- so this
    iterates to a fixed point rather than applying each rule once.

    Parameters
    ----------
    prev_weight:
        Previous weights, needed only when ``max_turnover`` is set. Turnover is
        limited by blending toward the prior book, which reduces trade size
        proportionally rather than truncating arbitrary names.

    Returns
    -------
    Frame with ``weight`` replaced by the constrained weights.

    Notes
    -----
    Convergence is not guaranteed for pathological constraint sets (a name cap
    below ``1/n``, say). After ``max_iter`` the last iterate is returned, still
    satisfying the box constraint but possibly not the exposure targets exactly.
    """
    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (ts, SYMBOL, weight), where="apply_constraints")
    has_industry = industry in lf.columns and constraints.max_industry is not None

    # A name cap can make the gross target unreachable: n names capped at
    # max_weight admit at most n * max_weight of gross. Silently delivering 0.8
    # when 1.0 was requested shows up later as unexplained underperformance, so
    # say so once rather than never.
    widest = int(lf.group_by(ts).len()["len"].max() or 0)
    reachable = widest * max(abs(constraints.max_weight), abs(constraints.min_weight))
    if reachable < constraints.gross_exposure - 1e-12:
        warnings.warn(
            f"gross_exposure={constraints.gross_exposure} is unreachable: the widest "
            f"cross-section has {widest} names capped at {constraints.max_weight}, "
            f"allowing at most {reachable:.3f} gross. Weights will respect the cap and "
            f"fall short of the target -- raise the cap or widen the universe.",
            UserWarning,
            stacklevel=2,
        )

    out_parts: list[pl.DataFrame] = []
    for part in lf.partition_by(ts, maintain_order=True):
        w = np.nan_to_num(part[weight].to_numpy().astype(float))

        codes = (
            part[industry].cast(pl.Categorical).to_physical().to_numpy()
            if has_industry
            else None
        )
        # Order matters and is by hardness, softest first: the *last* operation
        # is the one guaranteed to hold on exit. Name and industry caps are risk
        # limits and must never be breached; gross and net are targets. Doing
        # the gross rescale last would push weights back over the cap -- which
        # is exactly the bug this ordering fixes. When a cap binds, gross
        # therefore lands slightly under its target rather than the cap being
        # violated.
        for _ in range(max_iter):
            before = w.copy()

            gross = np.abs(w).sum()
            if gross > 0:
                w *= constraints.gross_exposure / gross

            if constraints.net_exposure is not None:
                drift = w.sum() - constraints.net_exposure
                if abs(drift) > 1e-12 and w.size:
                    w -= drift / w.size

            if codes is not None:
                cap = float(constraints.max_industry)  # type: ignore[arg-type]
                for code in np.unique(codes):
                    m = codes == code
                    net = w[m].sum()
                    if abs(net) > cap > 0:
                        w[m] *= cap / abs(net)

            w = np.clip(w, constraints.min_weight, constraints.max_weight)

            if np.max(np.abs(w - before)) < 1e-12:
                break

        if constraints.max_turnover is not None and prev_weight and prev_weight in part.columns:
            prev = np.nan_to_num(part[prev_weight].to_numpy().astype(float))
            trade = np.abs(w - prev).sum()
            if trade > constraints.max_turnover > 0:
                lam = constraints.max_turnover / trade
                w = prev + lam * (w - prev)

        out_parts.append(part.with_columns(pl.Series(weight, w)))

    res = pl.concat(out_parts) if out_parts else lf
    return restore(res.sort([ts, SYMBOL], maintain_order=True), want)


@dataclass(frozen=True)
class HedgeResult:
    """Index-future hedge sizing for a long-short book."""

    hedge: str
    beta: float
    notional: float
    contracts: int
    residual_notional: float
    multiplier: float

    def __repr__(self) -> str:
        return (
            f"HedgeResult({self.hedge}: beta={self.beta:+.3f} "
            f"{self.contracts:+d} contracts, residual={self.residual_notional:,.0f} CNY)"
        )


def market_neutral(
    weights: Frame,
    *,
    capital: float,
    index_price: float,
    hedge: str = "IM",
    beta_col: str | None = "beta",
    weight: str = "weight",
    multiplier: float | None = None,
) -> HedgeResult:
    """Size an index-futures hedge against a book's net beta exposure.

    Parameters
    ----------
    hedge:
        Contract root. ``IM`` (CSI 1000) suits a small-cap tilted A-share book;
        ``IF`` (CSI 300) a large-cap one. Hedging a small-cap book with IF
        leaves a large-minus-small residual that frequently exceeds the market
        exposure being removed.
    beta_col:
        Per-name beta. When absent every name is assumed beta 1.0, which
        overstates the hedge for a low-beta book.

    Returns
    -------
    HedgeResult
        Contracts are **integral** -- futures cannot be traded fractionally --
        and ``residual_notional`` reports the exposure that rounding leaves
        behind. For small books that residual can be a large share of the
        intended hedge, and it is reported rather than hidden.
    """
    lf = to_polars(weights)
    require_columns(lf, (weight,), where="market_neutral")

    mult = multiplier if multiplier is not None else INDEX_MULTIPLIERS.get(hedge)
    if mult is None:
        raise ValueError(
            f"unknown hedge instrument {hedge!r}; pass `multiplier` explicitly. "
            f"Known: {sorted(INDEX_MULTIPLIERS)}"
        )
    if index_price <= 0:
        raise ValueError(f"index_price must be > 0, got {index_price}")

    w = np.nan_to_num(lf[weight].to_numpy().astype(float))
    if beta_col and beta_col in lf.columns:
        beta_vec = np.nan_to_num(lf[beta_col].to_numpy().astype(float), nan=1.0)
    else:
        beta_vec = np.ones_like(w)

    net_beta = float((w * beta_vec).sum())
    exposure = net_beta * capital
    contract_value = index_price * mult
    contracts = int(np.round(-exposure / contract_value))
    residual = exposure + contracts * contract_value

    return HedgeResult(
        hedge=hedge,
        beta=net_beta,
        notional=exposure,
        contracts=contracts,
        residual_notional=residual,
        multiplier=mult,
    )
