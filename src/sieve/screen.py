"""Factor screening, orthogonalization and pool admission.

The problem this solves: the twentieth momentum variant added to a pool is
almost never a new bet. It correlates 0.9 with the existing ones, adds no
independent information, and inflates every risk estimate that assumes the
factors are distinct. Worse, its apparent standalone IC is real -- it looks
like a discovery right up until it does nothing in the portfolio.

Screening is therefore a **three-way** decision, not a pass/fail. A candidate
is admitted, rejected as redundant, or admitted *after* being orthogonalized
against the incumbents. The third branch is what keeps a genuinely new
component of a correlated factor rather than discarding it.

Every admission is logged with what it was tested against and on which sample,
because "why is this factor in the pool" is a question that gets asked six
months later when nobody remembers.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl

from crucible.determinism import stable_hash
from crucible.frames import SYMBOL, TS, Frame, flavor, require_columns, restore, to_polars

__all__ = [
    "correlation_matrix",
    "ScreenResult",
    "screen",
    "orthogonalize",
    "AdmissionRecord",
    "FactorPool",
]

Verdict = Literal["admit", "reject", "orthogonalize"]


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties shared -- the Spearman transform.

    Local rather than ``scipy.stats.rankdata`` so :mod:`sieve` keeps its only
    hard dependencies as numpy and polars.
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # Share ranks across tied runs.
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def correlation_matrix(
    df: Frame,
    factors: Sequence[str],
    *,
    method: str = "spearman",
    ts: str = TS,
    min_names: int = 20,
) -> pl.DataFrame:
    """Cross-sectional correlation between factors, averaged over time.

    Correlation is computed **within each timestamp** and then averaged across
    timestamps. Pooling all rows and correlating once would blend
    cross-sectional co-movement with shared time-series drift, and two factors
    that are unrelated in the cross-section can look highly correlated that way
    purely because both trend.

    Returns
    -------
    A square frame with a ``factor`` label column and one column per factor.
    Pairs never observed together carry null rather than zero -- zero would
    read as "independent", which is a much stronger claim than "unknown".
    """
    lf = to_polars(df)
    require_columns(lf, (ts, *factors), where="correlation_matrix")
    k = len(factors)
    if k < 2:
        raise ValueError(f"need at least 2 factors, got {k}")

    sums = np.zeros((k, k))
    counts = np.zeros((k, k))

    # One numpy correlation per timestamp rather than k*(k+1)/2 polars calls.
    # At 36 factors and 600 sessions the pairwise form is ~400k round trips
    # through the expression engine and takes minutes; this takes seconds.
    #
    # Rows with any null are dropped per timestamp (listwise deletion) so every
    # pair is measured on the same sample and the resulting matrix is
    # internally consistent. Pairwise deletion would give a matrix whose
    # entries came from different subsets, which can fail to be positive
    # semi-definite and quietly breaks anything that decomposes it.
    for part in lf.partition_by(ts, maintain_order=True):
        block = part.select(factors).drop_nulls()
        if block.height < min_names:
            continue
        arr = block.to_numpy().astype(float)
        if not np.isfinite(arr).all():
            keep = np.isfinite(arr).all(axis=1)
            arr = arr[keep]
            if arr.shape[0] < min_names:
                continue
        if method == "spearman":
            arr = np.apply_along_axis(_rankdata, 0, arr)
        # Constant columns have zero variance and no defined correlation.
        sd = arr.std(axis=0)
        good = sd > 0
        if good.sum() < 2:
            continue
        c = np.corrcoef(arr[:, good], rowvar=False)
        idx = np.flatnonzero(good)
        sums[np.ix_(idx, idx)] += np.nan_to_num(c)
        counts[np.ix_(idx, idx)] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mat = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    return pl.DataFrame(
        {"factor": list(factors), **{f: mat[:, j] for j, f in enumerate(factors)}}
    )


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of testing one candidate against a pool."""

    candidate: str
    verdict: Verdict
    max_abs_corr: float
    most_correlated: str | None
    correlations: dict[str, float]
    threshold: float
    n_periods: int
    reason: str

    @property
    def admitted(self) -> bool:
        """Whether the candidate enters the pool, possibly after residualising."""
        return self.verdict in ("admit", "orthogonalize")

    def __repr__(self) -> str:
        return (
            f"ScreenResult({self.candidate}: {self.verdict.upper()} "
            f"max|rho|={self.max_abs_corr:.3f} vs {self.most_correlated} -- {self.reason})"
        )


def screen(
    df: Frame,
    pool: Sequence[str],
    candidate: str,
    *,
    threshold: float = 0.7,
    reject_above: float = 0.95,
    method: str = "spearman",
    ts: str = TS,
    min_names: int = 20,
    corr: pl.DataFrame | None = None,
) -> ScreenResult:
    """Decide whether ``candidate`` adds anything to ``pool``.

    Parameters
    ----------
    threshold:
        Above this absolute correlation the candidate is not independent and
        must be orthogonalized before admission.
    reject_above:
        Above this it is a duplicate. Residualising would leave essentially
        pure noise, and admitting that noise as a "new factor" is worse than
        rejecting outright.

    Returns
    -------
    ScreenResult
        ``admit`` below ``threshold``, ``orthogonalize`` between the two, and
        ``reject`` above ``reject_above``.

    Notes
    -----
    An empty pool always admits -- the first factor has nothing to be redundant
    with.
    """
    if not 0.0 < threshold <= reject_above <= 1.0:
        raise ValueError(
            f"need 0 < threshold <= reject_above <= 1, got {threshold}, {reject_above}"
        )
    if not pool:
        return ScreenResult(
            candidate, "admit", 0.0, None, {}, threshold, 0, "pool is empty"
        )
    if candidate in pool:
        raise ValueError(f"{candidate!r} is already in the pool")

    lf = to_polars(df)
    # A caller screening many candidates against the same panel should compute
    # the full matrix once and pass it here. Recomputing per candidate is
    # O(k) redundant passes over the whole history for no new information.
    mat = (
        corr
        if corr is not None
        else correlation_matrix(
            lf, [*pool, candidate], method=method, ts=ts, min_names=min_names
        )
    )
    row = mat.filter(pl.col("factor") == candidate)
    if row.height == 0:
        raise KeyError(f"{candidate!r} is absent from the supplied correlation matrix")
    corrs = {p: float(row[p][0]) for p in pool if row[p][0] is not None}

    if not corrs:
        return ScreenResult(
            candidate,
            "reject",
            0.0,
            None,
            {},
            threshold,
            0,
            "no timestamp had enough overlapping names to measure correlation",
        )

    worst = max(corrs, key=lambda p: abs(corrs[p]))
    m = abs(corrs[worst])
    n_periods = int(lf[ts].n_unique())

    if m >= reject_above:
        verdict: Verdict = "reject"
        reason = f"duplicate of {worst} (|rho|={m:.3f} >= {reject_above})"
    elif m >= threshold:
        verdict = "orthogonalize"
        reason = f"overlaps {worst} (|rho|={m:.3f} >= {threshold}); residualise first"
    else:
        verdict = "admit"
        reason = f"independent enough (max |rho|={m:.3f} < {threshold})"

    return ScreenResult(candidate, verdict, m, worst, corrs, threshold, n_periods, reason)


def orthogonalize(
    df: Frame,
    candidate: str,
    pool: Sequence[str],
    *,
    method: str = "schmidt",
    out: str | None = None,
    ts: str = TS,
    min_names: int = 10,
) -> Frame:
    """Remove the pool's explanatory power from a candidate factor.

    Parameters
    ----------
    method:
        ``"schmidt"`` regresses the candidate on the pool within each timestamp
        and keeps the residual. Order-dependent when applied repeatedly, but
        the incumbents are left untouched -- which is what you want when the
        pool is already in production.

        ``"symmetric"`` applies a Lowdin transform to the candidate *and* the
        pool together, producing an orthogonal set that is collectively closest
        to the originals. Order-independent and fairer, but it **modifies the
        incumbents**, so every previously computed result on them is invalidated.

    Returns
    -------
    Frame with the residualised candidate in ``out`` (defaults to overwriting
    ``candidate``). Under ``"symmetric"`` the pool columns are rewritten too.

    Notes
    -----
    Fitted within each timestamp, so no cross-time information leaks into the
    residual.
    """
    if method not in ("schmidt", "symmetric"):
        raise ValueError(f"method must be 'schmidt' or 'symmetric', got {method!r}")

    want = flavor(df)
    lf = to_polars(df)
    require_columns(lf, (ts, candidate, *pool), where="orthogonalize")
    tgt = candidate if out is None else out

    cols = [*pool, candidate]
    result = {c: np.full(lf.height, np.nan) for c in (cols if method == "symmetric" else [candidate])}
    lf = lf.with_row_index("_row")

    for part in lf.partition_by(ts, maintain_order=True):
        valid = part.drop_nulls(cols)
        if valid.height < min_names:
            continue
        rows = valid["_row"].to_numpy()
        x = np.column_stack([valid[c].to_numpy().astype(float) for c in cols])
        x = x - x.mean(axis=0)

        if method == "schmidt":
            a, y = x[:, :-1], x[:, -1]
            design = np.column_stack([np.ones(len(y)), a])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            result[candidate][rows] = y - design @ beta
        else:
            # Lowdin: S^{-1/2} X, where S is the correlation matrix.
            sd = x.std(axis=0, ddof=1)
            if np.any(sd <= 0):
                continue
            z = x / sd
            s = np.corrcoef(z, rowvar=False)
            vals, vecs = np.linalg.eigh(s)
            vals = np.maximum(vals, 1e-12)
            s_inv_sqrt = vecs @ np.diag(vals**-0.5) @ vecs.T
            ortho = z @ s_inv_sqrt
            for j, c in enumerate(cols):
                result[c][rows] = ortho[:, j]

    res = lf.drop("_row")
    if method == "schmidt":
        written = [tgt]
        res = res.with_columns(pl.Series(tgt, result[candidate]))
    else:
        written = [c if c != candidate else tgt for c in cols]
        res = res.with_columns(
            [pl.Series(c if c != candidate else tgt, result[c]) for c in cols]
        )
    # Skipped cross-sections must be null, not NaN -- see forge.neutralize.
    res = res.with_columns([pl.col(c).fill_nan(None) for c in written])
    return restore(res, want)


@dataclass(frozen=True)
class AdmissionRecord:
    """Why one factor is in the pool.

    Deliberately verbose. Six months on, the only defence against a pool nobody
    trusts is a record of what each factor was tested against, on what sample,
    and with what result.
    """

    factor: str
    verdict: Verdict
    tested_against: tuple[str, ...]
    max_abs_corr: float
    most_correlated: str | None
    threshold: float
    sample_start: dt.datetime | None
    sample_end: dt.datetime | None
    n_periods: int
    n_names: int
    orthogonalized: bool
    reason: str
    sample_digest: str

    def describe(self) -> str:
        """One-line human summary."""
        span = (
            f"{self.sample_start:%Y-%m-%d}..{self.sample_end:%Y-%m-%d}"
            if self.sample_start and self.sample_end
            else "unknown sample"
        )
        vs = ", ".join(self.tested_against) or "(empty pool)"
        flag = " [orthogonalized]" if self.orthogonalized else ""
        return (
            f"{self.factor}: {self.verdict}{flag} | max|rho|={self.max_abs_corr:.3f} "
            f"vs [{vs}] | {span} | {self.n_periods} periods x {self.n_names} names"
        )


@dataclass
class FactorPool:
    """A set of admitted factors with an explicit admission log.

    Mutable by design -- it is a ledger that grows. Use :meth:`propose` rather
    than appending to :attr:`factors` directly, so nothing enters without a
    recorded justification.
    """

    factors: list[str] = field(default_factory=list)
    log: list[AdmissionRecord] = field(default_factory=list)
    threshold: float = 0.7
    reject_above: float = 0.95

    def propose(
        self,
        df: Frame,
        candidate: str,
        *,
        method: str = "spearman",
        ortho_method: str = "schmidt",
        ts: str = TS,
        min_names: int = 20,
        corr: pl.DataFrame | None = None,
    ) -> tuple[ScreenResult, Frame]:
        """Test ``candidate``, record the decision, and admit if warranted.

        Returns
        -------
        ``(result, frame)`` where ``frame`` carries the residualised candidate
        when the verdict was ``orthogonalize``, and is the input otherwise.

        Notes
        -----
        The frame is returned rather than mutated so the caller decides whether
        to keep the residualised version. A rejected candidate leaves the pool
        untouched but is still logged -- knowing what was tried and refused is
        as valuable as knowing what passed.
        """
        lf = to_polars(df)
        res = screen(
            lf,
            self.factors,
            candidate,
            threshold=self.threshold,
            reject_above=self.reject_above,
            method=method,
            ts=ts,
            min_names=min_names,
            corr=corr,
        )

        out: Frame = df
        if res.verdict == "orthogonalize":
            out = orthogonalize(lf, candidate, self.factors, method=ortho_method, ts=ts)

        self.log.append(self._record(lf, res, ts))
        if res.admitted:
            self.factors.append(candidate)
        return res, out

    def _record(self, lf: pl.DataFrame, res: ScreenResult, ts: str) -> AdmissionRecord:
        stamps = lf[ts]
        lo = stamps.min() if lf.height else None
        hi = stamps.max() if lf.height else None
        n_names = int(lf[SYMBOL].n_unique()) if SYMBOL in lf.columns else 0
        return AdmissionRecord(
            factor=res.candidate,
            verdict=res.verdict,
            tested_against=tuple(self.factors),
            max_abs_corr=res.max_abs_corr,
            most_correlated=res.most_correlated,
            threshold=res.threshold,
            sample_start=lo,  # type: ignore[arg-type]
            sample_end=hi,  # type: ignore[arg-type]
            n_periods=int(stamps.n_unique()) if lf.height else 0,
            n_names=n_names,
            orthogonalized=res.verdict == "orthogonalize",
            reason=res.reason,
            sample_digest=stable_hash("sample-v1", str(lo), str(hi), n_names, lf.height),
        )

    def admission_log(self) -> pl.DataFrame:
        """The log as a frame, for a tearsheet or an audit."""
        if not self.log:
            return pl.DataFrame(
                schema={"factor": pl.Utf8, "verdict": pl.Utf8, "max_abs_corr": pl.Float64}
            )
        return pl.DataFrame(
            [
                {
                    "factor": r.factor,
                    "verdict": r.verdict,
                    "tested_against": ", ".join(r.tested_against),
                    "max_abs_corr": r.max_abs_corr,
                    "most_correlated": r.most_correlated,
                    "n_periods": r.n_periods,
                    "n_names": r.n_names,
                    "orthogonalized": r.orthogonalized,
                    "reason": r.reason,
                    "sample_digest": r.sample_digest,
                }
                for r in self.log
            ]
        )

    def __len__(self) -> int:
        return len(self.factors)

    def __contains__(self, name: object) -> bool:
        return name in self.factors

    def __repr__(self) -> str:
        return f"FactorPool({len(self.factors)} admitted, {len(self.log)} tested)"
