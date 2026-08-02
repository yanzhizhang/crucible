"""Covariance estimation for portfolio construction.

The sample covariance of 5000 names estimated from 250 observations is
singular and, worse, its extreme eigenvalues are almost pure noise. An
optimizer handed that matrix will happily find the "minimum variance"
portfolio that loads entirely on the noisiest eigenvector -- the classic
error-maximisation failure, where a better-looking objective produces a worse
portfolio.

Two defences are provided. Ledoit-Wolf shrinkage pulls the sample matrix
toward a structured target, and a factor model imposes structure directly. For
a wide A-share cross-section the factor model is usually the right answer; for
a few dozen instruments (futures books) shrinkage is enough.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["sample_covariance", "ledoit_wolf", "factor_covariance", "nearest_psd"]


def sample_covariance(returns: np.ndarray, *, ddof: int = 1) -> np.ndarray:
    """Plain sample covariance of a ``(T, N)`` return matrix.

    Reliable only when ``T`` comfortably exceeds ``N``. Below roughly ``T > 3N``
    the estimate is dominated by noise, and at ``T < N`` it is singular --
    provided here as a baseline and for the small-``N`` case, not as a default.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"returns must be 2-D (T, N), got shape {x.shape}")
    if x.shape[0] <= ddof:
        raise ValueError(f"need more than {ddof} observations, got {x.shape[0]}")
    return np.cov(x, rowvar=False, ddof=ddof)


def ledoit_wolf(returns: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a constant-correlation target.

    Returns
    -------
    ``(covariance, shrinkage_intensity)``. The intensity is estimated from the
    data, in ``[0, 1]``; values near 1 mean the sample matrix carried almost no
    usable information and the target is doing the work -- worth knowing before
    trusting the optimizer's output.

    Notes
    -----
    The constant-correlation target keeps each asset's own variance and
    replaces the correlation matrix with a single average correlation. That
    preserves the part of the sample estimate that is well measured
    (individual volatilities) and regularises the part that is not (pairwise
    correlations).
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    if t < 2:
        raise ValueError(f"need at least 2 observations, got {t}")

    x = x - x.mean(axis=0)
    sample = x.T @ x / t

    var = np.diag(sample)
    std = np.sqrt(np.maximum(var, 1e-300))
    outer_std = np.outer(std, std)
    corr = sample / np.where(outer_std > 0, outer_std, 1.0)

    off = ~np.eye(n, dtype=bool)
    r_bar = float(corr[off].mean()) if n > 1 else 0.0
    target = r_bar * outer_std
    np.fill_diagonal(target, var)

    # pi: variance of the sample covariance entries
    x2 = x**2
    phi_mat = (x2.T @ x2) / t - sample**2
    pi = float(phi_mat.sum())

    # gamma: squared distance between sample and target
    gamma = float(((sample - target) ** 2).sum())
    if gamma <= 0:
        return target, 1.0

    # rho: covariance between sample entries and the target's estimation error
    rho_diag = float(np.diag(phi_mat).sum())
    term = ((x**3).T @ x) / t - var[:, None] * sample
    np.fill_diagonal(term, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(outer_std > 0, np.divide.outer(std, std), 0.0)
    rho_off = float((r_bar / 2.0) * np.nan_to_num(ratio * term + ratio.T * term.T).sum())
    rho = rho_diag + rho_off

    intensity = float(np.clip((pi - rho) / gamma / t, 0.0, 1.0))
    return intensity * target + (1.0 - intensity) * sample, intensity


def factor_covariance(
    returns: np.ndarray,
    exposures: np.ndarray,
    *,
    min_specific_var: float = 1e-8,
) -> np.ndarray:
    """Structured covariance from a factor model.

    ``Sigma = B F B' + D`` where ``B`` is the ``(N, K)`` exposure matrix, ``F``
    the ``(K, K)`` factor covariance and ``D`` the diagonal specific variance.

    This is the estimator that scales: it needs ``K(K+1)/2`` covariance
    parameters instead of ``N(N+1)/2``, so a 5000-name universe with 40 factors
    estimates 820 numbers rather than 12.5 million. It is also PSD by
    construction as long as ``F`` is.

    Parameters
    ----------
    returns:
        ``(T, N)`` asset returns.
    exposures:
        ``(N, K)`` factor exposures, assumed known at the estimation date.
    min_specific_var:
        Floor on idiosyncratic variance. A name with zero estimated specific
        risk makes the optimizer treat it as a riskless arbitrage against its
        factor twin and take an unbounded position in it.
    """
    r = np.asarray(returns, dtype=float)
    b = np.asarray(exposures, dtype=float)
    t, n = r.shape
    if b.shape[0] != n:
        raise ValueError(f"exposures has {b.shape[0]} rows but returns has {n} assets")

    # Cross-sectional regression each period recovers factor returns.
    f_ret = np.empty((t, b.shape[1]))
    for i in range(t):
        beta, *_ = np.linalg.lstsq(b, r[i], rcond=None)
        f_ret[i] = beta

    f_cov = np.cov(f_ret, rowvar=False, ddof=1)
    f_cov = np.atleast_2d(f_cov)

    resid = r - f_ret @ b.T
    specific = np.maximum(resid.var(axis=0, ddof=1), min_specific_var)

    return b @ f_cov @ b.T + np.diag(specific)


def nearest_psd(matrix: np.ndarray, *, epsilon: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues to make a symmetric matrix PSD.

    Shrinkage and factor models are PSD by construction, but a hand-assembled
    or interpolated covariance may not be, and most optimizers fail opaquely on
    an indefinite matrix rather than saying so.
    """
    m = np.asarray(matrix, dtype=float)
    m = (m + m.T) / 2.0
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, epsilon)
    out = vecs @ np.diag(vals) @ vecs.T
    return (out + out.T) / 2.0
