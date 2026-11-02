r"""The C++ scoring kernel behind ``implied_match`` (``cpp/score_kernel.cpp``), built on first use and called through ctypes.

``load()`` returns the library or None (no g++ / build failed -> callers keep their numpy path). ``rank``, ``rank_cols``
and ``col_scores`` have the numpy signatures of ``implied_match``. ``python kernel.py`` self-tests the kernel against
the numpy versions on random data with NaNs, ties and zeros.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CPP = Path(__file__).resolve().parent / "cpp"
SRC = CPP / "score_kernel.cpp"
LIB = CPP / "build" / "libscore_kernel.so"
_lib: ctypes.CDLL | None | bool = False  # False = not tried yet


def load() -> ctypes.CDLL | None:
    """Build if stale, load once; None when the kernel is unavailable."""
    global _lib
    if _lib is not False:
        return _lib
    _lib = None
    try:
        if not LIB.exists() or LIB.stat().st_mtime < SRC.stat().st_mtime:
            if not shutil.which("g++"):
                return None
            LIB.parent.mkdir(exist_ok=True)
            subprocess.run(["g++", "-std=c++17", "-O2", "-shared", "-fPIC", "-ffp-contract=off", str(SRC), "-o", str(LIB)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        lib = ctypes.CDLL(str(LIB))
        dp, ip = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64)
        lib.ck_rank.argtypes = [dp, ctypes.c_int64, dp]
        lib.ck_rank_cols.argtypes = [dp, ctypes.c_int64, ctypes.c_int64, dp]
        lib.ck_col_scores.argtypes = [dp, dp, dp, ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_int64, ip, dp, dp, dp]
        _lib = lib
    except (OSError, subprocess.CalledProcessError):
        _lib = None
    return _lib


def _c(a: np.ndarray) -> np.ndarray:
    """1-D contiguous; 2-D in Fortran (column-contiguous) order, which the kernel indexes as B[c * n + i]."""
    return np.asfortranarray(a, dtype=np.float64) if a.ndim == 2 else np.ascontiguousarray(a, dtype=np.float64)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks, NaN kept."""
    lib = load()
    x = _c(x)
    out = np.empty_like(x)
    lib.ck_rank(_ptr(x), x.size, _ptr(out))
    return out


def rank_cols(B: np.ndarray) -> np.ndarray:
    """Ranks of every column."""
    lib = load()
    B = _c(B)
    out = np.empty(B.shape, dtype=np.float64, order="F")
    if B.shape[1]:
        lib.ck_rank_cols(_ptr(B), B.shape[0], B.shape[1], _ptr(out))
    return out


def col_scores(y: np.ndarray, B: np.ndarray, RB: np.ndarray | None, tol: float, min_n: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n, ratio, exact_scaled, spearman) per column, as implied_match.col_scores."""
    lib = load()
    y, B = _c(y), _c(B)
    n_cols = B.shape[1]
    n = np.zeros(n_cols, dtype=np.int64)
    ratio, exact, rho = (np.empty(n_cols) for _ in range(3))
    rbp = _ptr(_c(RB)) if RB is not None else ctypes.POINTER(ctypes.c_double)()
    if n_cols:
        lib.ck_col_scores(_ptr(y), _ptr(B), rbp, B.shape[0], n_cols, tol, min_n, n.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                          _ptr(ratio), _ptr(exact), _ptr(rho))
    return n, ratio, exact, rho


def selftest() -> None:
    """Kernel vs numpy on random data; exits 1 on a mismatch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import implied_match as im

    if load() is None:
        print("kernel unavailable (no g++ or build failed)")
        sys.exit(1)
    rng = np.random.default_rng(3)
    N, K = 5000, 40
    B = rng.normal(0, 1, (N, K))
    B[rng.random((N, K)) < 0.05] = np.nan
    B[:, 3] = np.round(B[:, 3])  # ties
    B[rng.random((N, K)) < 0.02] = 0.0
    y = 2.0 * np.nan_to_num(B[:, 7]) + rng.normal(0, 0.1, N)
    y[rng.random(N) < 0.03] = np.nan
    bad = 0
    r_np, r_c = im._rank_np(y), rank(y)
    bad += int(not np.array_equal(np.nan_to_num(r_np, nan=-1), np.nan_to_num(r_c, nan=-1)))
    RB_np, RB_c = im._rank_cols_np(B), rank_cols(B)
    bad += int(not np.array_equal(np.nan_to_num(RB_np, nan=-1), np.nan_to_num(RB_c, nan=-1)))
    a = im._col_scores_np(y, B, RB_np)
    b = col_scores(y, B, RB_c, im.TOL)
    for name, u, v, tol in (("n", a[0], b[0], 0), ("ratio", a[1], b[1], 1e-12), ("exact", a[2], b[2], 1e-12), ("rho", a[3], b[3], 1e-10)):
        d = np.nanmax(np.abs(np.asarray(u, float) - np.asarray(v, float))) if np.isfinite(u).any() else 0.0
        same_nan = np.array_equal(np.isnan(np.asarray(u, float)), np.isnan(np.asarray(v, float)))
        print(f"{name}: max |numpy - kernel| = {d:.3g}, nan pattern equal {same_nan}")
        bad += int(d > tol or not same_nan)
    print("kernel selftest", "FAILED" if bad else "ok")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    selftest()
