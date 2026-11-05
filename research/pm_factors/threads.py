"""Cap every thread pool a worker process may start (polars / rayon, OpenMP, BLAS, numba, Arrow).

Each library sizes its pool to the whole machine by default -- 256 threads per process on 97 -- so a pool of
dozens of workers ends up with tens of thousands of threads fighting over the cores. Call before the workers are
spawned: spawned children inherit the environment and read it when they import the libraries.
"""

from __future__ import annotations

import os

VARS = ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "ARROW_NUM_THREADS")


def cap(n: int) -> None:
    """Every library pool in processes started from now on gets ``n`` threads."""
    for k in VARS:
        os.environ[k] = str(max(1, n))
