"""Background resource sampler for any long-running task.

Wrap a task in :class:`ResourceMonitor` and get back wall time, per-core CPU utilisation, peak
RSS of the whole process tree, disk read throughput and (when an NVIDIA GPU and ``pynvml`` are
present) GPU utilisation and memory. Every run can be appended to a Parquet history so a
regression shows up as a number, not a feeling.

Sampling runs on its own thread and only reads OS counters, so the overhead is negligible next
to the work being measured.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Self

import polars as pl
import psutil

__all__ = ["ResourceMonitor", "RunStats", "append_history"]


@dataclass(frozen=True)
class RunStats:
    """Summary of one monitored run. ``n_samples`` says how many observations back the numbers."""

    task: str
    wall_s: float
    rows: int
    rows_per_s: float
    cpu_avg_pct: float
    cpu_peak_pct: float
    cores: int
    per_core_avg_pct: list[float]
    rss_peak_mb: float
    read_mb: float
    read_mb_s: float
    gpu_util_avg_pct: float | None
    gpu_mem_peak_mb: float | None
    n_samples: int
    host: str = field(default_factory=platform.node)
    started_at: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        """One human-readable summary line."""
        gpu = (
            f" gpu={self.gpu_util_avg_pct:.0f}%/{self.gpu_mem_peak_mb:.0f}MB"
            if self.gpu_util_avg_pct is not None
            else ""
        )
        return (
            f"[{self.task}] wall={self.wall_s:.2f}s rows={self.rows:,} "
            f"({self.rows_per_s / 1e6:.2f}M/s) cpu avg={self.cpu_avg_pct:.0f}% "
            f"peak={self.cpu_peak_pct:.0f}% of {self.cores} cores rss_peak={self.rss_peak_mb:.0f}MB "
            f"read={self.read_mb:.0f}MB ({self.read_mb_s:.0f}MB/s){gpu} n={self.n_samples}"
        )


def _tree_rss(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _tree_read_bytes(proc: psutil.Process) -> int:
    total = 0
    for p in [proc, *proc.children(recursive=True)]:
        try:
            io = p.io_counters()
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
        # read_chars counts reads served from the page cache too; that is what the task consumed.
        total += getattr(io, "read_chars", io.read_bytes)
    return total


class _Gpu:
    def __init__(self) -> None:
        self.handles: list[Any] = []
        try:
            import pynvml  # type: ignore[import-not-found]

            pynvml.nvmlInit()
            self._nv = pynvml
            self.handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:
            self.handles = []

    def sample(self) -> tuple[float, float] | None:
        if not self.handles:
            return None
        util = [self._nv.nvmlDeviceGetUtilizationRates(h).gpu for h in self.handles]
        mem = [self._nv.nvmlDeviceGetMemoryInfo(h).used for h in self.handles]
        return sum(util) / len(util), sum(mem) / 2**20


class ResourceMonitor:
    """Context manager sampling CPU / RSS / IO / GPU every ``interval`` seconds.

    Parameters
    ----------
    task:
        Name recorded with the run (e.g. ``"W1.decode_day"``).
    interval:
        Sampling period in seconds.
    params:
        Free-form parameters stored with the run (date, thread count, ...).

    Examples
    --------
    >>> with ResourceMonitor("W1.decode_day") as mon:  # doctest: +SKIP
    ...     n = do_work()
    ...     mon.rows = n
    >>> print(mon.stats.line())  # doctest: +SKIP
    """

    def __init__(
        self, task: str, *, interval: float = 0.25, params: dict[str, Any] | None = None
    ) -> None:
        self.task = task
        self.interval = interval
        self.params = params or {}
        self.rows = 0
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._cpu: list[list[float]] = []
        self._rss: list[int] = []
        self._gpu: list[tuple[float, float]] = []
        self._gpu_dev = _Gpu()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.stats: RunStats | None = None

    def _run(self) -> None:
        psutil.cpu_percent(percpu=True)
        while not self._stop.wait(self.interval):
            self._cpu.append(psutil.cpu_percent(percpu=True))
            self._rss.append(_tree_rss(self._proc))
            g = self._gpu_dev.sample()
            if g is not None:
                self._gpu.append(g)

    def __enter__(self) -> Self:
        """Start sampling."""
        self._started = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        self._read0 = _tree_read_bytes(self._proc)
        self._t0 = time.perf_counter()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop sampling and fill :attr:`stats`."""
        wall = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join()
        read_mb = (_tree_read_bytes(self._proc) - self._read0) / 2**20
        cores = psutil.cpu_count() or 1
        per_core = (
            [sum(s[i] for s in self._cpu) / len(self._cpu) for i in range(cores)]
            if self._cpu
            else [0.0] * cores
        )
        totals = [sum(s) / cores for s in self._cpu] or [0.0]
        self.stats = RunStats(
            task=self.task,
            wall_s=wall,
            rows=self.rows,
            rows_per_s=self.rows / wall if wall > 0 else 0.0,
            cpu_avg_pct=sum(totals) / len(totals),
            cpu_peak_pct=max(totals),
            cores=cores,
            per_core_avg_pct=per_core,
            rss_peak_mb=max(self._rss, default=0) / 2**20,
            read_mb=read_mb,
            read_mb_s=read_mb / wall if wall > 0 else 0.0,
            gpu_util_avg_pct=(sum(g[0] for g in self._gpu) / len(self._gpu)) if self._gpu else None,
            gpu_mem_peak_mb=max(g[1] for g in self._gpu) if self._gpu else None,
            n_samples=len(self._cpu),
            started_at=self._started,
            params=self.params,
        )


def append_history(stats: RunStats, path: str | Path) -> None:
    """Append one run to the Parquet history at ``path`` (created if missing)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(stats)
    row["params"] = repr(row["params"])
    new = pl.DataFrame([row])
    if p.exists():
        new = pl.concat([pl.read_parquet(p), new], how="diagonal_relaxed")
    new.write_parquet(p)
