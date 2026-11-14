r"""One day, one command: ticks -> the 144 model columns at a time grid -> PM's model scores, everything computed in C++.

The three programs that read the day's ticks (whole book, state engine, base quantities) do not depend on each other, so
they run side by side; then the formula layer (featvec), then the model (lgbpred). Each step's output goes to
``<out>/log/<step>_<D>.txt``; the last lines and the seconds of every step are printed.

    python research/featgen/day.py --date 20261112 --source dysl2 --state $SEED/20261112.jz --seed-dir $SEED \
        --catalog $DATA/catalog_dysl2.duckdb --grid 3 --side $O/side_20261112.npz --cal $O/cal_20261112.json \
        --models $DATA/models_gen/B --out data/day_3s [--pq research/featgen/cpp/build/cmake/lgbpred_pq] [--serial]

Outputs under ``--out``: book/date=<D>.parquet, engine/marks_<D>.parquet, base/base_<D>.parquet, vec/vec_<D>.parquet,
pred/pred_<D>.parquet. ``--side`` / ``--cal`` are ic_score --dump-x's side tables of the day (CED industry and exposures,
training-CDF points); ``--prev`` (yesterday's state for the engine's carry rules) defaults to the latest file before the
day in ``--seed-dir``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"


def start(name: str, cmd: list[str], log: Path, env: dict[str, str]) -> tuple[str, subprocess.Popen, Path, float]:
    return name, subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env), log, time.perf_counter()  # noqa: S603


def finish(job: tuple[str, subprocess.Popen, Path, float]) -> float:
    """Wait for one step; its seconds. A failed step ends the run with the tail of its log."""
    name, p, log, t0 = job
    rc = p.wait()
    sec = time.perf_counter() - t0
    tail = log.read_text(errors="replace").strip().splitlines()[-1:] or [""]
    print(f"  {name:<8} {sec:6.1f}s  {tail[0][:200]}", flush=True)
    if rc:
        print("\n".join(log.read_text(errors="replace").splitlines()[-12:]))
        raise SystemExit(f"{name} failed (exit {rc}), log {log}")
    return sec


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--source", default=os.environ.get("CRUCIBLE_SOURCE", ""), help="CRUCIBLE_SOURCE of the tick loaders: feitu | dysl2 | pmmp")
    ap.add_argument("--state", type=Path, required=True, help="PM's state file of the day (.jz): the stocks and the stop time")
    ap.add_argument("--seed-dir", type=Path, required=True, help="directory of .jz files (yesterday's state)")
    ap.add_argument("--prev", type=Path, help="yesterday's state for the engine (default: the latest before the day in --seed-dir)")
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--marks", help="ic_score's grid spec: five | rand:N[:seed] | a comma list")
    ap.add_argument("--grid", type=float, help="instead of --marks: every GRID seconds from GRID to 14220")
    ap.add_argument("--side", type=Path, required=True)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--models", type=Path, help="model root (month folders); without it the run stops after the vectors")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--pq", type=Path, help="lgbpred_pq (CMake build with Arrow): the model step reads and writes parquet itself")
    ap.add_argument("--float32", action="store_true", help="round the bases to float32 as ic_score's cache does")
    ap.add_argument("--out", type=Path, default=Path("data/day"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--serial", action="store_true", help="run the three tick readers one after another (for timing comparisons)")
    a = ap.parse_args()
    D, out = a.date, a.out
    (out / "log").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, **({"CRUCIBLE_SOURCE": a.source} if a.source else {}))
    py = [sys.executable, "-u"]
    grid = ["--grid", repr(a.grid)] if a.grid else ["--marks", a.marks or "five"]
    marks = ",".join(repr(float(x)) for x in np.arange(a.grid, 14220.0 + 1e-9, a.grid)) if a.grid else (a.marks or "five")
    prev = a.prev or next(iter(sorted((p for p in a.seed_dir.glob("2*.jz") if p.stem < D), reverse=True)), None)
    th = str(a.threads)
    readers = [
        ("book", [*py, str(HERE / "book_day.py"), "--date", D, "--state", str(a.state), *grid, "--out", str(out / "book"),
                  "--work", str(out / "work/book"), "--threads", th]),
        ("engine", [*py, str(ENGINE / "run_cpp.py"), "--date", D, "--universe", f"pm:{a.state}", *(["--prev", str(prev)] if prev else []),
                    "--catalog", str(a.catalog), "--marks", marks, "--threads", th, "--out", str(out / "engine"), "--work", str(out / "work/engine")]),
        ("base", [*py, str(HERE / "run_day.py"), "--date", D, "--state", str(a.state), "--seed-dir", str(a.seed_dir), "--catalog", str(a.catalog),
                  *grid, "--out", str(out / "base"), "--work", str(out / "work/base"), "--threads", th]),
    ]
    t0 = time.perf_counter()
    print(f"{D}: tick readers {'one after another' if a.serial else 'side by side'}", flush=True)
    if a.serial:
        secs = [finish(start(n, c, out / "log" / f"{n}_{D}.txt", env)) for n, c in readers]
    else:
        secs = [finish(j) for j in [start(n, c, out / "log" / f"{n}_{D}.txt", env) for n, c in readers]]
    t1 = time.perf_counter()
    finish(start("vec", [*py, str(HERE / "vec_day.py"), "--date", D, "--base", str(out / f"base/base_{D}.parquet"), "--full-book",
                         str(out / f"book/date={D}.parquet"), "--engine", str(out / f"engine/marks_{D}.parquet"), "--side", str(a.side), "--cal", str(a.cal),
                         *(["--float32"] if a.float32 else []), "--out", str(out / "vec"), "--threads", th], out / "log" / f"vec_{D}.txt", env))
    t2 = time.perf_counter()
    if a.models:
        finish(start("pred", [*py, str(HERE / "pred_day.py"), "--date", D, "--x", str(out / f"vec/vec_{D}.parquet"), "--models", str(a.models),
                              "--feature-csv", str(a.feature_csv), *(["--pq", str(a.pq)] if a.pq else []), "--out", str(out / "pred"),
                              "--work", str(out / "work/pred"), "--threads", th], out / "log" / f"pred_{D}.txt", env))
    t3 = time.perf_counter()
    print(f"{D}: tick readers {t1 - t0:.0f}s (their own seconds add up to {sum(secs):.0f}), formula layer {t2 - t1:.0f}s, model {t3 - t2:.0f}s, "
          f"total {t3 - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
