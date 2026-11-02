r"""Run the C++ engine over one day: the same CLI and the same output file as ``run.py``, one process, stock chunks on threads.

    python research/engine/run_cpp.py --date 20261009 --universe pm:<jz> --prev <prev jz> --threads 16

Steps: export the day's events (``export.py``, one DuckDB query) + ``rules.tsv`` (from ``rules.EMAS``) + ``prev.bin``
into ``<work>/<date>/``, build ``cpp/build/engine`` if it is missing or older than its sources, run it, gzip its JSON to
``<out>/engine_<date>.json.gz``. ``verify.py`` reads that file unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export import load_events, write_events, write_prev, write_rules
from rules import EMAS
from run import CATALOG, DEFAULT_STOP, load_prev
from stream import universe_from_state

CPP = Path(__file__).resolve().parent / "cpp"


def ensure_binary(cpp: Path = CPP) -> Path:
    """Build cpp/build/engine when it is missing or older than any source; return its path."""
    exe = cpp / "build" / "engine"
    srcs = list(cpp.glob("*.hpp")) + list(cpp.glob("*.cpp")) + [cpp / "CMakeLists.txt"]
    if exe.exists() and exe.stat().st_mtime >= max(s.stat().st_mtime for s in srcs):
        return exe
    print(f"building {exe} ...", flush=True)
    exe.parent.mkdir(exist_ok=True)
    if shutil.which("cmake"):
        subprocess.run(["cmake", "-S", str(cpp), "-B", str(cpp / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(cpp / "build"), "-j4"], check=True, stdout=subprocess.DEVNULL)
    else:  # no cmake on the box: the one-TU build with the same flags
        subprocess.run(["g++", "-std=c++17", "-O2", "-DNDEBUG", "-ffp-contract=off", "-pthread", str(cpp / "main.cpp"), "-o", str(exe)], check=True)
    return exe


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--universe", required=True, help="pm:<state .jz> | thr:<samplerR parquet> | all")
    ap.add_argument("--stop", type=float, help="local second of day PM stopped at (default: from the state file, else 15:02:30)")
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--threads", type=int, default=8, help="engine threads (stock chunks)")
    ap.add_argument("--prev", type=Path, help="previous day's state (PM .jz or engine file) for the |c<N> carry variants")
    ap.add_argument("--out", type=Path, default=Path("data/engine"))
    ap.add_argument("--work", type=Path, default=Path("data/engine/work"), help="where events.bin / rules.tsv / prev.bin go")
    ap.add_argument("--keep", action="store_true", help="keep the work files (default: delete events.bin after the run)")
    a = ap.parse_args()
    t0 = time.perf_counter()
    exe = ensure_binary()
    con = duckdb.connect(str(a.catalog), read_only=True)
    kind, _, src = a.universe.partition(":")
    stop = a.stop
    if kind == "pm":
        sids, pm_stop = universe_from_state(Path(src))
        stop = stop or pm_stop
    elif kind == "thr":
        sids = sorted(int(x) for x in pl.read_parquet(src)["sid"].to_list())
    else:
        sids = [int(r[0]) for r in con.sql(f"SELECT DISTINCT symbol_id FROM raw_quotation WHERE date = {int(a.date)} ORDER BY 1").fetchall()]
    stop = stop or DEFAULT_STOP
    work = a.work / str(a.date)
    work.mkdir(parents=True, exist_ok=True)
    names = write_rules(EMAS, work / "rules.tsv")
    n_prev = write_prev(load_prev(a.prev), names, work / "prev.bin")
    ev = load_events(con, int(a.date), sids, stop)
    con.close()
    write_events(ev, work / "events.bin")
    t1 = time.perf_counter()
    print(f"{a.date}: {len(ev)} events, {len(sids)} stocks wanted, {len(names)} rules, {n_prev} carry seeds exported ({t1 - t0:.0f}s)", flush=True)
    cmd = [str(exe), str(work / "events.bin"), str(work / "rules.tsv"), str(work / "state.json"), "--prev", str(work / "prev.bin"),
           "--threads", str(a.threads)]
    subprocess.run(cmd, check=True)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"engine_{a.date}.json.gz"
    with (work / "state.json").open("rb") as f, gzip.open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
    if not a.keep:
        (work / "events.bin").unlink()
    print(f"{a.date}: stop {stop:.3f} -> {dst} ({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
