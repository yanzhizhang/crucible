r"""Run the C++ engine over one day: the same CLI and the same output file as ``run.py``, one process, stock chunks on threads.

    python research/engine/run_cpp.py --date 20261009 --universe pm:<jz> --prev <prev jz> --threads 16

Steps: export the day's events (``export.py``, one DuckDB query) + ``rules.tsv`` (from ``rules.EMAS``) + ``prev.bin``
into ``<work>/<date>/``, build ``cpp/build/engine`` if it is missing or older than its sources, run it, gzip its JSON to
``<out>/engine_<date>.json.gz``. ``verify.py`` reads that file unchanged.

``--marks five`` (or a comma list of trading-clock seconds) also writes the state as of each 5-minute mark (09:35 ... 14:55,
every event received by the mark applied) to ``<out>/marks_<date>.parquet``: one row per (sid, mark), each rule as
``<name>.sum`` with ``<name>.wgt`` = 1, the running scalars, and event counts since 09:30 -- ``ic_score``'s state input.
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
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lake as L  # --lake: tickaudit's merged ticks
from export import load_events, write_band, write_chan, write_events, write_prev, write_rules
from rules import EMAS
from run import CATALOG, DEFAULT_STOP, load_prev
from stream import universe_from_state

CPP = Path(__file__).resolve().parent / "cpp"
FIVE = np.arange(300.0, 14101.0, 300.0)  # trading-clock seconds 09:35 ... 14:55, as feature_intraday.MARKS


def mark_grid(spec: str) -> np.ndarray:
    """'five', 'rand:N[:seed]' (N random trading-clock seconds in (0, 13800), the same list as ic_score.mark_grid) or a comma list."""
    if spec == "five":
        return FIVE
    if spec.startswith("rand:"):
        n, *seed = (int(x) for x in spec[5:].split(":"))
        return np.sort(np.random.default_rng(seed[0] if seed else 0).choice(np.arange(1, 13800), n, replace=False)).astype(float)
    return np.array([float(x) for x in spec.split(",")])


def read_marks(path: Path) -> pl.DataFrame:
    """marks.bin -> one row per (sid, mark): sid, mark, then every key (rules as <name>.sum / <name>.wgt = 1)."""
    b = path.read_bytes()
    if b[:8] != b"CRUMK001":
        raise SystemExit(f"bad marks header in {path}")
    ns = int(np.frombuffer(b, "<u8", 1, 8)[0])
    nm, nk = (int(x) for x in np.frombuffer(b, "<u4", 2, 16))
    marks = np.frombuffer(b, "<f8", nm, 24)
    o = 24 + 8 * nm
    kl = int(np.frombuffer(b, "<u8", 1, o)[0])
    keys = b[o + 8: o + 8 + kl].decode().splitlines()
    rec = np.frombuffer(b, np.dtype([("sid", "<i8"), ("v", "<f8", (nm, nk))]), ns, o + 8 + kl)
    v = rec["v"].reshape(ns * nm, nk)
    cols = {"sid": np.repeat(rec["sid"], nm), "mark": np.tile(marks, ns)}
    for i, k in enumerate(keys):
        cols[k] = v[:, i]
        if k.endswith(".sum"):
            cols[k[:-4] + ".wgt"] = np.ones(ns * nm)
    return pl.DataFrame(cols)


def prev_channel(date: str, sids: list[int], hist: str, catalog: Path) -> dict[int, tuple[float, float]]:
    """sid -> (max high, min low) over the previous 3 trading days with data (featgen/zzeg_channel.day_hl, cached per day)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "featgen"))
    from zzeg_channel import DATA, day_hl  # noqa: PLC0415

    if hist == "auto":
        hist = "dysl2_raw,feitu_raw" if "dysl2" in catalog.name else "feitu_raw,dysl2_raw"
    raws = [DATA / x for x in hist.split(",") if x]
    days = sorted({p.name[5:] for r in raws for k in ("quotation", "transaction") for p in (r / f"kind={k}").glob("date=*") if p.name[5:] < date})
    got = []
    for d in reversed(days):
        hl = day_hl(d, sids, raws)
        if hl is not None and hl.height > 0.5 * len(sids):
            got.append(hl)
        if len(got) == 3:
            break
    if not got:
        return {}
    g = pl.concat(got).group_by("symbol_id").agg(pl.col("h").max(), pl.col("l").min())
    return {int(r[0]): (float(r[1]), float(r[2])) for r in g.iter_rows() if r[1] is not None and r[2] is not None}


def ensure_binary(cpp: Path = CPP) -> Path:
    """Build cpp/build/engine when it is missing or older than any source; return its path."""
    exe = cpp / "build" / "engine"
    srcs = list(cpp.glob("*.hpp")) + list(cpp.glob("families/*.hpp")) + list(cpp.glob("*.cpp")) + [cpp / "CMakeLists.txt"]
    if exe.exists() and exe.stat().st_mtime >= max(s.stat().st_mtime for s in srcs):
        return exe
    print(f"building {exe} ...", flush=True)
    exe.parent.mkdir(exist_ok=True)
    if shutil.which("cmake"):
        subprocess.run(["cmake", "-S", str(cpp), "-B", str(cpp / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(cpp / "build"), "-j4"], check=True, stdout=subprocess.DEVNULL)
    else:  # no cmake on the box: the one-TU build with the same flags
        subprocess.run(["g++", "-std=c++20", "-O2", "-DNDEBUG", "-ffp-contract=off", "-pthread", str(cpp / "main.cpp"), "-o", str(exe)], check=True)
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
    ap.add_argument("--seed-all", default="", help="c<N> / w<N> / wt<N>: seed EVERY ema from --prev (N s of decay), not only the |c<N> variants")
    ap.add_argument("--out", type=Path, default=Path("data/engine"))
    ap.add_argument("--work", type=Path, default=Path("data/engine/work"), help="where events.bin / rules.tsv / prev.bin go")
    ap.add_argument("--keep", action="store_true", help="keep the work files (default: delete events.bin after the run)")
    ap.add_argument("--chan-hist", default="auto", help="where the previous days' highs / lows come from (zzeg.c): auto = the catalog's source first, "
                    "or a comma list of raw dirs under /work/crucible_data; 'none' = no channel")
    ap.add_argument("--marks", help="'five' (09:35 ... 14:55), 'rand:N[:seed]' or trading-clock seconds a,b,...: also the state at each mark")
    ap.add_argument("--pm-mp", type=Path, help="PM's own feed: the decoded .mp cache dir (mp_load.py) instead of the catalog's ticks")
    ap.add_argument("--lake", type=Path, help="read tickaudit's lake (e.g. /work/tickaudit_lake) instead of --catalog: merged ticks + one source's snapshots")
    ap.add_argument("--snap-source", default="datayes", help="--lake: the tick/ source the snapshots come from (not merged yet)")
    ap.add_argument("--local-source", default="datayes", help="--lake: whose receive stamp orders the events (local_<src>), or 'first'")
    a = ap.parse_args()
    t0 = time.perf_counter()
    exe = ensure_binary()
    con = None if a.lake else duckdb.connect(str(a.catalog), read_only=True)
    kind, _, src = a.universe.partition(":")
    stop = a.stop
    if kind == "pm":
        sids, pm_stop = universe_from_state(Path(src))
        stop = stop or pm_stop
    elif kind == "thr":
        sids = sorted(int(x) for x in pl.read_parquet(src)["sid"].to_list())
    elif a.lake:
        sids = L.universe(int(a.date), a.snap_source, a.lake)
    else:
        sids = [int(r[0]) for r in con.sql(f"SELECT DISTINCT symbol_id FROM raw_quotation WHERE date = {int(a.date)} ORDER BY 1").fetchall()]
    stop = stop or DEFAULT_STOP
    work = a.work / str(a.date)
    work.mkdir(parents=True, exist_ok=True)
    names = write_rules(EMAS, work / "rules.tsv", a.seed_all)
    prev_state = load_prev(a.prev)
    n_prev = write_prev(prev_state, names, work / "prev.bin", a.seed_all)
    n_band = write_band(prev_state, work / "band.bin")
    if a.lake:
        print(f"{a.date}: lake {a.lake}, snapshots from {a.snap_source}; {L.missed(int(a.date), a.local_source, a.lake)}", flush=True)
        ev = L.load_events(int(a.date), sids, stop, a.snap_source, a.local_source, a.lake)
    elif a.pm_mp:
        from export_mp import constants, load_events_mp  # noqa: PLC0415
        import os  # noqa: PLC0415
        from export_mp import hybrid  # noqa: PLC0415
        if os.environ.get("PM_MP_MODE", "all") == "snap":
            ev = hybrid(load_events_mp(a.pm_mp, sids, stop, constants(con, int(a.date), sids), trades=False), load_events(con, int(a.date), sids, stop))
        else:
            ev = load_events_mp(a.pm_mp, sids, stop, constants(con, int(a.date), sids))
        con.close()
        print(f"{a.date}: PM feed {a.pm_mp}", flush=True)
    else:
        ev = load_events(con, int(a.date), sids, stop)
        con.close()
    write_events(ev, work / "events.bin")
    n_chan = 0
    if a.chan_hist != "none":
        chan = L.prev_channel(str(a.date), sids, a.lake) if a.lake and a.chan_hist == "auto" else \
            prev_channel(str(a.date), sids, a.chan_hist, a.catalog)
        n_chan = write_chan(chan, work / "chan.bin")
    t1 = time.perf_counter()
    print(f"{a.date}: {len(ev)} events, {len(sids)} stocks wanted, {len(names)} rules, {n_prev} carry seeds, {n_chan} channel rows exported ({t1 - t0:.0f}s)", flush=True)
    cmd = [str(exe), str(work / "events.bin"), str(work / "rules.tsv"), str(work / "state.json"), "--prev", str(work / "prev.bin"),
           "--threads", str(a.threads)] + (["--chan", str(work / "chan.bin")] if n_chan else []) + (["--band", str(work / "band.bin")] if n_band else [])
    marks = mark_grid(a.marks) if a.marks else None
    if marks is not None:
        cmd += ["--marks", ",".join(repr(float(x)) for x in marks), "--marks-out", str(work / "marks.bin")]
    subprocess.run(cmd, check=True)
    if marks is not None:
        a.out.mkdir(parents=True, exist_ok=True)
        mk = read_marks(work / "marks.bin")
        mk.write_parquet(a.out / f"marks_{a.date}.parquet")
        (work / "marks.bin").unlink()
        print(f"{a.date}: state at {marks.size} marks x {mk['sid'].n_unique()} stocks -> {a.out / f'marks_{a.date}.parquet'}")
    a.out.mkdir(parents=True, exist_ok=True)
    if getattr(a, "pm_mp", None):
        from export_mp import index_grids, inject_index  # noqa: PLC0415
        raw = Path(str(a.pm_mp).replace("pm_mp_cache", "pm_mp"))
        g = index_grids(raw)
        print(f"{a.date}: index grids {sorted(g)} from {raw} -> {inject_index(work / 'state.json', g)} stocks", flush=True)
    dst = a.out / f"engine_{a.date}.json.gz"
    with (work / "state.json").open("rb") as f, gzip.open(dst, "wb") as g:
        shutil.copyfileobj(f, g)
    if not a.keep:
        (work / "events.bin").unlink()
    print(f"{a.date}: stop {stop:.3f} -> {dst} ({time.perf_counter() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
