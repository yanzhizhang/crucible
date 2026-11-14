r"""Many days through the C++ chain, each one checked against PM's own day-end state: the backfill of 3-second vectors.

Per day: side tables (side_day.py), the whole chain (day.py: book, state engine, bases, formula layer, model), then the
engine's day-end state against PM's ``.jz`` of the day (research/engine/verify.py --pm). A day needs PM's state file
(the stocks and the stop time) under --state-dir; the tick source is picked by date (--dysl2-from: DataYes from that day
on, feitu before). Intermediates (book, engine marks, bases) are deleted after each day unless --keep lists them; the
vectors are ~1 GB a day at 3 s, so --keep decides the disk cost.

    python research/featgen/backfill_days.py --dates 20261110,20261111,20261112 --grid 3 --out data/backfill_3s
    python research/featgen/backfill_days.py --dates 20260901-20261113 --grid 3 --keep pred --out data/backfill_3s

One line per day: seconds per step, the verify summary (keys at 1.0 on SZSE / all), rows written.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINE = HERE.parent / "engine"
DATA = Path("/work/crucible_data")


def days(spec: str, state_dirs: list[Path]) -> list[str]:
    """A comma list, or FROM-TO over the days that have a state file in any of the dirs."""
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return sorted({p.stem for sd in state_dirs for p in sd.glob("2*.jz") if lo <= p.stem <= hi})
    return [x.strip() for x in spec.split(",") if x.strip()]


def run(cmd: list[str], log: Path, env: dict[str, str]) -> tuple[int, float, str]:
    t0 = time.perf_counter()
    with log.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False).returncode  # noqa: S603
    tail = log.read_text(errors="replace").strip().splitlines()
    return rc, time.perf_counter() - t0, tail[-1] if tail else ""


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True)
    ap.add_argument("--state-dir", default="/data/prod/h/HS300/p1_intrading_dcp,/work/crucible_data/seed_probe/HS300",
                    help="comma list of dirs of PM's day-end state files (.jz); a day takes the first dir that has its file")
    ap.add_argument("--dysl2-from", default="20261104", help="first day read from DataYes (dysl2); earlier days from feitu")
    ap.add_argument("--grid", type=float, default=3.0)
    ap.add_argument("--models", type=Path, default=DATA / "models_gen" / "B")
    ap.add_argument("--pq", type=Path, help="lgbpred_pq for the model step")
    ap.add_argument("--out", type=Path, default=Path("data/backfill_3s"))
    ap.add_argument("--keep", default="pred", help="comma list of outputs kept per day: vec, pred, base, engine, book")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed-map", default="", help="D=YYYYMMDD,...: day D's yesterday state is that day's .jz (what 224's PM read, e.g. a stale sod link)")
    a = ap.parse_args()
    keep = {x for x in a.keep.split(",") if x}
    py = [sys.executable, "-u"]
    log_dir = a.out / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    dirs = [Path(x) for x in a.state_dir.split(",") if x]
    todo = days(a.dates, dirs)
    print(f"{len(todo)} days: {todo[0] if todo else '-'} .. {todo[-1] if todo else '-'}; keep {sorted(keep)} -> {a.out}", flush=True)
    for d in todo:
        st = next((sd / f"{d}.jz" for sd in dirs if (sd / f"{d}.jz").exists()), None)
        if st is None:
            print(f"{d}: no state file in {[str(x) for x in dirs]}, skipped", flush=True)
            continue
        src = "dysl2" if d >= a.dysl2_from else "feitu"
        cat = DATA / ("catalog.duckdb" if src == "feitu" else f"catalog_{src}.duckdb")
        env = dict(os.environ, CRUCIBLE_SOURCE=src)
        day_out = a.out / "days" / d
        side = a.out / "side"
        rc1, s1, t1 = run([*py, str(HERE / "side_day.py"), "--date", d, "--state", str(st), "--models", str(a.models), "--out", str(side)],
                          log_dir / f"side_{d}.txt", env)
        if rc1:
            print(f"{d}: side tables FAILED ({t1[:160]})", flush=True)
            continue
        seed_dir = st.parent
        smap = dict(x.split("=") for x in a.seed_map.split(",") if "=" in x)
        if d in smap:  # a dir holding only the mapped file: both the engine's --prev and run_day's seed pick it as "latest before D"
            src_jz = next((sd / f"{smap[d]}.jz" for sd in dirs if (sd / f"{smap[d]}.jz").exists()), None)
            if src_jz is None:
                print(f"{d}: seed-map file {smap[d]}.jz not found, skipped", flush=True)
                continue
            seed_dir = a.out / "seedmap" / d
            seed_dir.mkdir(parents=True, exist_ok=True)
            link = seed_dir / src_jz.name
            if not link.exists():
                link.symlink_to(src_jz.resolve())
        rc2, s2, t2 = run([*py, str(HERE / "day.py"), "--date", d, "--source", src, "--state", str(st), "--seed-dir", str(seed_dir), "--catalog", str(cat),
                           "--grid", repr(a.grid), "--side", str(side / f"side_{d}.npz"), "--cal", str(side / f"cal_{d}.json"), "--models", str(a.models),
                           *(["--pq", str(a.pq)] if a.pq else []), "--out", str(day_out), "--threads", str(a.threads)], log_dir / f"day_{d}.txt", env)
        if rc2:
            print(f"{d}: chain FAILED in {s2:.0f}s ({t2[:200]}); log {log_dir / f'day_{d}.txt'}", flush=True)
            continue
        rc3, s3, t3 = run([*py, str(ENGINE / "verify.py"), "--date", d, "--engine", str(day_out / f"engine/engine_{d}.json.gz"), "--pm", str(st),
                           "--catalog", str(cat), "--csv", str(a.out / f"verify_{d}.csv")], log_dir / f"verify_{d}.txt", env)
        vf = a.out / f"verify_{d}.csv"
        if not rc3 and vf.exists():  # keys whose day-end value equals PM's on every SZSE stock / on every stock
            import polars as pl

            v = pl.read_csv(vf)
            t3 = (f"verify: {v.height} keys, SZSE all equal {int((v['SZSE'] >= 1).sum())}, all stocks equal {int((v['all'] >= 1).sum())}, "
                  f"median share {v['all'].median():.4f}")
        for what, path in (("vec", day_out / "vec"), ("pred", day_out / "pred"), ("base", day_out / "base"), ("engine", day_out / "engine"),
                           ("book", day_out / "book")):
            if what not in keep and path.exists():
                shutil.rmtree(path)
        shutil.rmtree(day_out / "work", ignore_errors=True)
        size = sum(f.stat().st_size for f in day_out.rglob("*") if f.is_file()) / 2**20
        print(f"{d}: {src}{f' seed {smap[d]}' if d in smap else ''}; side {s1:.0f}s, chain {s2:.0f}s, verify {s3:.0f}s{' FAILED' if rc3 else ''}; {t3[:150]}; kept {size:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
