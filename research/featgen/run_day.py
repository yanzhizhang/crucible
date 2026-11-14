r"""One day's base quantities from the C++ program, for PM's stocks, at a time grid; optionally checked against Python.

Loads the day exactly as ``ic_score`` does (same queries: ``dcp_ema.load(extra=True)``, ``tick_flow.load_orders``, the stop
from PM's state file, yesterday's state from ``--seed-dir``), packs it for ``featgen`` and writes
``<out>/base_<D>.parquet`` (sid, mark, one column per base quantity).

    CRUCIBLE_SOURCE=dysl2 python research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED \
        --marks rand:240 --check 8                 # the scoring marks, 8 stocks compared with the Python definitions
    ... --grid 3                                   # every 3 s of the session: the per-snapshot vectors' base
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featio as IO  # noqa: E402  (puts pm_factors on the path)

import dcp_build as B  # noqa: E402
import dcp_ema as E  # noqa: E402
import tick_flow as TF  # noqa: E402
from ic_score import mark_grid  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--state", type=Path, required=True, help="PM's state file of the day (.jz): the stocks and the stop time")
    ap.add_argument("--seed-dir", type=Path, help="directory of .jz files: the latest one before the day is yesterday's state")
    ap.add_argument("--catalog", type=Path, default=B.CATALOG)
    ap.add_argument("--marks", help="ic_score's grid spec: five | rand:N[:seed] | a comma list (trading-clock seconds)")
    ap.add_argument("--grid", type=float, help="instead of --marks: every GRID seconds from GRID to 14220")
    ap.add_argument("--out", type=Path, default=Path("data/featgen"))
    ap.add_argument("--work", type=Path, default=Path("data/featgen/work"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--check", type=int, default=0, help="compare this many stocks (evenly spread) with the Python definitions")
    a = ap.parse_args()
    t0 = time.perf_counter()
    T = np.arange(a.grid, 14220.0 + 1e-9, a.grid) if a.grid else mark_grid(a.marks or "five")
    pm = B.load_state(a.state)
    tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
    stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
    sids = sorted(pm["sid"].unique().to_list())
    seed: dict[int, dict[str, float]] = {}
    if a.seed_dir:
        fs = sorted(p for p in a.seed_dir.glob("2*.jz") if p.stem < a.date)
        if fs:
            with gzip.open(fs[-1]) as fh:
                raw = json.loads(fh.read())
            seed = {int(k): {x: float(kv[x]) for x in IO.SEED if isinstance(kv.get(x), (int, float))} for k, kv in raw.items()}
            print(f"{a.date}: yesterday's state from {fs[-1].name} for {len(seed)} stocks")
    con = duckdb.connect(str(a.catalog), read_only=True)
    up = E.load(con, int(a.date), sids, stop, extra=True)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    orders = {int(g["sid"][0]): g for g in TF.load_orders(con, int(a.date), sids, stop).partition_by("sid")}
    con.close()
    miss = [c for c in IO.FAST if fast and c not in next(iter(fast.values())).columns]
    print(f"{a.date}: {len(sids)} stocks, stop {stop:.3f}, {T.size} times; snapshots {sum(g.height for g in fast.values()):,}, trades "
          f"{sum(g.height for g in slow.values()):,}, orders {sum(g.height for g in orders.values()):,}; snapshot columns missing: {miss or 'none'} "
          f"({time.perf_counter() - t0:.0f}s load)", flush=True)
    stocks = [{"sid": s, "gf": fast.get(s), "gs": slow.get(s), "go": orders.get(s), "seed": seed.get(s, {})} for s in sids]
    work = a.work / a.date
    work.mkdir(parents=True, exist_ok=True)
    inp, outb, names = work / "in.bin", work / "out.bin", work / "names.txt"
    t1 = time.perf_counter()
    IO.write_inputs(inp, T, stocks)
    t2 = time.perf_counter()
    IO.run(inp, outb, names, a.threads)
    t3 = time.perf_counter()
    nm, got = IO.read_output(outb, names)
    cols = {"sid": np.repeat(np.array(list(got), np.int64), T.size), "mark": np.tile(T, len(got))}
    for i, k in enumerate(nm):
        cols[k] = np.concatenate([v[i] for v in got.values()])
    df = pl.DataFrame(cols)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"base_{a.date}.parquet"
    df.write_parquet(dst)
    fin = df.select(pl.exclude("sid", "mark").is_finite().mean()).row(0)
    low = sorted(zip(fin, nm))[:6]
    print(f"{a.date}: pack {t2 - t1:.1f}s, featgen {t3 - t2:.1f}s; {df.height:,} rows x {len(nm)} base quantities -> {dst}; "
          f"finite share lowest: {', '.join(f'{k} {v:.3f}' for v, k in low)}", flush=True)
    inp.unlink()
    outb.unlink()
    if a.check:
        pick = [sids[i] for i in np.unique(np.linspace(0, len(sids) - 1, a.check).astype(int))]
        t4 = time.perf_counter()
        ref = {s: IO.reference(fast.get(s), slow.get(s), orders.get(s), seed.get(s, {}), T) for s in pick}
        t5 = time.perf_counter()
        bad, lines = IO.compare(nm, {s: got[s] for s in pick}, ref)
        print("\n".join(ln for ln in lines if not ln.startswith("  ok ") or "cells differing      0" not in ln))
        print(f"{a.date}: {len(pick)} stocks checked against Python ({(t5 - t4) / len(pick):.1f}s a stock in Python, "
              f"{(t3 - t2) / len(sids):.3f}s a stock in C++): {len(nm) - bad} of {len(nm)} base quantities agree, {bad} off")
    print(f"{a.date}: done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
