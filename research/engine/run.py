r"""Run the engine over one day: every stock's event stream through the rules, the day-end state written like PM's.

    python research/engine/run.py --date 20261009 --universe pm:/data/prod/h/HS300/p1_intrading_dcp/20261009.jz
    python research/engine/run.py --date 20261009 --universe thr:/data/prod/hstats/HS300/ZZCR/20261008/samplerR.parquet --stop 54150

Output: ``<out>/engine_<date>.json.gz`` -- ``{sid: {key: value}}``, the layout of PM's ``.jz`` (NaN written as null).
``--universe pm:<jz>`` also takes the stop from the file (its latest ``zzba.p.t``), as ``dcp_build --stop auto``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rules import EMAS, on_snap, on_trade
from state import Stock
from stream import streams, universe_from_state

DEFAULT_STOP = 54150.005  # 15:02:30 local
CATALOG = Path("/work/crucible_data/catalog.duckdb")


def run_day(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> dict[int, dict[str, float]]:
    """Every stock's day-end state."""
    out = {}
    for sid, _mkt, events in streams(con, date, sids, stop):
        st = Stock(EMAS)
        for e in events:
            if e[0] == "snap":
                on_snap(st, *e[1:12])
            else:
                on_trade(st, *e[1:6])
        out[sid] = st.dump()
    return out


def run_chunk(args: tuple) -> dict[int, dict[str, float]]:
    """Run run_day for a chunk of stocks on its own catalog connection (a worker process)."""
    catalog, date, sids, stop = args
    con = duckdb.connect(str(catalog), read_only=True)
    con.execute("SET threads=2")
    try:
        return run_day(con, date, sids, stop)
    finally:
        con.close()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--universe", required=True, help="pm:<state .jz> | thr:<samplerR parquet> | all")
    ap.add_argument("--stop", type=float, help="local second of day PM stopped at (default: from the state file, else 15:02:30)")
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--workers", type=int, default=1, help="stock chunks run in parallel")
    ap.add_argument("--out", type=Path, default=Path("data/engine"))
    a = ap.parse_args()
    t0 = time.perf_counter()
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
    if a.workers > 1:
        import multiprocessing as mp
        chunks = [sids[i::a.workers * 2] for i in range(a.workers * 2)]
        state = {}
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for part in pool.imap_unordered(run_chunk, [(a.catalog, int(a.date), c, stop) for c in chunks if c]):
                state.update(part)
    else:
        state = run_day(con, int(a.date), sids, stop)
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"engine_{a.date}.json.gz"
    with gzip.open(dst, "wt") as f:
        json.dump({str(s): {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in kv.items()} for s, kv in state.items()}, f)
    n_keys = len(next(iter(state.values()))) if state else 0
    print(f"{a.date}: {len(state)} stocks x {n_keys} keys, stop {stop:.3f} -> {dst} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
