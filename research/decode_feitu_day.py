"""Decode one day of feitu v3 minute dumps into raw Parquet, in parallel, with resource stats.

Input layout (one directory per stream, one ``.zst`` per minute)::

    <root>/v3_order/v3_order_2026_06_15_10_30.zst
    <root>/v3_transaction/...  <root>/v3_quotation/...  <root>/v3_index/...

Output (raw vendor values, no unit conversion -- normalisation is a separate, auditable step)::

    <out>/kind=<order|transaction|quotation|index>/date=YYYYMMDD/part-HHMM.parquet
    <out>/kind=<...>/date=YYYYMMDD/_manifest.parquet   one row per source file

The decoder is the C++ kernel (``crucible_kernels.feitu``, GIL released), fanned out over a
thread pool; Parquet writing (pyarrow, also GIL-free) runs in the same workers.

Usage (WSL)::

    python research/decode_feitu_day.py --root /mnt/c/Users/zzz/Documents/20260615 \
        --date 20260615 --out /work/crucible_data/feitu_raw
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resmon import ResourceMonitor, append_history
from crucible_kernels import feitu

KINDS = ("order", "transaction", "quotation", "index")
#: Decoded columns + the Parquet write buffers, as a multiple of the compressed file size.
#: Measured on 20260615: ~3x raw columns, ~1x writer; 5x leaves headroom.
MEM_PER_COMPRESSED_BYTE = 5


class MemoryBudget:
    """Weighted semaphore: a task holds ``cost`` bytes of the budget while it runs.

    Keeps the open-of-day minute files (hundreds of MB compressed each) from all decoding at
    once and pushing the box into swap, while small files still fan out across every core.
    A single task larger than the whole budget is admitted alone rather than deadlocking.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.used = 0
        self._cv = threading.Condition()

    def acquire(self, cost: int) -> int:
        """Block until ``cost`` bytes fit; return the amount actually held."""
        cost = min(cost, self.total)
        with self._cv:
            self._cv.wait_for(lambda: self.used + cost <= self.total)
            self.used += cost
        return cost

    def release(self, cost: int) -> None:
        """Return ``cost`` bytes to the budget."""
        with self._cv:
            self.used -= cost
            self._cv.notify_all()


_MINUTE = re.compile(r"_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.zst$")
_LEVEL_COLS = ("bid_px", "bid_vol", "bid_no", "ask_px", "ask_vol", "ask_no")


def _to_table(cols: dict[str, Any]) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    for name, col in cols.items():
        if name in _LEVEL_COLS:
            flat = pa.array(np.ascontiguousarray(col).reshape(-1))
            arrays[name] = pa.FixedSizeListArray.from_arrays(flat, feitu.levels)
        elif name == "symbol_str":
            arrays[name] = pa.array(col, type=pa.string())
        else:
            arrays[name] = pa.array(col)
    return pa.table(arrays)


def _decode_one(kind: str, src: Path, dst_dir: Path, overwrite: bool) -> dict[str, Any]:
    m = _MINUTE.search(src.name)
    if m is None:
        raise ValueError(f"unexpected dump name {src.name!r}")
    hhmm = m.group(4) + m.group(5)
    dst = dst_dir / f"part-{hhmm}.parquet"
    row: dict[str, Any] = {
        "file": src.name,
        "minute": hhmm,
        "bytes": src.stat().st_size,
        "rows": 0,
        "n_truncated_books": 0,
        "exchange_ts_min": None,
        "exchange_ts_max": None,
        "arrival_ts_min": None,
        "arrival_ts_max": None,
    }
    if dst.exists() and not overwrite:
        meta = pq.read_metadata(dst)
        row["rows"] = meta.num_rows
        row["skipped"] = True
        return row
    if kind == "quotation":
        cols, truncated = feitu.decode_quotation(str(src))
        row["n_truncated_books"] = int(truncated)
    else:
        cols = getattr(feitu, f"decode_{kind}")(str(src))
    n = len(cols["time"])
    row["rows"] = n
    if n:
        row["exchange_ts_min"] = int(cols["time"].min())
        row["exchange_ts_max"] = int(cols["time"].max())
        row["arrival_ts_min"] = int(cols["spider_ts"].min())
        row["arrival_ts_max"] = int(cols["spider_ts"].max())
    tmp = dst.with_suffix(".parquet.tmp")
    pq.write_table(
        _to_table(cols), tmp, compression="zstd", compression_level=3, row_group_size=1 << 20
    )
    tmp.replace(dst)
    row["skipped"] = False
    return row


def decode_day(
    root: Path,
    date: str,
    out: Path,
    *,
    kinds: tuple[str, ...],
    threads: int,
    overwrite: bool,
    mem_budget: int,
) -> int:
    """Decode every minute file of ``kinds`` under ``root``; return total rows."""
    total = 0
    budget = MemoryBudget(mem_budget)

    def run(kind: str, f: Path, d: Path) -> dict[str, Any]:
        held = budget.acquire(f.stat().st_size * MEM_PER_COMPRESSED_BYTE)
        try:
            return _decode_one(kind, f, d, overwrite)
        finally:
            budget.release(held)

    for kind in kinds:
        src_dir = root / f"v3_{kind}"
        # Largest first: the open-of-day files dominate, so start them early to cut the tail.
        files = sorted(src_dir.glob(f"v3_{kind}_*.zst"), key=lambda f: -f.stat().st_size)
        if not files:
            print(f"[{kind}] no files under {src_dir} -- stream absent for {date}")
            continue
        dst_dir = out / f"kind={kind}" / f"date={date}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        with ResourceMonitor(
            f"W1.decode.{kind}", params={"date": date, "threads": threads, "files": len(files)}
        ) as mon:
            with ThreadPoolExecutor(threads) as ex:
                rows = list(ex.map(lambda f, k=kind, d=dst_dir: run(k, f, d), files))
            rows.sort(key=lambda r: r["minute"])
            mon.rows = sum(r["rows"] for r in rows)
        assert mon.stats is not None
        print(mon.stats.line())
        append_history(mon.stats, out.parent / "bench" / "history.parquet")
        pa_rows = pa.Table.from_pylist(rows)
        pq.write_table(pa_rows, dst_dir / "_manifest.parquet")
        total += mon.rows
    return total


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="directory holding v3_<kind>/")
    ap.add_argument("--date", required=True, help="trading day YYYYMMDD")
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--kinds", default=",".join(KINDS))
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--mem-frac",
        type=float,
        default=0.6,
        help="fraction of currently available memory the decode may hold",
    )
    a = ap.parse_args()
    mem_budget = int(psutil.virtual_memory().available * a.mem_frac)
    kinds = tuple(k for k in a.kinds.split(",") if k)
    bad = set(kinds) - set(KINDS)
    if bad:
        raise SystemExit(f"unknown kinds {sorted(bad)}; known {KINDS}")
    print(f"memory budget {mem_budget / 2**30:.1f} GiB, threads {a.threads}")
    n = decode_day(
        a.root,
        a.date,
        a.out,
        kinds=kinds,
        threads=a.threads,
        overwrite=a.overwrite,
        mem_budget=mem_budget,
    )
    print(f"done {a.date}: {n:,} rows")


if __name__ == "__main__":
    main()
