r"""PM's own feed (.mp, see mp_read.py) as the same per-stock frames dcp_ema.load returns, so the probes run on it unchanged.

Layout (r224, 20261119): snapshot = sid, recv epoch, exch epoch, cum volume, cum amount, ask px 1-10, bid px 1-10,
ask vol 1-10, bid vol 1-10, f45 (per-stock constant), last, f47 (count), f48 (per-stock constant), high, low;
SH trade = sid, recv, exch, buy order no, sell order no, price, qty, channel, seq (no side: the later order is the aggressor);
SZ trade = sid, recv, exch, bid seq, offer seq, price, qty, channel, seq (price 0 = cancel, dropped).
Clocks as the runtime's TermCFH: tod = epoch mod 86400 (UTC); t = tod - 5400 before 11:30, 7200 at lunch, tod - 10800 after.
``cache`` decodes a day once into parquet (pure-Python msgpack is slow); ``load`` reads the cache. Read-only on the .mp files.

    python research/pm_factors/mp_load.py --day /work/crucible_data/pm_mp/20261119 --out /work/crucible_data/pm_mp_cache/20261119
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_read  # noqa: E402

SNAP = ["sid", "recv", "exch", "cv", "ca"] + [f"a{i}" for i in range(1, 11)] + [f"b{i}" for i in range(1, 11)] \
    + [f"av{i}" for i in range(1, 11)] + [f"bv{i}" for i in range(1, 11)] + ["f45", "last", "f47", "f48", "high", "low"]
TRADE = ["sid", "recv", "exch", "bno", "sno", "price", "vol", "chan", "seq"]


def cache(day: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for mkt in ("SH", "SZ"):
        for kind, cols in (("snap", SNAP), ("trade", TRADE)):
            src, dst = day / f"{mkt}_HS300_{kind}.mp", out / f"{mkt}_{kind}.parquet"
            if dst.exists():
                continue
            t0 = time.time()
            a = mp_read.frame(src)
            pl.DataFrame({c: a[:, i] for i, c in enumerate(cols[:a.shape[1]])}).write_parquet(dst)
            print(f"   {src.name}: {a.shape[0]:,} records x {a.shape[1]} -> {dst} ({time.time() - t0:.0f} s)")


def clock(epoch: np.ndarray) -> np.ndarray:
    tod = np.mod(epoch, 86400.0)
    return np.where(tod < 12600, tod - 5400, np.where(tod <= 18000, 7200.0, tod - 10800))


def load(cache_dir: Path, sids: list[int], stop: float) -> dict[str, pl.DataFrame]:
    """{'fast': snapshots, 'slow': trades} with dcp_ema.load's column names, received by the stop (local seconds of day)."""
    fast, slow = [], []
    for mkt in ("SH", "SZ"):
        q = pl.read_parquet(cache_dir / f"{mkt}_snap.parquet").filter(pl.col("sid").cast(pl.Int64).is_in(sids))  # the cache keeps sid as float64
        q = q.filter(((pl.col("recv") + 28800) % 86400) <= stop)
        fast.append(q.select(pl.col("sid").cast(pl.Int64), pl.lit(1 if mkt == "SH" else 2).alias("mkt"),
                             pl.col("recv").map_batches(lambda s: pl.Series(clock(s.to_numpy()))).alias("t_local"),
                             pl.col("exch").map_batches(lambda s: pl.Series(clock(s.to_numpy()))).alias("t_exch"),
                             (pl.col("recv") * 1e9).cast(pl.Int64).alias("ls"), ((pl.col("exch") + 28800) % 86400).alias("s"),
                             pl.col("last").alias("price"), pl.col("b1"), pl.col("a1"), pl.col("bv1"), pl.col("av1"),
                             pl.col("cv"), pl.col("ca")))
        t = pl.read_parquet(cache_dir / f"{mkt}_trade.parquet").filter(pl.col("sid").cast(pl.Int64).is_in(sids) & (pl.col("price") > 0))
        t = t.filter(((pl.col("recv") + 28800) % 86400) <= stop)
        if mkt == "SZ":
            t = t.filter((pl.col("bno") > 0) & (pl.col("sno") > 0))
        slow.append(t.select(pl.col("sid").cast(pl.Int64), pl.lit(1 if mkt == "SH" else 2).alias("mkt"),
                             pl.col("recv").map_batches(lambda s: pl.Series(clock(s.to_numpy()))).alias("t_local"),
                             pl.col("exch").map_batches(lambda s: pl.Series(clock(s.to_numpy()))).alias("t_exch"),
                             (pl.col("recv") * 1e9).cast(pl.Int64).alias("ls"), ((pl.col("exch") + 28800) % 86400).alias("s"),
                             pl.col("price"), pl.col("vol"), (pl.col("price") * pl.col("vol")).alias("amt"),
                             pl.when(pl.col("sno") > pl.col("bno")).then(1.0).otherwise(-1.0).alias("side"),  # r225: fields 3 / 4 are (sell, buy)
                             pl.col("sno").alias("bsq"), pl.col("bno").alias("ssq"), pl.col("seq")))
    f = pl.concat(fast).sort(["sid", "ls"], maintain_order=True)
    s = pl.concat(slow).sort(["sid", "ls", "seq"], maintain_order=True)
    return {"fast": f, "slow": s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--day", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cache(a.day, a.out)
    print("[done]")


if __name__ == "__main__":
    main()
