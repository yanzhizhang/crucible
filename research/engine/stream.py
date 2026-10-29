r"""One event stream per stock: snapshots and trades received by PM's stop, merged by local arrival.

Reads the catalog through ``dcp_ema.load`` (the same columns and the same ``stop`` cut the research scripts use) and
yields, per stock, ``("snap", t, s, lsod, price, b1, a1, hi, lo, op, pc, amount)`` and ``("trade", t, s, price, vol,
side)`` tuples in arrival order; ``t`` is the trading clock (09:30 = 0, lunch removed), ``s`` the exchange second
of day, ``lsod`` the local second of day.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

import dcp_ema as E  # tick loading


def _lsod(ls: np.ndarray) -> np.ndarray:
    """Local second of day of epoch-ns receive stamps."""
    return (ls / 1e9 + 28800.0) % 86400.0


def streams(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> Iterator[tuple[int, int, list[tuple]]]:
    """(sid, market, events) for every stock with ticks."""
    up = E.load(con, date, sids, stop)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")} if not up["fast"].is_empty() else {}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")} if not up["slow"].is_empty() else {}
    for sid in sids:
        gf, gs = fast.get(sid), slow.get(sid)
        if gf is None and gs is None:
            continue
        ev: list[tuple] = []
        if gf is not None:
            f = {c: gf[c].to_numpy().astype(float) for c in ("t_local", "s", "price", "b1", "a1", "hi", "lo", "op", "pc", "ca")}
            ls = gf["ls"].to_numpy().astype(float)
            lsod = _lsod(ls)
            ev += [("snap", float(f["t_local"][i]), float(f["s"][i]), float(lsod[i]), float(f["price"][i]), float(f["b1"][i]), float(f["a1"][i]),
                    float(f["hi"][i]), float(f["lo"][i]), float(f["op"][i]), float(f["pc"][i]), float(f["ca"][i]), float(ls[i]), 0)
                   for i in range(gf.height)]
        if gs is not None:
            g = {c: gs[c].to_numpy().astype(float) for c in ("t_local", "s", "price", "vol", "side")}
            ls = gs["ls"].to_numpy().astype(float)
            ev += [("trade", float(g["t_local"][i]), float(g["s"][i]), float(g["price"][i]), float(g["vol"][i]), float(g["side"][i]), float(ls[i]), 1)
                   for i in range(gs.height)]
        ev.sort(key=lambda e: (e[-2], e[-1]))  # receive stamp, snapshots before trades at the same stamp
        mkt = int((gf if gf is not None else gs)["mkt"][0])
        yield sid, mkt, ev


def universe_from_state(path: Path) -> tuple[list[int], float]:
    """Stocks and PM's stop (local second of day) from a day-end state file."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))
    import dcp_build as B

    pm = B.load_state(path)
    tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
    return sorted(pm["sid"].unique().to_list()), (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
