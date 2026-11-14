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


def load_orders(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> pl.DataFrame:
    """Order events (stamps only) received by the stop: the all-event clock ticks on them."""
    import dcp_build as B

    S, L = B.SOD, B.LSOD
    con.register("want_o", pl.DataFrame({"sid": sids}))
    out = con.sql(f"""SELECT symbol_id AS sid, {B.tc(L)} AS t_local, spider_ts AS ls, {S} AS s, price::DOUBLE AS price, volume::DOUBLE AS vol,
        CASE dir WHEN 1063 THEN 1.0 WHEN 1550 THEN -1.0 ELSE 0.0 END AS side
        FROM raw_order o JOIN want_o w ON w.sid = o.symbol_id WHERE o.date = {date} AND {L} <= {stop}
        ORDER BY sid, spider_ts{", seq_id" if "seq_id" in E.columns(con, "raw_order") else ""}, time""").pl()
    con.unregister("want_o")
    return out


def load_cancels(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> pl.DataFrame:
    """Cancel records of the transaction stream (SZSE trade_type 2) received by the stop: the bothc / all clocks tick on them."""
    import dcp_build as B

    S, L = B.SOD, B.LSOD
    con.register("want_c", pl.DataFrame({"sid": sids}))
    out = con.sql(f"""SELECT symbol_id AS sid, {B.tc(L)} AS t_local, spider_ts AS ls, {S} AS s
        FROM raw_transaction r JOIN want_c w ON w.sid = r.symbol_id WHERE r.date = {date} AND r.trade_type = 2 AND {L} <= {stop}
        ORDER BY sid, spider_ts{", seq_id" if "seq_id" in E.columns(con, "raw_transaction") else ""}, time""").pl()
    con.unregister("want_c")
    return out


def streams(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> Iterator[tuple[int, int, list[tuple]]]:
    """(sid, market, events) for every stock with ticks."""
    up = E.load(con, date, sids, stop)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")} if not up["fast"].is_empty() else {}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")} if not up["slow"].is_empty() else {}
    orders = {int(g["sid"][0]): g for g in load_orders(con, date, sids, stop).partition_by("sid")}
    cancels = {int(g["sid"][0]): g for g in load_cancels(con, date, sids, stop).partition_by("sid")}
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
        gc = cancels.get(sid)
        if gc is not None:
            ls = gc["ls"].to_numpy().astype(float)
            tl, s_ = gc["t_local"].to_numpy().astype(float), gc["s"].to_numpy().astype(float)
            ev += [("cancel", float(tl[i]), float(s_[i]), float(ls[i]), 3) for i in range(gc.height)]
        go = orders.get(sid)
        if go is not None:
            ls = go["ls"].to_numpy().astype(float)
            tl, s_ = go["t_local"].to_numpy().astype(float), go["s"].to_numpy().astype(float)
            px, vo, sd = (go[c].to_numpy().astype(float) for c in ("price", "vol", "side"))
            ev += [("order", float(tl[i]), float(s_[i]), float(px[i]), float(vo[i]), float(sd[i]), float(ls[i]), 2) for i in range(go.height)]
        ev.sort(key=lambda e: (e[-2], e[-1]))  # receive stamp; snapshots, then trades, then orders at the same stamp
        mkt = int((gf if gf is not None else gs)["mkt"][0])
        yield sid, mkt, ev


def universe_from_state(path: Path) -> tuple[list[int], float]:
    """Stocks and PM's stop (local second of day) from a day-end state file."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))
    import dcp_build as B

    pm = B.load_state(path)
    tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
    return sorted(pm["sid"].unique().to_list()), (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
