r"""Engine events from PM's own feed: the decoded .mp cache (research/pm_factors/mp_load.py's parquet, one file per market and
kind) as export.py's EVENT_DTYPE, so the C++ engine runs on exactly the stream the runtime saw.

Snapshot fields b1 / a1 / high / low / cum amount come from the feed; open and previous close are exchange constants taken
from the catalog (the .mp snapshot has no labelled open / pre-close). Trades: price > 0, SZ cancels (an order number 0) dropped,
side from the later order number (the aggressor): SH fields 3 / 4 are (sell, buy) (r225b), SZ fields 3 / 4 are (bid, offer). Orders and cancel records are not in the cache, so kinds 2 / 3 are absent. Clocks as the runtime's TermCFH
(recv / exch epochs, UTC tod; trading clock tod - 5400 before 11:30, 7200 at lunch, tod - 10800 after). Read-only.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from export import EVENT_DTYPE


def _tc(epoch: np.ndarray) -> np.ndarray:
    tod = np.mod(epoch, 86400.0)
    return np.where(tod < 12600, tod - 5400, np.where(tod <= 18000, 7200.0, tod - 10800))


def _sod(epoch: np.ndarray) -> np.ndarray:
    return np.mod(epoch + 28800.0, 86400.0)


def constants(con: duckdb.DuckDBPyConnection, date: int, sids: list[int]) -> dict[int, tuple[float, float]]:
    """sid -> (open, pre_close) from the catalog's snapshots (first positive value of the day)."""
    con.register("want", pl.DataFrame({"sid": sids}))
    rows = con.sql(f"""SELECT symbol_id, max(pre_close) FILTER (WHERE pre_close > 0), arg_min(open, spider_ts) FILTER (WHERE open > 0)
                       FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} GROUP BY 1""").fetchall()
    con.unregister("want")
    return {int(s): (float(op) if op else np.nan, float(pc) if pc else np.nan) for s, pc, op in rows}


def load_events_mp(cache: Path, sids: list[int], stop: float, consts: dict[int, tuple[float, float]], trades: bool = True) -> np.ndarray:
    parts = []
    want = set(sids)
    for mkt, code in (("SH", 1), ("SZ", 2)):
        q = pl.read_parquet(cache / f"{mkt}_snap.parquet").with_columns(pl.col("sid").cast(pl.Int64))
        q = q.filter(pl.col("sid").is_in(list(want)))
        recv, exch = q["recv"].to_numpy(), q["exch"].to_numpy()
        keep = _sod(recv) <= stop
        sid = q["sid"].to_numpy()[keep]
        a = np.empty(int(keep.sum()), dtype=EVENT_DTYPE)
        a["sid"], a["mkt"], a["kind"] = sid, code, 0
        a["t"], a["s"], a["lsod"] = _tc(recv[keep]), _sod(exch[keep]), _sod(recv[keep])
        a["price"] = q["last"].to_numpy()[keep]
        op = np.array([consts.get(int(s), (np.nan, np.nan))[0] for s in sid]); pc = np.array([consts.get(int(s), (np.nan, np.nan))[1] for s in sid])
        for k, v in enumerate((q["b1"].to_numpy()[keep], q["a1"].to_numpy()[keep], q["high"].to_numpy()[keep], q["low"].to_numpy()[keep], op, pc,
                               q["ca"].to_numpy()[keep])):
            a["f"][:, k] = v
        parts.append((a, recv[keep], np.zeros(a.size)))
        if not trades:
            continue
        t = pl.read_parquet(cache / f"{mkt}_trade.parquet").with_columns(pl.col("sid").cast(pl.Int64))
        t = t.filter(pl.col("sid").is_in(list(want)) & (pl.col("price") > 0))
        if mkt == "SZ":
            t = t.filter((pl.col("bno") > 0) & (pl.col("sno") > 0))
        recv, exch = t["recv"].to_numpy(), t["exch"].to_numpy()
        keep = _sod(recv) <= stop
        b = np.empty(int(keep.sum()), dtype=EVENT_DTYPE)
        b["sid"], b["mkt"], b["kind"] = t["sid"].to_numpy()[keep], code, 1
        b["t"], b["s"], b["lsod"] = _tc(recv[keep]), _sod(exch[keep]), _sod(recv[keep])
        b["price"] = t["price"].to_numpy()[keep]
        b["f"][:] = 0.0
        b["f"][:, 0] = t["vol"].to_numpy()[keep]
        # the later order is the aggressor; fields 3 / 4 read as (sell, buy) on both markets (r225b; flipping SZ: r244 zzgh 0 exact)
        b["f"][:, 1] = np.where(t["sno"].to_numpy()[keep] > t["bno"].to_numpy()[keep], 1.0, -1.0)
        parts.append((b, recv[keep], t["seq"].to_numpy()[keep].astype(np.float64)))
    ev = np.concatenate([p[0] for p in parts])
    ls = np.concatenate([p[1] for p in parts])
    sq = np.concatenate([p[2] for p in parts])
    order = np.lexsort((ev["s"], sq, ev["kind"], ls, ev["sid"]))  # as export.py: sid, arrival, kind, sequence, exchange time
    return ev[order]


def hybrid(snaps: np.ndarray, other: np.ndarray) -> np.ndarray:
    """PM-feed snapshots + the catalog's trades / orders / cancels (kinds 1-3), ordered by sid, local second of day, kind."""
    ev = np.concatenate([snaps, other[other["kind"] != 0]])
    return ev[np.lexsort((ev["kind"], ev["lsod"], ev["sid"]))]


INDEX_KEYS = {"zzfb.c3": "SZ_HS300_index", "zzfb.c5": "SZ_ZZ500_index"}  # r246: c3 = CSI 300, c5 = CSI 500 (SZ copy of the stream)


def index_grids(raw_day: Path, n: int = 711, session: float = 14220.0) -> dict[str, np.ndarray]:
    """zzfb.c3 / c5: the index's cumulative amount, last value received strictly before each slot's end (local trading
    clock), 0 before the first record. r246 (1119): 1.0000 exact on all 711 slots for both."""
    import mp_read  # noqa: PLC0415  (research/pm_factors)

    step, out = session / n, {}
    for key, name in INDEX_KEYS.items():
        p = raw_day / f"{name}.mp"
        if not p.exists():
            continue
        a = mp_read.frame(p)
        t = _tc(a[:, 1]); o = np.argsort(t, kind="stable"); t, amt = t[o], a[o, 6]
        idx = np.searchsorted(t, (np.arange(n) + 1) * step, side="left") - 1
        out[key] = np.where(idx >= 0, amt[np.clip(idx, 0, None)], 0.0)
    return out


def inject_index(state_json: Path, grids: dict[str, np.ndarray]) -> int:
    """Write the market-wide grids into every stock of the engine's day-end state.json (same values for all stocks)."""
    import json  # noqa: PLC0415

    d = json.loads(state_json.read_text())
    st = d["stocks"] if isinstance(d, dict) and isinstance(d.get("stocks"), dict) else d
    n = 0
    for v in st.values():
        if isinstance(v, dict):
            for key, g in grids.items():
                for k, x in enumerate(g):
                    v[f"{key}.{k}"] = float(x)
            n += 1
    state_json.write_text(json.dumps(d))
    return n
