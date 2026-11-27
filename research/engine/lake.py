r"""The engine's events straight from tickaudit's lake (``/work/tickaudit_lake``), no catalog and no feitu translation.

Ticks come from the MERGED layer (all sources majority-voted, one row per (channel, seq), gaps are real holes); the
10-level snapshots are not merged yet, so they come from one source's ``tick/`` table (``--snap-source``)::

    merged/sse_tick/day=D/data.parquet      SH UA5803: T -> kind 1 trade, A / D -> kind 2 order (S status rows dropped)
    merged/szse_order/day=D/data.parquet    SZ 300192 -> kind 2 order
    merged/szse_trade/day=D/data.parquet    SZ 300191: F -> kind 1 trade, 4 -> kind 3 cancel
    tick/sse_snap|szse_snap/day=D/source=S/data.parquet   -> kind 0 snapshot (bid1, ask1, high, low, open, pre_close, amount)

The exchange's fixed-point int64 is divided to doubles only here, at the engine boundary (SH price / qty x1e3, value
x1e5; SZ tick price x1e4, qty x1e2; SZ snapshot preClose x1e4, other prices x1e6, value x1e4). The receive stamp that
orders events is ``local_<--local-source>`` (the feed being reproduced), else ``first_local_ns`` when that source missed
the row; ``--local-source first`` uses first_local_ns throughout. Only ``*_stock`` streams are read.

``events.bin`` is the same layout as ``export.load_events`` writes, so the C++ engine is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

import dcp_build as B  # trading clock
from export import EVENT_DTYPE

LAKE = Path("/work/tickaudit_lake")
SSE, SZSE = 3553, 3554
ABSENT = -(2**63)


def sod(ns: str) -> str:
    """Local (CST) second of day of a UTC-ns column."""
    return f"(({ns} / 1e9 + 28800) % 86400)"


def f(col: str, scale: float) -> str:
    """A fixed-point column as a double, INT64_MIN (absent) as 0."""
    return f"(CASE WHEN {col} = {ABSENT} THEN 0 ELSE {col} END / {scale!r})::DOUBLE"


def merged(table: str, date: int, lake: Path = LAKE) -> Path:
    return lake / "merged" / table / f"day={date}" / "data.parquet"


def snap(table: str, date: int, src: str, lake: Path = LAKE) -> Path:
    return lake / "tick" / table / f"day={date}" / f"source={src}" / "data.parquet"


def recv(path: Path, local_src: str) -> str:
    """The receive-stamp expression of a merged file: the chosen source's column, else the earliest of all."""
    if local_src == "first":
        return "first_local_ns"
    cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{path}')").pl()["column_name"].to_list()
    if f"local_{local_src}" not in cols:
        raise SystemExit(f"{path} has no local_{local_src} (sources: {[c[6:] for c in cols if c.startswith('local_')]})")
    return f"coalesce(local_{local_src}, first_local_ns)"


def check(date: int, snap_src: str, lake: Path = LAKE) -> list[Path]:
    """Every input file of the day, or stop naming the missing ones."""
    need = [merged(t, date, lake) for t in ("sse_tick", "szse_order", "szse_trade")] + \
           [snap(t, date, snap_src, lake) for t in ("sse_snap", "szse_snap")]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise SystemExit(f"tickaudit lake has no {miss}: run tickaudit's batch / merge for {date} first")
    return need


def universe(date: int, snap_src: str, lake: Path = LAKE) -> list[int]:
    """Every stock with a snapshot that day (``--universe all``)."""
    q = " UNION ".join(f"SELECT DISTINCT securityId::INT AS sid FROM read_parquet('{snap(t, date, snap_src, lake)}') WHERE stream LIKE '%_stock'"
                       for t in ("sse_snap", "szse_snap"))
    return sorted(int(r[0]) for r in duckdb.sql(q).fetchall())


def load_events(date: int, sids: list[int], stop: float, snap_src: str = "datayes", local_src: str = "datayes",
                lake: Path = LAKE) -> np.ndarray:
    """Every snapshot, trade, order and cancel received by the stop, in the engine's order (as export.load_events)."""
    sh, so, st, qs, qz = check(date, snap_src, lake)
    con = duckdb.connect()
    con.register("want", pl.DataFrame({"sid": sids}))
    ls_sh, ls_so, ls_st = recv(sh, local_src), recv(so, local_src), recv(st, local_src)
    side_sh = "CASE flag WHEN 'B' THEN 1.0 WHEN 'S' THEN -1.0 ELSE 0.0 END"  # orders keep their own flag
    # a trade with flag N (the auctions) takes the later order's side, as the runtime does with SH's order numbers
    side_sh_t = "CASE flag WHEN 'B' THEN 1.0 WHEN 'S' THEN -1.0 ELSE (CASE WHEN buyOrderNo > sellOrderNo THEN 1.0 ELSE -1.0 END) END"
    side_sz = "CASE side WHEN '1' THEN 1.0 WHEN '2' THEN -1.0 ELSE 0.0 END"
    # an SZSE trade's aggressor is the leg with the larger ApplSeqNum (decode_tdf_day's rule)
    side_tr = "CASE WHEN bidApplSeqNum > offerApplSeqNum THEN 1.0 ELSE -1.0 END"

    def part(src: Path, mkt: int, kind: int, ls: str, tns: str, price: str, fs: list[str], sq: str, where: str) -> str:
        L, S = sod(ls), sod(tns)
        cols = ", ".join(f"{x}::DOUBLE AS f{i}" for i, x in enumerate(fs + ["0.0"] * (7 - len(fs))))
        return (f"SELECT securityId::INT AS sid, {mkt}::SMALLINT AS mkt, {kind}::SMALLINT AS kind, ({B.tc(L)})::DOUBLE AS t, "
                f"{S}::DOUBLE AS s, {L}::DOUBLE AS lsod, {ls} AS ls, {price} AS price, {cols}, {sq} AS sq "
                f"FROM read_parquet('{src}') x JOIN want w ON w.sid = x.securityId "
                f"WHERE stream LIKE '%_stock' AND {where} AND {L} <= {stop!r}")

    def quote(src: Path, mkt: int) -> str:
        ps, pre, val = (1e3, 1e3, 1e5) if mkt == SSE else (1e6, 1e4, 1e4)
        loc = "CASE WHEN localNs > 0 THEN localNs ELSE timeNs END"
        return part(src, mkt, 0, loc, "timeNs", f("lastPx", ps),
                    [f("bid1_price", ps), f("ask1_price", ps), f("highPx", ps), f("lowPx", ps), f("openPx", ps), f("preClosePx", pre),
                     f("totalValue", val)], "NULL::DOUBLE", "TRUE")

    sql = " UNION ALL ".join([
        quote(qs, SSE), quote(qz, SZSE),
        part(sh, SSE, 1, ls_sh, "tickTimeNs", f("price", 1e3), [f("qty", 1e3), side_sh_t], "bizIndex::DOUBLE", "type = 'T'"),
        part(sh, SSE, 2, ls_sh, "tickTimeNs", f("price", 1e3), [f("qty", 1e3), side_sh], "bizIndex::DOUBLE", "type IN ('A', 'D')"),
        part(st, SZSE, 1, ls_st, "transactTimeNs", f("lastPx", 1e4), [f("lastQty", 1e2), side_tr], "applSeqNum::DOUBLE", "execType = 'F'"),
        part(so, SZSE, 2, ls_so, "transactTimeNs", f("price", 1e4), [f("orderQty", 1e2), side_sz], "applSeqNum::DOUBLE", "TRUE"),
        part(st, SZSE, 3, ls_st, "transactTimeNs", "0.0::DOUBLE", [f("lastQty", 1e2)], "applSeqNum::DOUBLE", "execType = '4'"),
    ])
    df = con.sql(f"SELECT sid, mkt, kind, t, s, lsod, price, f0, f1, f2, f3, f4, f5, f6 FROM ({sql}) ORDER BY sid, ls, kind, sq, s").pl()
    con.close()
    out = np.empty(df.height, dtype=EVENT_DTYPE)
    out["sid"], out["mkt"], out["kind"] = df["sid"].to_numpy(), df["mkt"].to_numpy(), df["kind"].to_numpy()
    for c in ("t", "s", "lsod", "price"):
        out[c] = df[c].to_numpy().astype(np.float64)
    for k in range(7):
        out["f"][:, k] = df[f"f{k}"].to_numpy().astype(np.float64)
    return out


def missed(date: int, local_src: str, lake: Path = LAKE) -> str:
    """How many merged rows the chosen receive source did not have (they fall back to first_local_ns)."""
    if local_src == "first":
        return "receive stamp: first_local_ns"
    parts = []
    for t in ("sse_tick", "szse_order", "szse_trade"):
        p = merged(t, date, lake)
        n, m, tie = duckdb.sql(f"SELECT count(*), count(*) FILTER (WHERE local_{local_src} IS NULL), count(*) FILTER (WHERE tie_fields <> 0) "
                               f"FROM read_parquet('{p}') WHERE stream LIKE '%_stock'").fetchone()
        parts.append(f"{t} {n:,} rows, {m:,} without local_{local_src}, {tie:,} with tied fields")
    return "; ".join(parts)


def day_hl(date: str, sids: list[int], lake: Path = LAKE) -> pl.DataFrame | None:
    """The day's traded high / low per stock from the merged ticks (zzeg.c channel), None when the day is not merged."""
    sh, st = merged("sse_tick", int(date), lake), merged("szse_trade", int(date), lake)
    if not (sh.exists() and st.exists()):
        return None
    con = duckdb.connect()
    con.register("want", pl.DataFrame({"sid": sids}))
    df = con.sql(f"""SELECT sid AS symbol_id, max(p) AS h, min(p) AS l FROM (
        SELECT securityId::INT AS sid, price / 1e3 AS p FROM read_parquet('{sh}') WHERE stream LIKE '%_stock' AND type = 'T' AND price > 0
        UNION ALL SELECT securityId::INT, lastPx / 1e4 FROM read_parquet('{st}') WHERE stream LIKE '%_stock' AND execType = 'F' AND lastPx > 0
        ) x JOIN want w USING (sid) GROUP BY 1""").pl()
    con.close()
    return df


def prev_channel(date: str, sids: list[int], lake: Path = LAKE) -> dict[int, tuple[float, float]]:
    """sid -> (max high, min low) over the previous 3 merged days (run_cpp.prev_channel's lake twin)."""
    days = sorted(p.name[4:] for p in (lake / "merged" / "sse_tick").glob("day=*") if p.name[4:] < date)
    got = []
    for d in reversed(days):
        hl = day_hl(d, sids, lake)
        if hl is not None and hl.height > 0.5 * len(sids):
            got.append(hl)
        if len(got) == 3:
            break
    if not got:
        return {}
    g = pl.concat(got).group_by("symbol_id").agg(pl.col("h").max(), pl.col("l").min())
    return {int(r[0]): (float(r[1]), float(r[2])) for r in g.iter_rows() if r[1] is not None and r[2] is not None}
