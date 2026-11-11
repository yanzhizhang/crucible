r"""Write the C++ engine's inputs: ``events.bin`` from the catalog, ``rules.tsv`` from ``rules.EMAS``, ``prev.bin`` from a state file.

The event layout mirrors ``cpp/event.hpp`` (96 bytes, every number a double, sorted by sid, local receive stamp, kind);
``rules.tsv`` is the EMAS table one rule per line (name, clock, half-life, value kind, carry seconds or -1); ``prev.bin``
holds yesterday's ``<base>.sum / .wgt`` for every seeded rule (``|c<N>`` / ``|w<N>`` / ``|wt<N>``) as (sid, rule index, sum,
wgt), undecayed -- the engine applies the seed kind and ``exp(-N ln2 / half)`` itself, exactly as ``state.Stock`` does.
Order events (kind 2: price, volume, side) tick the all-event clock and feed the order-flow imbalances; cancel records of
the transaction stream (kind 3, SZSE trade_type 2) tick the bothc / tradec / all clocks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

import dcp_build as B  # clocks
from state import seed_of

EVENT_DTYPE = np.dtype([("sid", "<i4"), ("mkt", "<i2"), ("kind", "<i2"), ("t", "<f8"), ("s", "<f8"), ("lsod", "<f8"),
                        ("price", "<f8"), ("f", "<f8", (7,))])
assert EVENT_DTYPE.itemsize == 96
PREV_DTYPE = np.dtype([("sid", "<i4"), ("rule", "<i4"), ("sum", "<f8"), ("wgt", "<f8")])


def seed_tag(name: str) -> str:
    """The seed column of rules.tsv: "c0" / "w0" / "wt0" ..., "-" for an unseeded rule."""
    sd = seed_of(name)
    return f"{sd[0]}{int(sd[1])}" if sd else "-"


def write_rules(emas: dict[str, tuple[str, float, str]], dst: Path, seed_all: str = "") -> list[str]:
    """rules.tsv in EMAS order; returns the rule names (the engine's rule indices). ``seed_all`` ("c0", "c600" ...) seeds
    every rule without a tag of its own from yesterday's state too (the main keys, not only the |c<N> variants)."""
    names = list(emas)
    with dst.open("w", newline="\n") as f:
        for p in names:
            clock, half, what = emas[p]
            tag = seed_tag(p) if (seed_of(p) or not seed_all) else seed_all
            f.write(f"{p}\t{clock}\t{half!r}\t{what}\t{tag}\n")
    return names


def write_prev(prev: dict[int, dict[str, float]], names: list[str], dst: Path, seed_all: str = "") -> int:
    """prev.bin: yesterday's sum / wgt for every carry rule (every rule with ``seed_all``) and stock that has them."""
    rows = []
    for i, p in enumerate(names):
        if seed_of(p) is None and not seed_all:
            continue
        base = p.partition("|")[0]
        for sid, kv in prev.items():
            w = kv.get(f"{base}.wgt")
            if w:
                rows.append((sid, i, kv[f"{base}.sum"], w))
    arr = np.array(rows, dtype=PREV_DTYPE)
    with dst.open("wb") as f:
        f.write(b"CRUPV001" + np.uint64(len(arr)).tobytes())
        f.write(arr.tobytes())
    return len(arr)


def load_events(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> np.ndarray:
    """Every snapshot and trade received by the stop as one structured array, in the engine's order."""
    S, L = B.SOD, B.LSOD
    con.register("want", pl.DataFrame({"sid": sids}))
    df = con.sql(f"""
        SELECT sid, mkt, kind, t, s, lsod, price, f0, f1, f2, f3, f4, f5, f6 FROM (
          SELECT symbol_id AS sid, market_id AS mkt, 0::SMALLINT AS kind, {B.tc(L)}::DOUBLE AS t, {S}::DOUBLE AS s, {L}::DOUBLE AS lsod,
                 spider_ts AS ls, price::DOUBLE AS price, q.bid_px[1]::DOUBLE AS f0, q.ask_px[1]::DOUBLE AS f1, high::DOUBLE AS f2,
                 low::DOUBLE AS f3, open::DOUBLE AS f4, pre_close::DOUBLE AS f5, total_amount::DOUBLE AS f6
          FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} AND {L} <= {stop}
          UNION ALL
          SELECT symbol_id, market_id, 1::SMALLINT, {B.tc(L)}::DOUBLE, {S}::DOUBLE, {L}::DOUBLE, spider_ts, price::DOUBLE,
                 volume::DOUBLE, CASE dir WHEN 1063 THEN 1.0 WHEN 1550 THEN -1.0 ELSE 0.0 END, 0.0, 0.0, 0.0, 0.0, 0.0
          FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id WHERE r.date = {date} AND r.trade_type = 1 AND {L} <= {stop}
          UNION ALL
          SELECT symbol_id, market_id, 2::SMALLINT, {B.tc(L)}::DOUBLE, {S}::DOUBLE, {L}::DOUBLE, spider_ts, price::DOUBLE,
                 volume::DOUBLE, CASE dir WHEN 1063 THEN 1.0 WHEN 1550 THEN -1.0 ELSE 0.0 END, 0.0, 0.0, 0.0, 0.0, 0.0
          FROM raw_order o JOIN want w ON w.sid = o.symbol_id WHERE o.date = {date} AND {L} <= {stop}
          UNION ALL
          SELECT symbol_id, market_id, 3::SMALLINT, {B.tc(L)}::DOUBLE, {S}::DOUBLE, {L}::DOUBLE, spider_ts, 0.0,
                 volume::DOUBLE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
          FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id WHERE r.date = {date} AND r.trade_type = 2 AND {L} <= {stop}
        ) ORDER BY sid, ls, kind""").pl()
    con.unregister("want")
    n = df.height
    out = np.empty(n, dtype=EVENT_DTYPE)
    col = lambda c: df[c].to_numpy()  # noqa: E731 -- one column as numpy (nulls -> NaN for floats)
    out["sid"], out["mkt"], out["kind"] = col("sid"), col("mkt"), col("kind")
    for c in ("t", "s", "lsod", "price"):
        out[c] = col(c).astype(np.float64)  # nulls arrive as NaN, as the Python stream's astype(float) made them
    for k in range(7):
        out["f"][:, k] = col(f"f{k}").astype(np.float64)
    return out


def write_events(arr: np.ndarray, dst: Path) -> None:
    """events.bin: header + records."""
    with dst.open("wb") as f:
        f.write(b"CRUEV001" + np.uint64(len(arr)).tobytes())
        f.write(arr.tobytes())
