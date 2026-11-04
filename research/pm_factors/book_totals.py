r"""Whole-book resting totals at each 5-minute bar end, from our own order-book replay.

PM's raw feed (``/data/prod/mp``, see ``research/decode_pm_mp.py``) carries only ten snapshot levels -- no
whole-book total volume and no weighted average price -- yet ZZAL an / bn track ``total x weighted average
price`` at ratio 1.000 and are never exact. So PM most likely sums its own reconstructed book. This replays
every stock's day with ``bt.engine``'s book (``bt.events`` cache) and records, at each bar end, the resting
volume and amount per side::

    btot_x atot_x bamt_x aamt_x   book after the last event with exchange time <= bar end
    btot_a atot_a bamt_a aamt_a   the same on the trade clock (SSE: receive time - 25 ms, running max)

The 15:00 bar reads the book at 14:57:00 (continuous session end), like the snapshot columns. Written as
``<basics>/date=D/book.parquet`` (symbol_id, minute, ...); ``window_probe.py`` reads it as ``book.<col>``.

    python research/pm_factors/book_totals.py --dates 20261020,20261021 --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit, types
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bt import engine as E  # noqa: E402
from bt.events import load as load_events  # noqa: E402

BASICS_PM = Path("/work/crucible_data/store/pm_factors/basics_pm")
SSE, SZSE = 3553, 3554
NS_MIN, NS_DAY, NS_LOCAL = 60_000_000_000, 86_400_000_000_000, 8 * 3600 * 1_000_000_000
SSE_LAG_NS = 25_000_000
BARS = np.array([*range(575, 691, 5), *range(785, 901, 5)], np.int64)
CONT_END = 897


@njit(cache=True, nogil=True)
def totals_at(kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, venue, rec):
    """Resting (bid vol, ask vol, bid amount, ask amount) after event ``rec[k]`` (sorted, -1 = before any)."""
    n = kind.shape[0]
    bidv = np.zeros(n_ticks, np.int64)
    askv = np.zeros(n_ticks, np.int64)
    oid = Dict.empty(key_type=types.int64, value_type=types.int64)
    o_side = np.zeros(n + 1, np.int64)
    o_tick = np.zeros(n + 1, np.int64)
    o_rem = np.zeros(n + 1, np.int64)
    o_prio = np.zeros(n + 1, np.int64)
    o_placed = np.zeros(n + 1, np.int8)
    stats = np.zeros(E.N_STATS, np.int64)
    pkey = np.int64(-1)
    nslot = 0
    best_bid = -1
    best_ask = n_ticks
    out = np.zeros((rec.shape[0], 4), np.float64)
    r = 0
    while r < rec.shape[0] and rec[r] < 0:
        r += 1
    for i in range(n):
        if venue == E.VENUE_SZSE:
            best_bid, best_ask, nslot, pkey = E._apply_szse(
                i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
                o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats, pkey)
        else:
            best_bid, best_ask, nslot = E._apply_sse(
                i, kind, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, oid, o_side, o_tick,
                o_rem, o_prio, o_placed, nslot, bidv, askv, best_bid, best_ask, stats)
        while r < rec.shape[0] and rec[r] == i:
            bv = 0.0
            av = 0.0
            ba = 0.0
            aa = 0.0
            for lv in range(n_ticks):
                if bidv[lv] > 0:
                    bv += bidv[lv]
                    ba += bidv[lv] * (lv + lo_tick) / 100.0
                if askv[lv] > 0:
                    av += askv[lv]
                    aa += askv[lv] * (lv + lo_tick) / 100.0
            out[r, 0] = bv
            out[r, 1] = av
            out[r, 2] = ba
            out[r, 3] = aa
            r += 1
    return out


def cuts(t0: int) -> np.ndarray:
    """Bar-end instants (ns UTC) for the day containing ``t0``; the 15:00 bar reads 14:57:00."""
    day = (t0 + NS_LOCAL) // NS_DAY * NS_DAY - NS_LOCAL
    return day + np.minimum(BARS, CONT_END) * NS_MIN


def build_day(d: str, out: Path) -> str:
    """Replay every stock of the day's basics universe and write ``book.parquet``."""
    m5 = out / f"date={d}" / "m5.parquet"
    if not m5.exists():
        return f"{d}: no {m5}"
    codes = pl.read_parquet(m5, columns=["symbol_id"])["symbol_id"].unique().cast(pl.Int64).to_list()
    uni = pl.DataFrame({"symbol_id": codes, "market_id": [SSE if c >= 600000 else SZSE for c in codes]},
                       schema={"symbol_id": pl.Int32, "market_id": pl.Int16})
    ev, lim = load_events(d, uni)
    parts = ev.partition_by("symbol_id", "market_id", as_dict=True)
    rows = []
    for li in lim.iter_rows(named=True):
        e = parts.get((li["symbol_id"], li["market_id"]))
        if e is None or e.height == 0:
            continue
        lo = int(li["dn_tick"]) - 2
        n_ticks = int(li["up_tick"]) - lo + 3
        a = {c: e[c].to_numpy().astype(np.int64) for c in ("kind", "xts", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")}
        sse = li["market_id"] == SSE
        cut = cuts(int(a["xts"][0]))
        rx = np.searchsorted(np.maximum.accumulate(a["xts"]), cut, side="right") - 1  # events are in sequence order
        clk = np.maximum.accumulate(a["ats"] - SSE_LAG_NS) if sse else a["xts"]
        ra = np.searchsorted(clk, cut, side="right") - 1
        args = (a["kind"], a["side"], a["tick"], a["qty"], a["id_a"], a["id_b"], a["otype"], lo, n_ticks,
                E.VENUE_SSE if sse else E.VENUE_SZSE)
        tx = totals_at(*args, rx)
        ta = totals_at(*args, ra)
        rows.append(pl.DataFrame({"symbol_id": np.full(BARS.size, li["symbol_id"], np.int64), "minute": BARS,
                                  "btot_x": tx[:, 0], "atot_x": tx[:, 1], "bamt_x": tx[:, 2], "aamt_x": tx[:, 3],
                                  "btot_a": ta[:, 0], "atot_a": ta[:, 1], "bamt_a": ta[:, 2], "aamt_a": ta[:, 3]}))
    if not rows:
        return f"{d}: nothing replayed"
    pl.concat(rows).write_parquet(out / f"date={d}" / "book.parquet")
    return f"{d}: {len(rows)} stocks"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--out", type=Path, default=BASICS_PM, help="basics directory (date=D/m5.parquet gives the universe)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    days = a.dates.split(",")
    with ProcessPoolExecutor(max_workers=min(a.workers, len(days))) as ex:
        futs = {ex.submit(build_day, d, a.out): d for d in days}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as err:  # noqa: BLE001 -- one bad day must not stop the rest
                print(f"{futs[f]}: FAILED {err!r}", flush=True)
    print(f"{len(days)} days, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
