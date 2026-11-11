r"""Whole-order-book factors at ``ic_score``'s marks, from our own replay of the order and trade streams.

PM rebuilds the full book inside its factor program, so its book factors can see what a ten-level snapshot cannot: the
volume resting beyond level 10, the number of resting orders and their sizes, and how long the resting volume has been
there. This replays every stock's day with ``bt.engine``'s book (``bt.events`` cache, in exchange sequence order) and
reads the book after the last event that had ARRIVED by each mark (local arrival clock, as every ``gen.`` column).

Names follow ``lit_factors`` (whole words, no letter codes); ``lit_factors.DESCRIBE`` holds one sentence each. Written as
``<out>/date=D.parquet`` (sid, mark, one column per factor); ``ic_score --full-book <out>`` adds them as ``gen.<name>``.

    CRUCIBLE_SOURCE=dysl2 python research/pm_factors/full_book.py --dates 20261104,20261105,20261106 --workers 3
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit, types
from numba.typed import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threads  # noqa: E402
from bt import engine as E  # noqa: E402
from bt import events as EV  # noqa: E402

SSE, SZSE = 3553, 3554
NS_DAY, NS_LOCAL = 86_400_000_000_000, 8 * 3600 * 1_000_000_000
OUT = Path("/work/crucible_data/store/pm_factors") / ("full_book" if EV.SOURCE == "feitu" else f"full_book_{EV.SOURCE}")
BANDS = (0.005, 0.01, 0.02, 0.05)  # distance from the mid, as a share of it
AGES_S = (60.0, 300.0)
# raw readings per side, in this order (bid block then ask block)
RAW = ("vol", "amt", "v50", "v100", "v200", "v500", "v10lvl", "nlev", "vdist", "cnt", "cnt100", "cntbest", "vbest", "new60", "new300")
NR = len(RAW)



@njit(cache=True, nogil=True)
def book_at(kind, ats, side, tick, qty, id_a, id_b, otype, lo_tick, n_ticks, venue, rec, rec_ns):
    """Raw readings (``RAW`` for the bid side then the ask side, then crossed flag) after event ``rec[k]`` (sorted)."""
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
    out = np.full((rec.shape[0], 2 * NR + 1), np.nan)
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
            bb = n_ticks - 1
            while bb >= 0 and bidv[bb] == 0:
                bb -= 1
            ba = 0
            while ba < n_ticks and askv[ba] == 0:
                ba += 1
            if bb < 0 or ba >= n_ticks:  # one side empty: no mid, no reading
                r += 1
                continue
            mid = (bb + ba) / 2.0 + lo_tick
            v = np.zeros(2 * NR)
            for sd in range(2):
                o = sd * NR
                seen = 0
                lv = bb if sd == 0 else ba
                while 0 <= lv < n_ticks:
                    q = bidv[lv] if sd == 0 else askv[lv]
                    if q > 0:
                        px = lv + lo_tick
                        dist = abs(px - mid) / mid
                        seen += 1
                        v[o] += q
                        v[o + 1] += q * px / 100.0
                        if dist <= 0.005:
                            v[o + 2] += q
                        if dist <= 0.01:
                            v[o + 3] += q
                        if dist <= 0.02:
                            v[o + 4] += q
                        if dist <= 0.05:
                            v[o + 5] += q
                        if seen <= 10:
                            v[o + 6] += q
                        v[o + 8] += q * dist
                    lv += -1 if sd == 0 else 1
                v[o + 7] = seen
            for s in range(nslot):  # the resting orders: counts, the best level's, and the volume by age
                if o_placed[s] != 1 or o_rem[s] <= 0:
                    continue
                o = 0 if o_side[s] > 0 else NR
                lv = o_tick[s]
                if lv < 0 or lv >= n_ticks:
                    continue
                v[o + 9] += 1
                if abs(lv + lo_tick - mid) / mid <= 0.01:
                    v[o + 10] += 1
                if (o == 0 and lv == bb) or (o == NR and lv == ba):
                    v[o + 11] += 1
                    v[o + 12] += o_rem[s]
                age = (rec_ns[r] - ats[o_prio[s]]) / 1e9
                if age <= 60.0:
                    v[o + 13] += o_rem[s]
                if age <= 300.0:
                    v[o + 14] += o_rem[s]
            out[r, : 2 * NR] = v
            out[r, 2 * NR] = 1.0 if bb >= ba else 0.0
            r += 1
    return out, stats


def factors(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Named factors from ``book_at``'s raw readings (rows = marks)."""
    b = {k: raw[:, i] for i, k in enumerate(RAW)}
    a = {k: raw[:, NR + i] for i, k in enumerate(RAW)}
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        imb = lambda x, y: (x - y) / (x + y)  # noqa: E731 -- (bid - ask) / (bid + ask)
        out["full_book_depth_imbalance"] = imb(b["vol"], a["vol"])
        out["full_book_amount_imbalance"] = imb(b["amt"], a["amt"])
        for bp, k in ((50, "v50"), (100, "v100"), (200, "v200"), (500, "v500")):
            out[f"full_book_depth_imbalance_within_{bp}bp"] = imb(b[k], a[k])
        out["full_book_depth_imbalance_beyond_level10"] = imb(b["vol"] - b["v10lvl"], a["vol"] - a["v10lvl"])
        for tag, s in (("bid", b), ("ask", a)):
            out[f"{tag}_depth_share_beyond_level10_full_book"] = 1.0 - s["v10lvl"] / s["vol"]
            out[f"{tag}_depth_share_beyond_200bp_full_book"] = 1.0 - s["v200"] / s["vol"]
            out[f"{tag}_depth_weighted_distance_full_book_rel"] = s["vdist"] / s["vol"]
            out[f"{tag}_price_levels_count_log"] = np.log(s["nlev"])
            out[f"{tag}_mean_resting_order_size_log"] = np.log(s["vol"] / s["cnt"])
            out[f"full_book_{tag}_depth_log"] = np.log(s["vol"])
            for w, k in ((60, "new60"), (300, "new300")):
                out[f"{tag}_resting_volume_share_placed_last_{w}s"] = s[k] / s["vol"]
        out["depth_weighted_distance_ask_minus_bid_full_book_rel"] = a["vdist"] / a["vol"] - b["vdist"] / b["vol"]
        out["price_levels_count_imbalance"] = imb(b["nlev"], a["nlev"])
        out["resting_order_count_imbalance_full_book"] = imb(b["cnt"], a["cnt"])
        out["resting_order_count_imbalance_within_100bp"] = imb(b["cnt100"], a["cnt100"])
        out["mean_resting_order_size_bid_over_ask_log"] = np.log((b["vol"] / b["cnt"]) / (a["vol"] / a["cnt"]))
        out["best_level_mean_order_size_bid_over_ask_log"] = np.log((b["vbest"] / b["cntbest"]) / (a["vbest"] / a["cntbest"]))
        out["best_level_order_count_imbalance"] = imb(b["cntbest"], a["cntbest"])
        for w, k in ((60, "new60"), (300, "new300")):
            out[f"resting_volume_placed_last_{w}s_imbalance"] = imb(b[k], a[k])
        out["full_book_depth_over_10_levels_log"] = np.log((b["vol"] + a["vol"]) / (b["v10lvl"] + a["v10lvl"]))
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def mark_ns(t0: int, marks: np.ndarray) -> np.ndarray:
    """Trading-clock seconds (09:30 = 0, lunch removed) -> ns UTC on the day containing ``t0``."""
    day = (t0 + NS_LOCAL) // NS_DAY * NS_DAY - NS_LOCAL
    sec = np.where(marks <= 7200, 34200 + marks, 46800 + marks - 7200)
    return day + np.round(sec * 1e9).astype(np.int64)


def universe(d: str, signals: Path) -> list[int]:
    """Return the stocks PM scored that day (``pm_signals`` output); every day's file when the day has none."""
    f = signals / f"date={d}.parquet"
    fs = [f] if f.exists() else sorted(signals.glob("date=*.parquet"))
    if not fs:
        raise SystemExit(f"no pm_signals files under {signals}: pass --sids")
    return sorted({int(x) for g in fs for x in pl.read_parquet(g, columns=["sid"])["sid"].unique().to_list()})


def build_day(d: str, out: Path, marks: np.ndarray, sids: list[int]) -> str:
    """Replay the day's stocks and write ``date=D.parquet``."""
    t0 = time.time()
    uni = pl.DataFrame({"symbol_id": sids, "market_id": [SSE if c >= 600000 else SZSE for c in sids]},
                       schema={"symbol_id": pl.Int32, "market_id": pl.Int16})
    ev, lim = EV.load(d, uni)
    parts = ev.partition_by("symbol_id", "market_id", as_dict=True)
    rows, crossed, empty, bad = [], 0, 0, np.zeros(E.N_STATS, np.int64)
    for li in lim.iter_rows(named=True):
        e = parts.get((li["symbol_id"], li["market_id"]))
        if e is None or e.height == 0:
            continue
        lo = int(li["dn_tick"]) - 2
        n_ticks = int(li["up_tick"]) - lo + 3
        a = {c: e[c].to_numpy().astype(np.int64) for c in ("kind", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")}
        cut = mark_ns(int(a["ats"][0]), marks)
        rec = np.searchsorted(np.maximum.accumulate(a["ats"]), cut, side="right") - 1  # events are in sequence order
        raw, st = book_at(a["kind"], a["ats"], a["side"], a["tick"], a["qty"], a["id_a"], a["id_b"], a["otype"], lo, n_ticks,
                          E.VENUE_SSE if li["market_id"] == SSE else E.VENUE_SZSE, rec, cut)
        bad += st
        crossed += int(np.nansum(raw[:, 2 * NR]))
        empty += int(np.isnan(raw[:, 0]).sum())
        rows.append(pl.DataFrame({"sid": np.full(marks.size, li["symbol_id"], np.int64), "mark": marks, **factors(raw)}))
    if not rows:
        return f"{d}: nothing replayed (no events under {EV.RAW})"
    df = pl.concat(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"date={d}.parquet")
    n = len(rows) * marks.size
    fin = df.select(pl.exclude("sid", "mark").is_not_null().mean()).row(0)
    nb = int(lim["band_from_pre_close"].sum()) if "band_from_pre_close" in lim.columns else 0
    return (f"{d}: {len(rows)} stocks x {marks.size} marks ({nb} with no price limits in the snapshots: pre_close +-30 %), "
            f"{len(df.columns) - 2} factors; marks with a crossed book {crossed / n:.4f}, "
            f"with an empty side {empty / n:.4f}; unresolved cancels {bad[E.S_UNRESOLVED_CANCEL]:,}, unresolved passive fills "
            f"{bad[E.S_UNRESOLVED_PASSIVE]:,}, out-of-band adds {bad[E.S_OUT_OF_BAND]:,}; finite share min {min(fin):.3f} "
            f"({time.time() - t0:.0f}s) -> {out / f'date={d}.parquet'}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", required=True)
    ap.add_argument("--marks", default="five", help="the same grid as ic_score --marks: five | rand:N[:seed] | a comma list")
    ap.add_argument("--signals", type=Path, default=Path("/work/crucible_data/pm_signals"), help="pm_signals output: the day's stocks")
    ap.add_argument("--sids", default="", help="comma list of stock codes instead of --signals")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8, help="polars / numba threads per worker")
    a = ap.parse_args()
    if a.marks == "five":
        marks = np.arange(300.0, 13801.0, 300.0)
    elif a.marks.startswith("rand:"):  # as ic_score.mark_grid
        n, *seed = (int(x) for x in a.marks[5:].split(":"))
        marks = np.sort(np.random.default_rng(seed[0] if seed else 0).choice(np.arange(1, 13800), n, replace=False)).astype(float)
    else:
        marks = np.array([float(x) for x in a.marks.split(",")])
    days = [x.strip() for x in a.dates.split(",") if x.strip()]
    t0 = time.time()
    threads.cap(a.threads)  # spawned workers inherit it; a forked polars pool can deadlock, so spawn
    with ProcessPoolExecutor(max_workers=min(a.workers, len(days)), mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(build_day, d, a.out, marks, [int(x) for x in a.sids.split(",")] if a.sids else universe(d, a.signals)): d
                for d in days}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as err:  # noqa: BLE001 -- one bad day must not stop the rest
                print(f"{futs[f]}: FAILED {err!r}", flush=True)
    print(f"{len(days)} days, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
