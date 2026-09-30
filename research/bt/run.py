"""Stage 7 runner: the realism ladder and the latency sweep for one or more days.

The same toy signal (:mod:`bt.strategy`) is run through six configurations that each add one
piece of reality, so the PnL waterfall shows where money goes:

====  =======================  ============================================================
step  name                     what changes
====  =======================  ============================================================
1     ideal_mid                fill instantly at the *known* mid; no latency, no costs
2     cross_spread             fill instantly at the known touch (pay the spread)
3     latency_true_book        order meets the *true* book after the latency; walks levels
4     passive_queue            entry joins the queue at the touch; timeout -> cross the rest
5     fees                     + commission (min per order), transfer fee, stamp duty on sells
6     t1_restore               + T+1: sells only out of the base; forced close at 14:56
====  =======================  ============================================================

Plus a latency sweep on step 3 (send latency 0 ... 300 ms).

Universe: the ``--top`` most-traded CSI 300 names of the day (by bar amount). Each stock is an
independent single-threaded replay; stocks run on a thread pool (the kernel releases the GIL).
Results: ``/work/crucible_data/store/bt/date=<D>/{fills,summary,sweep}.parquet``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resmon import ResourceMonitor, append_history  # noqa: E402
from bt import engine as E  # noqa: E402
from bt.events import load as load_events  # noqa: E402
from bt.pnl import decompose, summarize, to_frame, with_fees  # noqa: E402
from bt.strategy import intents as make_intents  # noqa: E402
from bt.venues import continuous_windows, rules_for  # noqa: E402
from toll.costs import CostModel  # noqa: E402

DATA = Path("/work/crucible_data")
REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "data" / "public_cache"
MS = 1_000_000
S = 1_000 * MS
_LOCAL = 8 * 3600 * S


@dataclass(frozen=True)
class Config:
    """One rung of the ladder."""

    name: str
    fill_model: int
    passive: bool = False
    fees: bool = False
    t1: bool = False
    lat_sig: int = 1 * MS
    lat_send: int = 3 * MS
    lat_report: int = 3 * MS
    passive_timeout: int = 30 * S
    hold: int = 5 * 60 * S
    max_walk: int = 10


LADDER = [
    Config("1_ideal_mid", E.FILL_MID_KNOWN),
    Config("2_cross_spread", E.FILL_TOUCH_KNOWN),
    Config("3_latency_true_book", E.FILL_EXCHANGE),
    Config("4_passive_queue", E.FILL_EXCHANGE, passive=True),
    Config("5_fees", E.FILL_EXCHANGE, passive=True, fees=True),
    Config("6_t1_restore", E.FILL_EXCHANGE, passive=True, fees=True, t1=True),
]
SWEEP_MS = (0, 1, 3, 10, 30, 100, 300)


def universe(date: str, top: int) -> pl.DataFrame:
    """Top ``top`` CSI 300 names of the day by traded amount, with keys for every data source."""
    w = pl.read_parquet(sorted((PUBLIC / "index_weights" / "index=000300").glob("*.parquet"))[-1])
    w = w.select(pl.col("code").alias("symbol"),
                 pl.when(pl.col("exchange") == "SH").then(pl.lit("XSHG")).otherwise(pl.lit("XSHE"))
                 .alias("exchange"))
    bars = pl.read_parquet(DATA / "store" / "bars" / "1min" / f"date={date}" / "part.parquet")
    amt = bars.group_by("symbol", "exchange").agg(pl.col("amount").sum(), pl.col("pre_close").first())
    u = w.join(amt, on=["symbol", "exchange"]).sort("amount", descending=True).head(top)
    return u.with_columns(
        pl.col("symbol").cast(pl.Int32).alias("symbol_id"),
        pl.when(pl.col("exchange") == "XSHG").then(3553).otherwise(3554).cast(pl.Int16).alias("market_id"),
    )


def _trip_qty(pre_close: float, symbol: str, notional: float) -> int:
    lots = max(int(notional / pre_close / 100), 1) * 100
    return max(lots, 200) if symbol.startswith("688") else lots  # STAR minimum order 200


def run_day(date: str, top: int, threads: int, notional: float) -> None:
    """Ladder + sweep for one day; writes fills / summary / sweep parquet."""
    u = universe(date, top)
    print(f"{date}: universe {u.height} names, {u['amount'].sum() / 1e8:.1f} 100M CNY traded")
    ev, lim = load_events(date, u.select("symbol_id", "market_id"))
    print(f"  events {ev.height:,} rows")
    bars = pl.read_parquet(DATA / "store" / "bars" / "1min" / f"date={date}" / "part.parquet").join(
        u.select("symbol", "exchange"), on=["symbol", "exchange"], how="semi")
    it = make_intents(bars)
    eod = (np.datetime64(f"{date[:4]}-{date[4:6]}-{date[6:]}T14:56:00", "ns").astype(np.int64)
           - _LOCAL)
    sess = continuous_windows(date)
    per_sym = {}
    for key, g in ev.partition_by(["symbol_id", "market_id"], as_dict=True).items():
        per_sym[key] = [g[c].to_numpy().astype(np.int64) for c in
                        ("kind", "xts", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")]
    lim_d = {(r["symbol_id"], r["market_id"]): r for r in lim.iter_rows(named=True)}
    it_d = {s: g for (s,), g in it.partition_by(["symbol"], as_dict=True).items()}
    jobs = []
    for r in u.iter_rows(named=True):
        key = (r["symbol_id"], r["market_id"])
        if key not in per_sym or key not in lim_d:
            continue
        li = lim_d[key]
        lo = int(li["dn_tick"]) - 2
        n_ticks = int(li["up_tick"]) - lo + 3
        g = it_d.get(r["symbol"])
        t_arr = g["t_local"].to_numpy().astype(np.int64) if g is not None else np.zeros(0, np.int64)
        s_arr = g["side"].to_numpy().astype(np.int64) if g is not None else np.zeros(0, np.int64)
        q = _trip_qty(r["pre_close"], r["symbol"], notional)
        venue = rules_for(r["symbol"], r["exchange"]).engine_code
        jobs.append((r["symbol"], per_sym[key], lo, n_ticks, venue, t_arr, s_arr, q))

    def one(cfg: Config, job) -> tuple[pl.DataFrame, np.ndarray]:
        sym, a, lo, n_ticks, venue, t_arr, s_arr, q = job
        F, stats, m_end, _, _ = E.simulate(*a, lo, n_ticks, venue, t_arr, s_arr, cfg.fill_model, cfg.lat_sig,
                                     cfg.lat_send, cfg.lat_report, cfg.passive, cfg.passive_timeout,
                                     cfg.hold, q, cfg.max_walk, cfg.t1, 20 * q, eod, sess)
        return to_frame(F, lo, m_end, sym, date), stats

    def run_cfg(cfg: Config) -> tuple[pl.DataFrame, dict]:
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: one(cfg, j), jobs))
        fills = pl.concat([f for f, _ in res if f.height], how="vertical") if res else pl.DataFrame()
        fills = decompose(with_fees(fills, CostModel() if cfg.fees else None, date))
        stats = np.sum([s for _, s in res], axis=0)
        summ = summarize(fills) | {
            "config": cfg.name, "date": date, "trips": int(stats[E.S_TRIPS]),
            "skip_t1": int(stats[E.S_SKIP_T1]), "skip_busy": int(stats[E.S_SKIP_BUSY]),
            "passive_filled": int(stats[E.S_PASSIVE_FILLED]),
            "passive_crossed": int(stats[E.S_PASSIVE_CROSSED]),
            "unresolved_passive": int(stats[E.S_UNRESOLVED_PASSIVE]),
            "unresolved_cancel": int(stats[E.S_UNRESOLVED_CANCEL]),
            "skip_session": int(stats[E.S_SKIP_SESSION]),
            "defer_session": int(stats[E.S_DEFER_SESSION]),
            "walk_crossed": int(stats[E.S_WALK_CROSSED]),
            "notional": float((fills["qty"] * fills["price"]).sum()) if fills.height else 0.0,
        }
        return fills.with_columns(pl.lit(cfg.name).alias("config")), summ

    out = DATA / "store" / "bt" / f"date={date}"
    out.mkdir(parents=True, exist_ok=True)
    fills_all, rows = [], []
    for cfg in LADDER:
        with ResourceMonitor("W4.backtest", params={"date": date, "config": cfg.name}) as mon:
            f, s = run_cfg(cfg)
            mon.rows = int(sum(len(j[1][0]) for j in jobs))
        assert mon.stats is not None
        append_history(mon.stats, DATA / "bench" / "history.parquet")
        fills_all.append(f)
        rows.append(s | {"wall_s": mon.stats.wall_s})
        print(f"  {cfg.name:22s} total {s['total']:>12,.0f}  signal {s['signal']:>11,.0f}  "
              f"latency {s['latency']:>10,.0f}  spread {s['spread']:>11,.0f}  fees {s['fees']:>10,.0f}  "
              f"restore {s['restore']:>9,.0f}  trips {s['trips']:>5}  "
              f"{mon.stats.rows_per_s / 1e6:5.1f}M ev/s")
    pl.concat(fills_all, how="diagonal_relaxed").write_parquet(out / "fills.parquet")
    pl.DataFrame(rows).write_parquet(out / "summary.parquet")
    sweep = []
    base = LADDER[2]
    for ms in SWEEP_MS:
        _, s = run_cfg(replace(base, name=f"sweep_{ms}ms", lat_send=ms * MS))
        sweep.append(s | {"lat_send_ms": ms})
        print(f"  latency sweep {ms:>4} ms: total {s['total']:>12,.0f}  latency {s['latency']:>10,.0f}")
    pl.DataFrame(sweep).write_parquet(out / "sweep.parquet")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="20260615,20260805")
    ap.add_argument("--top", type=int, default=30, help="most-traded CSI 300 names to simulate")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--notional", type=float, default=100_000.0, help="CNY per trip")
    a = ap.parse_args()
    for d in filter(None, a.dates.split(",")):
        run_day(d, a.top, a.threads, a.notional)


if __name__ == "__main__":
    main()
