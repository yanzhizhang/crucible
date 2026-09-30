"""Audit the replayed order book against the exchange's own snapshots.

For every continuous-session snapshot of a stock, take our replayed best bid/ask as of the last
event at or before the snapshot's exchange time and compare with the snapshot's level 1. A book
that drifts (orders never removed, wrong keys) shows up as a falling match rate and as fills
at impossible prices in the backtest -- so this runs before trusting any backtest result.
It is also the ``consistency.book_vs_mbo_match`` check the quality standard lists as pending.

Usage (WSL): ``python research/bt/audit_book.py --date 20260615 --symbols 600519,000001``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bt import engine as E  # noqa: E402
from bt.events import RAW, load as load_events  # noqa: E402
from bt.venues import rules_for  # noqa: E402


def _arrival_match(ats, tb, ta, lo, q) -> float:
    """Level 1 as of the last event that had *arrived* before the snapshot arrived."""
    amax = np.maximum.accumulate(ats)
    idx = np.searchsorted(amax, q["spider_ts"].to_numpy(), side="right") - 1
    ok = idx >= 0
    b = np.where(ok, (tb[np.maximum(idx, 0)] + lo) / 100.0, np.nan)
    a = np.where(ok, (ta[np.maximum(idx, 0)] + lo) / 100.0, np.nan)
    return float((np.isclose(b, q["b1"].to_numpy()) & np.isclose(a, q["a1"].to_numpy())).mean())


def _window_match(xts, tb, ta, lo, q, width_ns) -> float:
    """Match if level 1 equals the snapshot at *any* event in [T, T + width).

    Needed where snapshot times are floored (SZSE stamps whole seconds): the snapshot is a
    state from somewhere inside that window, not from its label.
    """
    t = q["time"].to_numpy()
    b1 = np.round(q["b1"].to_numpy() * 100).astype(np.int64) - lo
    a1 = np.round(q["a1"].to_numpy() * 100).astype(np.int64) - lo
    i0 = np.maximum(np.searchsorted(xts, t, side="right") - 1, 0)
    i1 = np.searchsorted(xts, t + width_ns, side="left")
    hit = 0
    for k in range(t.size):
        s, e = i0[k], max(i1[k], i0[k] + 1)
        hit += bool(np.any((tb[s:e] == b1[k]) & (ta[s:e] == a1[k])))
    return hit / t.size if t.size else 0.0


def audit(date: str, symbol: str, exchange: str) -> dict:
    """Match rates of replayed level 1 vs snapshots for one stock."""
    mid = 3553 if exchange == "XSHG" else 3554
    u = pl.DataFrame({"symbol_id": [int(symbol)], "market_id": [mid]},
                     schema={"symbol_id": pl.Int32, "market_id": pl.Int16})
    ev, lim = load_events(date, u)
    ev = ev.filter((pl.col("symbol_id") == int(symbol)) & (pl.col("market_id") == mid))
    li = lim.filter((pl.col("symbol_id") == int(symbol)) & (pl.col("market_id") == mid)).row(0, named=True)
    lo = int(li["dn_tick"]) - 2
    n_ticks = int(li["up_tick"]) - lo + 3
    a = [ev[c].to_numpy().astype(np.int64) for c in
         ("kind", "xts", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")]
    z = np.zeros(0, np.int64)
    venue = E.VENUE_SSE if exchange == "XSHG" else E.VENUE_SZSE
    _, stats, _, tb, ta = E.simulate(*a, lo, n_ticks, venue, z, z, 2, 0, 0, 0, False, 0, 0, 100, 10,
                                     False, 0, 2**62, E.ALL_DAY)
    q = (
        pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
        .filter((pl.col("symbol_id") == int(symbol)) & (pl.col("market_id") == mid)
                & (pl.col("status") == 3))
        .select("time", "spider_ts", pl.col("bid_px").arr.get(0).alias("b1"),
                pl.col("ask_px").arr.get(0).alias("a1"))
        .collect()
        .sort("time")
    )
    xts = a[1]
    idx = np.searchsorted(xts, q["time"].to_numpy(), side="right") - 1
    ok = idx >= 0
    ours_b = np.where(ok & (tb[np.maximum(idx, 0)] >= 0), (tb[np.maximum(idx, 0)] + lo) / 100.0, np.nan)
    ours_a = np.where(ok & (ta[np.maximum(idx, 0)] >= 0), (ta[np.maximum(idx, 0)] + lo) / 100.0, np.nan)
    b1, a1 = q["b1"].to_numpy(), q["a1"].to_numpy()
    mb = np.isclose(ours_b, b1, atol=1e-6)
    ma = np.isclose(ours_a, a1, atol=1e-6)
    bad = np.flatnonzero(~(mb & ma))
    first_bad = None
    if bad.size:
        k = bad[0]
        first_bad = (str(np.datetime64(int(q["time"][int(k)]) + 8 * 3600 * 10**9, "ns")),
                     float(ours_b[k]), float(b1[k]), float(ours_a[k]), float(a1[k]))
    return {"symbol": symbol, "exchange": exchange, "events": len(xts), "snapshots": q.height,
            "bid_match": float(mb.mean()), "ask_match": float(ma.mean()),
            "both_match": float((mb & ma).mean()),
            "arrival_match": _arrival_match(a[2], tb, ta, lo, q),
            "window_match": _window_match(xts, tb, ta, lo, q,
                                          int(rules_for(symbol, exchange).snapshot_lag_s * 1e9)),
            "ours_ask_below_snapshot": float((ours_a < a1 - 1e-6).mean()),
            "unresolved_passive": int(stats[E.S_UNRESOLVED_PASSIVE]),
            "unresolved_cancel": int(stats[E.S_UNRESOLVED_CANCEL]),
            "out_of_band": int(stats[E.S_OUT_OF_BAND]), "first_mismatch": first_bad}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", default="20260615")
    ap.add_argument("--symbols", default="600519.XSHG,601318.XSHG,000001.XSHE,300750.XSHE")
    a = ap.parse_args()
    for s in a.symbols.split(","):
        sym, ex = s.split(".")
        print(audit(a.date, sym, ex))


if __name__ == "__main__":
    main()
