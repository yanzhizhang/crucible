r"""Order-flow features from the order and trade streams at given times (what a snapshot cannot show).

Per stock, at each time of ``T`` (trading clock, local arrival), over trailing windows:

* trades by ticket size (amount buckets, yuan: small < 40k, mid 40k-200k, large 200k-1M, extra > 1M -- the usual
  四档 split): signed-amount imbalance of the large + extra tickets (``big_imb``), of the small ones (``small_imb``),
  and the big tickets' share of the amount;
* new orders (SSE ``update_type`` 1, every SZSE order): amount imbalance of all new orders and of the big ones;
* cancels (SSE ``update_type`` 2 orders, SZSE ``trade_type`` 2 transactions): buy-minus-sell cancel imbalance and
  cancelled over submitted amount;
* price impact: slope of the mid log change on the net signed volume between consecutive snapshots (Kyle lambda, per
  1e6 shares, over 5 and 15 minutes);
* queue refill: the bid1 / ask1 volume added while the price stayed, over the window, as an imbalance.

Everything is a running total read off at ``T`` (``searchsorted``), so one stock-day costs O(events + marks).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # trading clock SQL
import feature_intraday as FI  # at_marks

BUCKETS = (4e4, 2e5, 1e6)  # amount edges, yuan
WINDOWS = (60, 300, 900)
BUY, SELL = 1063, 1550


def load_orders(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> pl.DataFrame:
    """Load the new orders and cancels received by the stop: kind 1 = new order, 2 = cancel; side +1 buy / -1 sell; amount in yuan."""
    S, L = B.SOD, B.LSOD
    con.register("want_o", pl.DataFrame({"sid": sids}))
    df = con.sql(f"""
        SELECT symbol_id AS sid, {B.tc(L)} AS t_local, {S} AS s, update_type AS kind,
               CASE dir WHEN {BUY} THEN 1.0 WHEN {SELL} THEN -1.0 ELSE 0.0 END AS side, volume::DOUBLE AS vol, price * volume AS amt
        FROM raw_order o JOIN want_o w ON w.sid = o.symbol_id
        WHERE o.date = {date} AND update_type IN (1, 2) AND {L} <= {stop}
        UNION ALL
        SELECT symbol_id, {B.tc(L)}, {S}, 2, CASE dir WHEN {BUY} THEN 1.0 WHEN {SELL} THEN -1.0 ELSE 0.0 END, volume::DOUBLE, price * volume
        FROM raw_transaction r JOIN want_o w ON w.sid = r.symbol_id
        WHERE r.date = {date} AND trade_type = 2 AND {L} <= {stop}
        ORDER BY 1, 2""").pl()
    con.unregister("want_o")
    return df


def _upto(t: np.ndarray, w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Read the running total of w (events at sorted times t) off at each time x."""
    return np.concatenate([[0.0], np.cumsum(w)])[np.searchsorted(t, x, side="right")]


def _win(t: np.ndarray, v: np.ndarray, T: np.ndarray, w: float) -> np.ndarray:
    """Total of v over (T - w, T]."""
    return _upto(t, v, T) - _upto(t, v, T - w)


def flow(go: pl.DataFrame | None, gs: pl.DataFrame | None, gf: pl.DataFrame | None, T: np.ndarray) -> dict[str, np.ndarray]:
    """One stock: feature name -> values at T. go = orders (load_orders), gs = trades (dcp_ema.load 'slow'), gf = snapshots ('fast')."""
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if gs is not None and gs.height > 2:
            t = gs["t_local"].to_numpy().astype(float)
            amt, side = gs["amt"].to_numpy().astype(float), gs["side"].to_numpy().astype(float)
            cont = gs["s"].to_numpy() >= 34200
            big, small = (amt >= BUCKETS[1]) & cont, (amt < BUCKETS[0]) & cont
            for w in WINDOWS:
                ba, bn = _win(t, amt * big, T, w), _win(t, amt * side * big, T, w)
                sa, sn = _win(t, amt * small, T, w), _win(t, amt * side * small, T, w)
                out[f"big_imb{w}"] = bn / ba
                out[f"small_imb{w}"] = sn / sa
                out[f"big_share{w}"] = ba / _win(t, amt * cont, T, w)
        if go is not None and go.height > 2:
            t = go["t_local"].to_numpy().astype(float)
            kind, side = go["kind"].to_numpy(), go["side"].to_numpy().astype(float)
            amt = go["amt"].to_numpy().astype(float)
            cont = go["s"].to_numpy() >= 34200
            new, can = (kind == 1) & cont, (kind == 2) & cont
            bign = new & (amt >= BUCKETS[1])
            for w in WINDOWS:
                na = _win(t, amt * new, T, w)
                out[f"ord_imb{w}"] = _win(t, amt * side * new, T, w) / na
                out[f"ord_big_imb{w}"] = _win(t, amt * side * bign, T, w) / _win(t, amt * bign, T, w)
                out[f"cxl_imb{w}"] = _win(t, amt * side * can, T, w) / _win(t, amt * can, T, w)
                out[f"cxl_rate{w}"] = _win(t, amt * can, T, w) / na
        if gf is not None and gf.height > 2 and gs is not None and gs.height > 2:
            tf = gf["t_local"].to_numpy().astype(float)
            b1, a1 = gf["b1"].to_numpy().astype(float), gf["a1"].to_numpy().astype(float)
            bv, av = gf["bv1"].to_numpy().astype(float), gf["av1"].to_numpy().astype(float)
            two = (b1 > 0) & (a1 > 0)
            lm = np.where(two, np.log((b1 + a1) / 2), np.nan)
            ts = gs["t_local"].to_numpy().astype(float)
            nv = _upto(ts, gs["vol"].to_numpy().astype(float) * gs["side"].to_numpy().astype(float), tf) / 1e6
            dy, dx = np.diff(lm, prepend=np.nan), np.diff(nv, prepend=np.nan)
            okk = np.isfinite(dy) & np.isfinite(dx)
            xy, xx = np.where(okk, dy * dx, 0.0), np.where(okk, dx * dx, 0.0)
            for w in (300, 900):
                out[f"lambda{w}"] = _win(tf, xy, T, w) / _win(tf, xx, T, w)
            same_b = np.concatenate([[False], b1[1:] == b1[:-1]])
            same_a = np.concatenate([[False], a1[1:] == a1[:-1]])
            rb = np.where(same_b, np.clip(np.diff(bv, prepend=0.0), 0, None), 0.0)
            ra = np.where(same_a, np.clip(np.diff(av, prepend=0.0), 0, None), 0.0)
            for w in (60, 300):
                fb, fa = _win(tf, rb, T, w), _win(tf, ra, T, w)
                out[f"refill_imb{w}"] = (fb - fa) / (fb + fa)
                out[f"refill{w}"] = (fb + fa) / (FI.at_marks(tf, bv + av, T))
    return out
