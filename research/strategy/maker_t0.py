r"""Our own T0 strategy, backtested with passive (maker) fills on the real trade-by-trade queue.

The earlier result (own_model, 20 walk-forward days): the 5-minute model's top predictions earn ~1-7 bp at the mid and
lose after crossing the spread, so a taker T0 at 9 bp a round trip cannot pay. This backtest asks whether queuing for
the fill instead of paying the spread does.

Rules, per stock (one position at a time, on the base holding -- a sell first is selling the base and buying it back):

* **signal**: at each prediction mark, |pred| above a per-day threshold set from the PREVIOUS day's predictions so that
  about ``--target`` entries a day are proposed (no look-ahead);
* **entry**: a limit order at the near touch -- buy at bid1 / sell at ask1 of the last snapshot -- queued behind the
  displayed size (x ``--queue``: 1 = the whole queue ahead, 0.5 = half of it cancels before us); it fills when a trade
  prints through the price, or when trades AT the price aimed at our side (sell-initiated at our bid, buy-initiated at
  our ask) have eaten the queue ahead plus our size; cancelled after ``--ttl`` s unfilled;
* **exit**: on the far touch at the fill time, queued the same way, until ``hold`` s after the fill; still open then ->
  crossed at the market (taker) at the snapshot quote; everything closes by 14:55;
* **costs**: commission + other fees per side, stamp duty on the sell leg (defaults 1.5 + 0.5 per side, 5 on the sell =
  9 bp a round trip).

Reported per (target, hold, queue): entries proposed and filled, round trips a day, gross and net bp per round trip,
the share of exits that had to cross, the adverse move 60 s after the entry fill (bp, signed for our side), and net CNY a
day at ``--notional`` per order; next to it the same signals traded as a taker (cross at entry and at exit).

    python research/strategy/maker_t0.py --preds data/reports/stage7/walk/preds.parquet --targets 200,500,1000 --holds 300,900,1800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

import dcp_build as B  # catalog path, state files
import dcp_ema as E  # tick loading (trading clock, local arrival)

CLOSE = 14100.0  # 14:55 on the trading clock: everything flat by then
CATALOG = Path("/work/crucible_data/catalog.duckdb")


class Book:
    """One stock-day: snapshots (t, bid1, ask1, bid size, ask size) and trades (t, price, volume, aggressor side)."""

    def __init__(self, gf: pl.DataFrame, gs: pl.DataFrame | None) -> None:
        f = lambda g, c: g[c].to_numpy().astype(float)  # noqa: E731 -- column as float array
        self.qt, self.b1, self.a1, self.bv, self.av = (f(gf, c) for c in ("t_local", "b1", "a1", "bv1", "av1"))
        if gs is not None and gs.height:
            self.tt, self.px, self.vol, self.side = (f(gs, c) for c in ("t_local", "price", "vol", "side"))
        else:
            self.tt = self.px = self.vol = self.side = np.array([])

    def quote(self, t: float) -> tuple[float, float, float, float] | None:
        """Last snapshot at or before t with both sides quoted."""
        i = int(np.searchsorted(self.qt, t, side="right")) - 1
        while i >= 0 and not (self.b1[i] > 0 and self.a1[i] > 0):
            i -= 1
        return (self.b1[i], self.a1[i], self.bv[i], self.av[i]) if i >= 0 else None

    def passive_fill(self, buy: bool, price: float, t0: float, t1: float, ahead: float, size: float) -> float | None:
        """Fill time of a resting limit order placed at t0, alive until t1, or None."""
        i0, i1 = int(np.searchsorted(self.tt, t0, side="right")), int(np.searchsorted(self.tt, t1, side="right"))
        if i1 <= i0:
            return None
        tt, px, vol, sd = self.tt[i0:i1], self.px[i0:i1], self.vol[i0:i1], self.side[i0:i1]
        through = (px < price) if buy else (px > price)  # a print through our price: we were filled before it
        at = (px == price) & ((sd < 0) if buy else (sd > 0))  # aggressors at our price working down the queue
        cum = np.cumsum(np.where(at, vol, 0.0))
        k_at = int(np.searchsorted(cum, ahead + size - 1e-9))  # first trade where the queue ahead and our size are gone
        k_th = int(np.argmax(through)) if through.any() else tt.size
        k = min(k_at, k_th)
        return float(tt[k]) if k < tt.size else None


def round_trip(bk: Book, buy: bool, t: float, hold: float, ttl: float, queue: float, size: float) -> dict | None:
    """One passive round trip from a signal at t; None when the entry never fills."""
    q = bk.quote(t)
    if q is None:
        return None
    b1, a1, bv, av = q
    entry = b1 if buy else a1
    tf = bk.passive_fill(buy, entry, t, min(t + ttl, CLOSE), queue * (bv if buy else av), size)
    if tf is None:
        return None
    q = bk.quote(tf)
    if q is None:
        return None
    exit_px = q[1] if buy else q[0]  # resting on the far touch
    t_end = min(tf + hold, CLOSE)
    tx = bk.passive_fill(not buy, exit_px, tf, t_end, queue * (q[3] if buy else q[2]), size)
    crossed = tx is None
    if crossed:  # cross at the market at the deadline
        qe = bk.quote(t_end)
        if qe is None:
            return None
        exit_px, tx = (qe[0] if buy else qe[1]), t_end
    qa = bk.quote(tf + 60)
    mid60 = (qa[0] + qa[1]) / 2 if qa else np.nan
    sgn = 1.0 if buy else -1.0
    return {"entry": entry, "exit": exit_px, "t_fill": tf, "t_exit": tx, "crossed": crossed,
            "gross_bp": sgn * (exit_px / entry - 1) * 1e4, "adverse60_bp": sgn * (mid60 / entry - 1) * 1e4}


def taker_trip(bk: Book, buy: bool, t: float, hold: float) -> float | None:
    """The same signal crossed at entry and at exit (bp, gross)."""
    q0, q1 = bk.quote(t), bk.quote(min(t + hold, CLOSE))
    if q0 is None or q1 is None:
        return None
    return (q1[0] / q0[1] - 1) * 1e4 if buy else (q0[0] / q1[1] - 1) * 1e4


def thresholds(preds: pl.DataFrame, col: str, target: int) -> dict[int, float]:
    """Per day: the |pred| level that the previous day's predictions exceed about target times (a fraction carried over)."""
    days = sorted(preds["date"].unique().to_list())
    out = {}
    for d0, d1 in zip(days, days[1:], strict=False):
        a = np.abs(preds.filter(pl.col("date") == d0)[col].to_numpy())
        a = a[np.isfinite(a)]
        if a.size > target:
            out[d1] = float(np.quantile(a, 1 - target / a.size))
    return out


def simulate_day(con: duckdb.DuckDBPyConnection, d: int, sig: pl.DataFrame, col: str, holds: list[float], queues: list[float],
                 ttl: float, notional: float) -> list[dict]:
    """Every signal of one day through every (hold, queue): passive round trips and the taker counterfactual."""
    sids = sorted(sig["sid"].unique().to_list())
    up = E.load(con, d, sids, 54150.005)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")} if up["fast"].height else {}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")} if up["slow"].height else {}
    rows = []
    for sid, g in sig.group_by("sid"):
        sid = int(sid[0])
        if sid not in fast:
            continue
        bk = Book(fast[sid], slow.get(sid))
        g = g.sort("t")
        ts, ps = g["t"].to_numpy().astype(float), g[col].to_numpy()
        for hold in holds:
            for queue in queues:
                busy = -1.0  # one position at a time: signals before the last exit are skipped
                for t, p in zip(ts, ps, strict=True):
                    if t < busy or t + 60 > CLOSE:
                        continue
                    buy = p > 0
                    q = bk.quote(t)
                    if q is None:
                        continue
                    px = q[0] if buy else q[1]
                    size = max(100.0, np.floor(notional / px / 100) * 100)
                    r = round_trip(bk, buy, t, hold, ttl, queue, size)
                    tk = taker_trip(bk, buy, t, hold)
                    row = {"date": d, "sid": sid, "t": t, "hold": hold, "queue": queue, "buy": buy, "filled": r is not None,
                           "taker_gross_bp": tk if tk is not None else np.nan, "notional": size * px}
                    if r:
                        row.update(r)
                        busy = r["t_exit"]
                    rows.append(row)
    return rows


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preds", type=Path, required=True, help="own_model --save-preds output (date, sid, t, pred_*)")
    ap.add_argument("--model", default="lgbm", help="which pred_<model> column")
    ap.add_argument("--targets", default="200,500,1000", help="entries proposed a day over the universe")
    ap.add_argument("--holds", default="300,900,1800", help="holding time after the entry fill, s")
    ap.add_argument("--queues", default="1.0,0.5", help="share of the displayed queue ahead of us")
    ap.add_argument("--ttl", type=float, default=60.0, help="entry order lifetime, s")
    ap.add_argument("--notional", type=float, default=100000.0, help="CNY per order")
    ap.add_argument("--commission-bp", type=float, default=1.5)
    ap.add_argument("--other-bp", type=float, default=0.5)
    ap.add_argument("--stamp-bp", type=float, default=5.0)
    ap.add_argument("--catalog", type=Path, default=CATALOG)
    ap.add_argument("--out", type=Path, default=Path("data/reports/strategy"))
    a = ap.parse_args()
    col = f"pred_{a.model}"
    preds = pl.read_parquet(a.preds).filter(pl.col(col).is_finite())
    fee = 2 * (a.commission_bp + a.other_bp) + a.stamp_bp
    holds = [float(x) for x in a.holds.split(",")]
    queues = [float(x) for x in a.queues.split(",")]
    con = duckdb.connect(str(a.catalog), read_only=True)
    a.out.mkdir(parents=True, exist_ok=True)
    allrows = []
    for target in [int(x) for x in a.targets.split(",")]:
        thr = thresholds(preds, col, target)
        for d, level in sorted(thr.items()):
            sig = preds.filter((pl.col("date") == d) & (pl.col(col).abs() >= level))
            if sig.is_empty():
                continue
            rows = simulate_day(con, int(d), sig, col, holds, queues, a.ttl, a.notional)
            allrows += [{**r, "target": target} for r in rows]
            print(f"target {target} {d}: {sig.height} signals", flush=True)
    if not allrows:
        print("nothing simulated")
        return
    t = pl.DataFrame(allrows, infer_schema_length=None)
    t.write_parquet(a.out / "trips.parquet")
    ndays = t["date"].n_unique()
    fl = t.filter(pl.col("filled"))
    agg = (t.group_by("target", "hold", "queue").agg(
              pl.len().alias("signals"), pl.col("filled").mean().alias("fill_rate"),
              pl.col("taker_gross_bp").mean().alias("taker_gross"), (pl.col("taker_gross_bp") - fee).mean().alias("taker_net"))
           .join(fl.group_by("target", "hold", "queue").agg(
              (pl.len() / ndays).alias("trips_day"), pl.col("gross_bp").mean().alias("gross"), (pl.col("gross_bp") - fee).mean().alias("net"),
              (pl.col("gross_bp") > fee).mean().alias("win"), pl.col("crossed").mean().alias("crossed"),
              pl.col("adverse60_bp").mean().alias("adverse60"),
              (((pl.col("gross_bp") - fee) * pl.col("notional")).sum() / 1e4 / ndays).alias("net_cny_day")), on=["target", "hold", "queue"], how="left")
           .sort("target", "hold", "queue"))
    daily = (fl.with_columns(((pl.col("gross_bp") - fee) * pl.col("notional") / 1e4).alias("pnl"))
             .group_by("target", "hold", "queue", "date").agg(pl.col("pnl").sum())
             .group_by("target", "hold", "queue").agg((pl.col("pnl") > 0).mean().alias("up_days"),
                                                      (pl.col("pnl").mean() / pl.col("pnl").std() * np.sqrt(250)).alias("sharpe_ann")))
    agg = agg.join(daily, on=["target", "hold", "queue"], how="left")
    agg.write_csv(a.out / "summary.csv")
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=220, float_precision=2):
        print(f"\n== passive T0 over {ndays} days, fees {fee:g} bp a round trip; gross / net / adverse in bp per round trip, "
              f"net CNY a day at {a.notional:g} per order; taker = the same signals crossed both ways")
        print(agg)


if __name__ == "__main__":
    main()
