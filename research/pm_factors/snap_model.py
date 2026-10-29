r"""Stage 8: our model scored on every snapshot, PM-style (a decision each time a stock's snapshot arrives, every 3-6 s).

* **model**: LightGBM on the generic tick features (``gen.``) of own_model's 5-minute panel, walk-forward: the model
  for day d is fitted on the panel days before d (the first ``--min-train`` days only train);
* **scoring**: each test day, every stock at each of its own continuous-session snapshots (local arrival time, from
  09:35), the same features (``ic_score.generic`` at those times) -> the predicted next 5-minute relative return;
* **trades**: a snapshot whose |prediction| clears a threshold opens a round trip on that stock (long: buy then sell;
  short: sell base then buy back), held ``hold`` seconds, at most one open per stock; the threshold for a target of N
  trades a day is set on the previous test day's predictions, so the first test day only calibrates;
* **costs** per round trip: 2 x (commission + other fees) + stamp duty on the sell; at mid, and as taker (entry and exit
  at the touch of the deciding snapshot and of the last one before the exit); every trade is saved for the
  passive-fill simulation.

    python research/pm_factors/snap_model.py --panel data/reports/stage7/h300/panel.parquet --out data/reports/stage8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog
import dcp_ema as E  # tick loading and per-update arrays
import feature_intraday as FI  # at_marks
import ic_score as IC  # generic tick quantities
import own_model as O  # universe, LightGBM settings, END

try:
    from numba import njit
except ImportError:  # the plain loop is fine, only slower in the threshold search
    njit = None

CONT = ((34200, 41400), (46800, 53820))  # continuous session, exchange seconds of day


def _greedy(sid: np.ndarray, t: np.ndarray, idx: np.ndarray, hold: float) -> np.ndarray:
    """Of the signal rows idx (sorted by stock, time), those that open a trade: a stock is busy until its trade ends."""
    keep = np.zeros(idx.size, np.bool_)
    last, free = -1, -1e18
    for j in range(idx.size):
        i = idx[j]
        if sid[i] != last:
            last, free = sid[i], -1e18
        if t[i] >= free:
            keep[j] = True
            free = t[i] + hold
    return keep


greedy = njit(cache=True)(_greedy) if njit else _greedy


def score_day(job: tuple) -> str:
    """One test day: features and prediction at every snapshot of every stock, quotes now and after each hold -> parquet."""
    d, model_str, feats, holds, out, dcp, threads = job
    import lightgbm as lgb

    con = duckdb.connect(str(B.CATALOG), read_only=True)
    sids, stop, _ = O.universe(dcp, d)
    up = E.load(con, int(d), sids, stop)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    frames, Xs = [], []
    for sid in sids:
        gf, gs = fast.get(sid), slow.get(sid)
        if gf is None or gf.height < 3:
            continue
        a = E.arrays(gf, "fast")
        t, s = a["t_local"], gf["s"].to_numpy()
        dec = np.zeros(t.size, bool)
        for lo, hi in CONT:
            dec |= (s >= lo) & (s < hi)
        T = np.unique(t[dec & (t >= 300)])
        if T.size == 0:
            continue
        g = IC.generic(gf, gs, T)
        Xs.append(np.column_stack([np.asarray(g.get(f.removeprefix("gen."), np.full(T.size, np.nan)), float) for f in feats]))
        ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
        okb = np.isfinite(a["bid1"]) & np.isfinite(a["ask1"])
        cols = {"sid": np.full(T.size, sid), "t": T}
        for h in (0, *holds):
            late = T + h > O.END
            for k, m in (("bid", okb), ("ask", okb), ("mid", ok)):
                v = FI.at_marks(t[m], a[f"{k}1" if k != "mid" else "mid"][m], T + h)
                cols[f"{k}{h}"] = np.where(late, np.nan, v).astype(np.float32)
        frames.append(pl.DataFrame(cols))
    if not frames:
        return f"{d}: no snapshots"
    X = np.vstack(Xs)
    X[~np.isfinite(X)] = np.nan
    df = pl.concat(frames).with_columns(pl.Series("pred", lgb.Booster(model_str=model_str).predict(X, num_threads=threads)))
    df.write_parquet(out / f"snap_{d}.parquet")
    return f"{d}: {df.height} snapshots of {len(frames)} stocks"


def arrays(df: pl.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stock, time and |prediction| of the rows whose hold h still ends inside the session (others: |prediction| -1)."""
    ok = df[f"mid{h}"].is_not_nan().to_numpy() & df["mid0"].is_not_nan().to_numpy()
    return df["sid"].to_numpy().astype(np.int64), df["t"].to_numpy().astype(float), np.where(ok, np.abs(df["pred"].to_numpy()), -1.0)


def pick(sid: np.ndarray, t: np.ndarray, a: np.ndarray, thr: float, h: int) -> np.ndarray:
    """Rows that open a trade at threshold thr."""
    idx = np.flatnonzero(a > thr)
    return idx[greedy(sid, t, idx, float(h))]


def calibrate(sid: np.ndarray, t: np.ndarray, a: np.ndarray, h: int, n: int) -> float:
    """Threshold giving about n trades on these rows (bisection between n and 200 n raw signals)."""
    v = np.sort(a[a >= 0])[::-1]
    if v.size <= n:
        return 0.0
    hi, lo = v[n - 1], v[min(v.size - 1, 200 * n)]
    if pick(sid, t, a, lo, h).size < n:
        return lo
    for _ in range(25):
        mid = (hi + lo) / 2
        if pick(sid, t, a, mid, h).size > n:
            lo = mid
        else:
            hi = mid
    return hi


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--panel", type=Path, required=True, help="own_model's panel.parquet (gen. features, y = 5-minute relative return)")
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--holds", default="60,300,900", help="holding periods, s of trading clock")
    ap.add_argument("--targets", default="250,500,1000,2000", help="trades a day the threshold aims at")
    ap.add_argument("--commission-bp", type=float, default=1.5, help="per side")
    ap.add_argument("--other-bp", type=float, default=0.5, help="other fees per side")
    ap.add_argument("--stamp-bp", type=float, default=5.0, help="on the sell")
    ap.add_argument("--workers", type=int, default=8, help="days scored in parallel")
    ap.add_argument("--threads", type=int, default=4, help="LightGBM threads per scoring worker")
    ap.add_argument("--reuse", action="store_true", help="keep snap_<day>.parquet already scored")
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage8"))
    args = ap.parse_args()
    import lightgbm as lgb

    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    holds = [int(x) for x in args.holds.split(",")]
    targets = [int(x) for x in args.targets.split(",")]
    fee = 2 * (args.commission_bp + args.other_bp) + args.stamp_bp
    panel = pl.read_parquet(args.panel)
    feats = [c for c in panel.columns if c.startswith("gen.") and panel[c].is_finite().mean() > 0.5]
    days = sorted(panel["date"].unique().to_list())
    test = days[args.min_train:]
    print(f"panel {panel.height} rows, {len(feats)} gen. features, {len(days)} days; test {test[0]} .. {test[-1]}; fee {fee:g} bp a round trip")

    jobs = []
    for d in test:
        if args.reuse and (args.out / f"snap_{d}.parquet").exists():
            continue
        tr = panel.filter(pl.col("date") < d)
        X = tr.select(feats).to_numpy().astype(float)
        X[~np.isfinite(X)] = np.nan
        m = lgb.train(O.LGB_PARAMS, lgb.Dataset(X, tr["y"].to_numpy()), O.LGB_ROUNDS)
        jobs.append((str(d), m.model_to_string(), feats, holds, args.out, args.dcp_dir, args.threads))
    print(f"{len(jobs)} models fitted ({time.perf_counter() - t0:.0f}s)")
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(score_day, jobs):
            print(f"{msg} ({time.perf_counter() - t0:.0f}s)")

    days_rows, trades, prev = [], [], None
    for d in test:
        f = args.out / f"snap_{d}.parquet"
        if not f.exists():
            continue
        df = pl.read_parquet(f).sort("sid", "t")
        if "mid300" in df.columns:
            r = (df.filter(pl.col("mid300").is_not_nan() & pl.col("mid0").is_not_nan())
                 .select(pl.corr("pred", (pl.col("mid300") / pl.col("mid0")).log(), method="spearman")).item())
            print(f"{d}: {df.height} snapshots, rank corr of prediction with the next 5 minutes {r:.4f}")
        for h in holds if prev is not None else []:
            sid, t, a = arrays(df, h)
            psid, pt, pa = arrays(prev, h)
            for n in targets:
                i = pick(sid, t, a, calibrate(psid, pt, pa, h, n), h)
                q = df[i]
                lg = q["pred"].to_numpy() > 0
                with np.errstate(divide="ignore", invalid="ignore"):
                    mid = (q[f"mid{h}"] / q["mid0"] - 1).to_numpy() * 1e4
                    mid = np.where(lg, mid, -mid)
                    tk = np.where(lg, (q[f"bid{h}"] / q["ask0"] - 1).to_numpy(), (q["bid0"] / q[f"ask{h}"] - 1).to_numpy()) * 1e4
                    spr = ((q["ask0"] - q["bid0"]) / q["mid0"]).to_numpy() * 1e4
                ft = np.isfinite(tk)  # a side of the book can be empty (limit price): no taker price there
                days_rows.append({"date": d, "hold": h, "target": n, "trades": i.size, "long": int(lg.sum()),
                                  "mid_sum": float(np.nansum(mid)), "taker_sum": float(tk[ft].sum()), "taker_n": int(ft.sum()),
                                  "win_mid": int(np.sum(mid > fee)), "spread_sum": float(spr[ft].sum())})
                trades.append(q.with_columns(pl.lit(d).alias("date"), pl.lit(h).alias("hold"), pl.lit(n).alias("target"),
                                             pl.Series("mid_bp", mid), pl.Series("taker_bp", tk)))
        prev = df
    if not days_rows:
        print("no test day after the calibration day")
        return
    per = pl.DataFrame(days_rows).with_columns(((pl.col("mid_sum") / pl.col("trades")) - fee).alias("mid_net"))
    per.write_csv(args.out / "snap_days.csv")
    pl.concat(trades, how="diagonal_relaxed").write_parquet(args.out / "trades.parquet")
    agg = per.group_by("hold", "target").agg(
        pl.len().alias("days"), (pl.col("trades").sum() / pl.len()).alias("trades_day"), (pl.col("long").sum() / pl.col("trades").sum()).alias("long_share"),
        (pl.col("mid_sum").sum() / pl.col("trades").sum()).alias("mid_gross"), (pl.col("taker_sum").sum() / pl.col("taker_n").sum()).alias("taker_gross"),
        (pl.col("spread_sum").sum() / pl.col("taker_n").sum()).alias("spread_bp"), (pl.col("win_mid").sum() / pl.col("trades").sum()).alias("win_mid"),
        (pl.col("mid_net") > 0).sum().alias("days_mid_pos"),
    ).with_columns((pl.col("mid_gross") - fee).alias("mid_net"), (pl.col("taker_gross") - fee).alias("taker_net")).sort("hold", "target")
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(f"\n== snapshot-triggered round trips, out of sample (bp per round trip; fee {fee:g}; win_mid: share with mid return > fee)")
        print(agg.select("hold", "target", "days", "trades_day", "long_share", "spread_bp", "mid_gross", "mid_net", "taker_gross", "taker_net",
                         "win_mid", "days_mid_pos"))
    print(f"tables: {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
