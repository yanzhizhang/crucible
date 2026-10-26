r"""Stage 7: our own tk5m-style model on the reproduced state and tick features, scored out of sample.

Not PM's formulas one by one: the reproduced state rules plus generic tick quantities, turned into
scale-free features at every 5-minute mark, a model fitted on some days and scored on the others.

* **features** per (day, mark, stock): every generic tick quantity of ``ic_score.generic``
  (returns 5 s - 30 min of mid / microprice / last, trade imbalance 10 s - 15 min, realized
  volatility, spread, book imbalance, position vs open / previous close / VWAP / range) and, from the
  5-minute state rebuild (``feature_intraday.rebuild``), each price-like EMA as log(price / EMA), each
  range EMA over price, the return EMAs, log(day high / price), log(price / day low),
  log(snapshot high / price), PM's 10 s grid return and the cumulative amount imbalance;
* **label**: the next 5-minute mid log return, demeaned across stocks at each mark;
* **models**: ridge on per-mark ranks (numpy, always) and LightGBM when importable;
* **validation**: leave-one-day-out -- each day is predicted by a model fitted on the other days;
  per day the mean cross-sectional Spearman IC, and overall mean / t over all marks;
* **long-short**: at each mark the top / bottom ``--q`` of stocks by prediction, held 5 minutes,
  mean return per side and per round trip after ``--cost-bp`` per side (a signal check, not the T0
  execution of chrono).

    python research/pm_factors/own_model.py --dates 20260922,20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader
import dcp_ema as E  # tick loading and per-update arrays
import feature_intraday as FI  # the 5-minute state rebuild
import ic_score as IC  # marks, horizon, generic tick quantities, IC helpers

PRICE_EMA = ("zzpk.p", "zzab2.r", "zzup.pf", "zzxu.p", "zzup.ps", "zzeb.p", "zzba.p", "zzqj.r")
RANGE_EMA = ("zzxu.a", "zzvx.trEmaD", "zzba.r")
RET_EMA = ("zzbd.rEmaS", "zzbd.rEmaT")
LOG_EMA = ("zzbd.pEmaS", "zzbd.pEmaT")


def state_features(built: dict[str, np.ndarray], px: np.ndarray) -> dict[str, np.ndarray]:
    """Scale-free features from one stock's rebuilt state at the marks."""
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for p in PRICE_EMA:
            if f"{p}.sum" in built:
                out[f"st.lp_{p}"] = np.log(px / built[f"{p}.sum"])
        for p in RANGE_EMA:
            if f"{p}.sum" in built:
                out[f"st.rg_{p}"] = built[f"{p}.sum"] / px
        for p in RET_EMA:
            if f"{p}.sum" in built:
                out[f"st.{p}"] = built[f"{p}.sum"]
        for p in LOG_EMA:
            if f"{p}.sum" in built:
                out[f"st.lp_{p}"] = np.log(px) - built[f"{p}.sum"]
        if "zzpk.high" in built:
            out["st.dd_high"] = np.log(built["zzpk.high"] / px)
            out["st.du_low"] = np.log(px / built["zzpk.low"])
            out["st.lp_open"] = np.log(px / built["zzpk.open"])
        if "zzmh.hm" in built:
            out["st.dd_hm"] = np.log(built["zzmh.hm"] / px)
        if "zzxo.r.0" in built:
            out["st.grid_ret"] = np.log(built["zzxo.r.29"] / built["zzxo.r.0"])
        if "zzgh.nb" in built:
            out["st.cum_imb"] = built["zzgh.nb"]
    return out


def build_day(con: duckdb.DuckDBPyConnection, d: str, dcp: Path) -> pl.DataFrame | None:
    """One row per (mark, stock): features and the forward 5-minute mid return."""
    f = dcp / f"{d}.jz"
    if not f.exists():
        print(f"skip {d}: no state file (it fixes the universe and PM's stop)")
        return None
    pm = B.load_state(f)
    tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
    stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
    sids = sorted(pm["sid"].unique().to_list())
    up = E.load(con, int(d), sids, stop)
    if up["slow"].is_empty():
        print(f"skip {d}: no decoded ticks")
        return None
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    rows = []
    for sid in sids:
        gf, gs = fast.get(sid), slow.get(sid)
        if gf is None or gf.height < 3:
            continue
        a = E.arrays(gf, "fast")
        ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
        m0 = FI.at_marks(a["t_local"][ok], a["mid"][ok], IC.MARKS)
        m1 = FI.at_marks(a["t_local"][ok], a["mid"][ok], IC.MARKS + IC.HORIZON)
        tb = a["t_local"]
        okb = np.isfinite(a["bid1"]) & np.isfinite(a["ask1"])
        quotes = {}  # executable prices now and after each holding period (x_ prefix: not a feature)
        for h in (0, 300, 900):
            quotes[f"x_bid{h}"] = FI.at_marks(tb[okb], a["bid1"][okb], IC.MARKS + h)
            quotes[f"x_ask{h}"] = FI.at_marks(tb[okb], a["ask1"][okb], IC.MARKS + h)
            quotes[f"x_mid{h}"] = FI.at_marks(tb[ok], a["mid"][ok], IC.MARKS + h)
        px = FI.at_marks(a["t_local"][ok], a["mid"][ok], IC.MARKS)
        feats = {f"gen.{k}": v for k, v in IC.generic(gf, gs).items()}
        feats.update(state_features(FI.rebuild(gf, gs, IC.MARKS), px))
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(m1 / m0)
        rows.append(pl.DataFrame({"date": int(d), "mark": np.arange(IC.MARKS.size), "sid": sid, "y": y, **quotes,
                                  **{k: np.asarray(v, float) for k, v in feats.items()}}))
    return pl.concat(rows, how="diagonal_relaxed") if rows else None


def rank_by_mark(x: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Per-mark ranks scaled to [-0.5, 0.5]; NaN -> 0 (the middle)."""
    out = np.zeros_like(x)
    for g in np.unique(grp):
        m = grp == g
        for j in range(x.shape[1]):
            v = x[m, j]
            f = np.isfinite(v)
            r = np.zeros(v.size)
            if f.sum() > 1:
                r[f] = np.argsort(np.argsort(v[f])) / (f.sum() - 1) - 0.5
            out[np.flatnonzero(m), j] = r
    return out


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Ridge coefficients (with intercept) by the normal equations."""
    Xa = np.column_stack([np.ones(X.shape[0]), X])
    reg = lam * np.eye(Xa.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(Xa.T @ Xa + reg, Xa.T @ y)


def t0_backtest(panel: pl.DataFrame, preds: dict[str, np.ndarray], grp: np.ndarray, comm: float, stamp: float) -> list[dict]:
    """Round trips on the strongest predictions: buy-then-sell on the top, sell-base-then-buy-back on the bottom."""
    out = []
    q = {k: panel[k].to_numpy() for k in panel.columns if k.startswith("x_")}
    fee = 2 * comm + stamp
    for name, p in preds.items():
        rank = np.full(p.size, np.nan)
        for g in np.unique(grp):
            m = np.flatnonzero((grp == g) & np.isfinite(p))
            if m.size > 1:
                rank[m] = np.argsort(np.argsort(p[m])) / (m.size - 1)
        for h in (300, 900):
            with np.errstate(divide="ignore", invalid="ignore"):
                long_taker = (q[f"x_bid{h}"] / q["x_ask0"] - 1) * 1e4  # buy at the ask now, sell at the bid later
                short_taker = (q["x_bid0"] / q[f"x_ask{h}"] - 1) * 1e4  # sell base at the bid now, buy back at the ask later
                long_mid = (q[f"x_mid{h}"] / q["x_mid0"] - 1) * 1e4
                short_mid = -long_mid
            for top in (0.01, 0.02, 0.05, 0.1):
                lo, sh = rank >= 1 - top, rank <= top
                for fill, lr, sr in (("taker", long_taker, short_taker), ("mid", long_mid, short_mid)):
                    trades = np.concatenate([lr[lo], sr[sh]])
                    trades = trades[np.isfinite(trades)]
                    gross = float(trades.mean()) if trades.size else np.nan
                    out.append({"model": name, "hold_s": h, "top": top, "fill": fill, "trades": int(trades.size),
                                "gross_bp": gross, "net_bp": gross - fee, "win": float(np.mean(trades > fee)) if trades.size else np.nan,
                                "long_bp": float(np.nanmean(lr[lo])), "short_bp": float(np.nanmean(sr[sh]))})
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="")
    ap.add_argument("--from-panel", type=Path, help="reuse panel.parquet from an earlier run (must hold the x_ quote columns)")
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--ridge", type=float, default=50.0)
    ap.add_argument("--q", type=float, default=0.1, help="top / bottom share of stocks for the long-short check")
    ap.add_argument("--cost-bp", type=float, default=1.5, help="cost per side in bp for the decile check (rough)")
    ap.add_argument("--commission-bp", type=float, default=1.0, help="T0 backtest: commission per side, bp")
    ap.add_argument("--stamp-bp", type=float, default=5.0, help="T0 backtest: stamp duty on the sell leg, bp")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage7"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    days = []
    if args.from_panel:  # reuse a saved panel (features + labels + quotes) instead of rebuilding from ticks
        days = [pl.read_parquet(args.from_panel)]
        print(f"panel from {args.from_panel}: {days[0].height} rows")
    for d in [] if args.from_panel else [x.strip() for x in args.dates.split(",") if x.strip()]:
        df = build_day(con, d, args.dcp_dir)
        if df is not None:
            days.append(df)
            print(f"{d}: {df.height} rows, {df.width - 4} features ({time.perf_counter() - t0:.0f}s)")
    if not days or (len(days) < 2 and not args.from_panel):
        print("need at least two days")
        return
    panel = pl.concat(days, how="diagonal_relaxed").filter(pl.col("y").is_finite())
    panel = panel.with_columns((pl.col("y") - pl.col("y").mean().over("date", "mark")).alias("y"))
    panel.write_parquet(args.out / "panel.parquet")
    feats = [c for c in panel.columns if c.startswith(("gen.", "st."))]
    feats = [c for c in feats if panel[c].is_finite().mean() > 0.5]
    X = panel.select(feats).to_numpy().astype(float)
    X[~np.isfinite(X)] = np.nan
    y = panel["y"].to_numpy()
    date = panel["date"].to_numpy()
    grp = date * 100 + panel["mark"].to_numpy()
    R = rank_by_mark(X, grp)
    print(f"panel: {panel.height} rows, {len(feats)} features ({time.perf_counter() - t0:.0f}s)")
    try:
        import lightgbm as lgb  # optional
    except ImportError:
        lgb = None
        print("lightgbm not importable: ridge only")
    preds = {"ridge": np.full(y.size, np.nan), **({"lgbm": np.full(y.size, np.nan)} if lgb else {})}
    imp = np.zeros(len(feats))
    for d in np.unique(date):
        tr, te = date != d, date == d
        w = fit_ridge(R[tr], y[tr], args.ridge)
        preds["ridge"][te] = np.column_stack([np.ones(te.sum()), R[te]]) @ w
        if lgb:
            m = lgb.train({"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
                           "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1},
                          lgb.Dataset(X[tr], y[tr]), 300)
            preds["lgbm"][te] = m.predict(X[te])
            imp += m.feature_importance("gain")
    rows = []
    for name, p in preds.items():
        for d in np.unique(date):
            m = date == d
            ic = IC.summary(IC.ic_vec(p[m], y[m], grp[m]))
            ls = []
            for g in np.unique(grp[m]):
                k = grp == g
                o = np.argsort(p[k])
                n = max(1, int(args.q * k.sum()))
                ls.append((y[k][o[-n:]].mean() - y[k][o[:n]].mean()) / 2)  # per side, demeaned returns
            ls = np.array(ls) * 1e4
            rows.append({"model": name, "date": int(d), "ic": ic[0], "ic_t": ic[1], "marks": ic[2],
                         "ls_bp_side": float(np.nanmean(ls)), "net_bp_side": float(np.nanmean(ls) - args.cost_bp)})
        tot = IC.summary(IC.ic_vec(p, y, grp))
        print(f"{name}: out-of-sample IC {tot[0]:.4f} (t {tot[1]:.2f}, {tot[2]} marks)")
    res = pl.DataFrame(rows)
    t0_rows = t0_backtest(panel, preds, grp, args.commission_bp, args.stamp_bp)
    res.write_csv(args.out / "own_model_days.csv")
    with pl.Config(tbl_rows=60, tbl_width_chars=160, float_precision=4):
        print("\n== leave-one-day-out, per day (ls_bp_side: top-minus-bottom decile 5-minute return per side, bp; net after --cost-bp)")
        print(res.sort("model", "date"))
        print(res.group_by("model").agg(pl.col("ic").mean(), pl.col("ls_bp_side").mean(), pl.col("net_bp_side").mean()))
        if lgb:
            fi = pl.DataFrame({"feature": feats, "gain": imp}).sort("gain", descending=True).with_columns(
                (pl.col("gain") / pl.col("gain").sum()).alias("share"))
            fi.write_csv(args.out / "own_model_importance.csv")
            print("\n== LightGBM gain share, top 20")
            print(fi.head(20))
        uni = []
        for j, c in enumerate(feats):
            ic = IC.summary(IC.ic_vec(X[:, j], y, grp))
            uni.append({"feature": c, "ic": ic[0], "t": ic[1]})
        print("\n== single-feature IC (in sample, all days), strongest 20 by |t|")
        print(pl.DataFrame(uni).with_columns(pl.col("t").abs().fill_nan(None).alias("abs_t")).sort("abs_t", descending=True, nulls_last=True).head(20).drop("abs_t"))
        t0r = pl.DataFrame(t0_rows)
        t0r.write_csv(args.out / "own_model_t0.csv")
        print(f"\n== T0 round trips, out of sample (taker: buy at ask, sell at bid; mid: no spread; costs: commission {args.commission_bp} bp"
              f" per side + stamp {args.stamp_bp} bp on the sell; bp per round trip, raw returns, market included)")
        print(t0r.sort("model", "hold_s", "top"))
    print(f"tables: {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
