r"""Stage 7: our own tk5m-style model on the reproduced state and tick features, scored out of sample.

Not PM's formulas one by one: the reproduced state rules plus generic tick quantities, turned into
scale-free features at every 5-minute mark, a model fitted on some days and scored on the others.

* **features** per (day, mark, stock): the order-flow quantities of ``tick_flow`` (ticket-size buckets, new orders,
  cancels, price impact, queue refill), every generic tick quantity of ``ic_score.generic``
  (returns 5 s - 30 min of mid / microprice / last, trade imbalance 10 s - 15 min, realized
  volatility, spread, book imbalance, position vs open / previous close / VWAP / range) and, from the
  5-minute state rebuild (``feature_intraday.rebuild``), each price-like EMA as log(price / EMA), each
  range EMA over price, the return EMAs, log(day high / price), log(price / day low),
  log(snapshot high / price), PM's 10 s grid return and the cumulative amount imbalance;
* **label**: the next 5-minute mid log return (``--horizon`` 900 / 1800 / 3600 s: from the saved quotes),
  demeaned across stocks at each mark; rows with a move over 300 bp are listed first (a data check);
* **models**: ridge on per-mark ranks (numpy, always) and LightGBM when importable;
* **validation**: leave-one-day-out (``--cv lodo``) -- each day is predicted by a model fitted on the
  other days -- or walk-forward (``--cv walk``) -- only on the days before it; per day the mean
  cross-sectional Spearman IC, and overall mean / t over all marks;
* **universe**: PM's HS300 state file of the day fixes the stocks and the tick cut-off; a day without
  one takes the stocks of the nearest earlier file and stops at 15:02:30;
* **long-short**: at each mark the top / bottom ``--q`` of stocks by prediction, held 5 minutes,
  mean return per side and per round trip after ``--cost-bp`` per side (a signal check, not the T0
  execution of chrono).

    python research/pm_factors/own_model.py --dates 20260922,20260923,20260924,20260929,20260930,20261009
    python research/pm_factors/own_model.py --dates all --out data/reports/stage7/all
    python research/pm_factors/own_model.py --dates all --step 30 --workers 8 --cv walk --drop-dates 20260429,20260430 --out data/reports/stage7/dense
    python research/pm_factors/own_model.py --from-panel data/reports/stage7/all/panel.parquet --cv walk --out data/reports/stage7/walk
    python research/pm_factors/own_model.py --from-panel data/reports/stage7/all/panel.parquet --cv walk --drop-dates 20260429,20260430 --horizon 900
"""

from __future__ import annotations

import argparse
import itertools
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
import tick_flow as TF  # order-flow features

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


DEFAULT_STOP = 54150.005  # 15:02:30 local, PM's usual stop, for days without a state file
LGB_PARAMS = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
              "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1}
LGB_ROUNDS = 300


def universe(dcp: Path, d: str) -> tuple[list[int], float, str]:
    """Stocks and tick cut-off of one day: from its PM state file, else the nearest earlier (later) one with the default stop."""
    f = dcp / f"{d}.jz"
    if f.exists():
        pm = B.load_state(f)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        return sorted(pm["sid"].unique().to_list()), (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005, f.name
    files = sorted(p for p in dcp.glob("20*.jz") if p.stem.isdigit())
    if not files:
        return [], DEFAULT_STOP, ""
    near = max((p for p in files if p.stem <= d), default=files[0], key=lambda p: p.stem)
    return sorted(B.load_state(near)["sid"].unique().to_list()), DEFAULT_STOP, f"{near.name} (universe only)"


def build_day(con: duckdb.DuckDBPyConnection, d: str, dcp: Path, step: float = 300.0) -> pl.DataFrame | None:
    """One row per (mark, stock): features and the forward 5-minute mid return; marks every step seconds from 09:35."""
    marks = np.arange(300.0, 13801.0, step)
    sids, stop, src = universe(dcp, d)
    if not sids:
        print(f"skip {d}: no state file under {dcp} to fix the universe")
        return None
    print(f"{d}: {len(sids)} stocks from {src}, ticks up to {stop:.0f} s")
    up = E.load(con, int(d), sids, stop)
    if up["slow"].is_empty():
        print(f"skip {d}: no decoded ticks")
        return None
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    orders = {int(g["sid"][0]): g for g in TF.load_orders(con, int(d), sids, stop).partition_by("sid")}
    rows = []
    for sid in sids:
        gf, gs, go = fast.get(sid), slow.get(sid), orders.get(sid)
        if gf is None or gf.height < 3:
            continue
        a = E.arrays(gf, "fast")
        ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
        m0 = FI.at_marks(a["t_local"][ok], a["mid"][ok], marks)
        m1 = FI.at_marks(a["t_local"][ok], a["mid"][ok], marks + IC.HORIZON)
        tb = a["t_local"]
        okb = np.isfinite(a["bid1"]) & np.isfinite(a["ask1"])
        quotes = {}  # executable prices now and after each holding period (x_ prefix: not a feature)
        for h in (0, *HOLDS):
            quotes[f"x_bid{h}"] = FI.at_marks(tb[okb], a["bid1"][okb], marks + h)
            quotes[f"x_ask{h}"] = FI.at_marks(tb[okb], a["ask1"][okb], marks + h)
            quotes[f"x_mid{h}"] = FI.at_marks(tb[ok], a["mid"][ok], marks + h)
        px = FI.at_marks(a["t_local"][ok], a["mid"][ok], marks)
        feats = {f"gen.{k}": v for k, v in IC.generic(gf, gs, marks).items()}
        feats.update({f"flow.{k}": v for k, v in TF.flow(go, gs, gf, marks).items()})
        feats.update(state_features(FI.rebuild(gf, gs, marks), px))
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(m1 / m0)
        rows.append(pl.DataFrame({"date": int(d), "mark": np.arange(marks.size), "t": marks, "sid": sid, "y": y, **quotes,
                                  **{k: np.asarray(v, np.float32) for k, v in feats.items()}}))
    return pl.concat(rows, how="diagonal_relaxed") if rows else None


def build_job(job: tuple) -> tuple[str, pl.DataFrame | None]:
    """build_day in a worker process (its own catalog connection)."""
    d, dcp, step = job
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    try:
        return d, build_day(con, d, dcp, step)
    finally:
        con.close()


def rank_by_mark(x: np.ndarray, grp: np.ndarray) -> np.ndarray:
    """Per-mark ranks scaled to [-0.5, 0.5]; NaN -> 0 (the middle)."""
    out = np.zeros(x.shape, np.float32)
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


HOLDS = (300, 900, 1800, 3600)  # label horizons and holding periods, s of trading clock
END = 14220  # 14:57 on the trading clock: a window ending later would run into the close auction


def round_trips(panel: pl.DataFrame, h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Per row, bp: buy at the ask now and sell at the bid after h, sell base at the bid and buy back at the ask, and the mid return."""
    if f"x_mid{h}" not in panel.columns:
        return None
    q = {k: panel[k].to_numpy() for k in ("x_bid0", "x_ask0", "x_mid0", f"x_bid{h}", f"x_ask{h}", f"x_mid{h}")}
    late = panel["t"].to_numpy() + h > END
    with np.errstate(divide="ignore", invalid="ignore"):
        lt = (q[f"x_bid{h}"] / q["x_ask0"] - 1) * 1e4
        st = (q["x_bid0"] / q[f"x_ask{h}"] - 1) * 1e4
        lm = (q[f"x_mid{h}"] / q["x_mid0"] - 1) * 1e4
    return tuple(np.where(late, np.nan, v) for v in (lt, st, lm))


def t0_backtest(panel: pl.DataFrame, preds: dict[str, np.ndarray], grp: np.ndarray, comm: float, stamp: float) -> list[dict]:
    """Round trips on the strongest predictions: buy-then-sell on the top, sell-base-then-buy-back on the bottom."""
    out = []
    fee = 2 * comm + stamp
    for name, p in preds.items():
        rank = np.full(p.size, np.nan)
        for g in np.unique(grp):
            m = np.flatnonzero((grp == g) & np.isfinite(p))
            if m.size > 1:
                rank[m] = np.argsort(np.argsort(p[m])) / (m.size - 1)
        for h in HOLDS:
            if (rt := round_trips(panel, h)) is None:
                continue
            long_taker, short_taker, long_mid = rt
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


SPREAD_BINS = (0.0, 3.0, 6.0, 10.0, 20.0, np.inf)  # quoted spread at entry, bp of mid


def spread_backtest(panel: pl.DataFrame, preds: dict[str, np.ndarray], grp: np.ndarray, comm: float, stamp: float) -> list[dict]:
    """Taker round trips by quoted spread at entry: the top / bottom 5% by prediction, and only trades the prediction pays for."""
    out = []
    q = {k: panel[k].to_numpy() for k in panel.columns if k.startswith("x_")}
    fee = 2 * comm + stamp
    with np.errstate(divide="ignore", invalid="ignore"):
        spr = (q["x_ask0"] - q["x_bid0"]) / q["x_mid0"] * 1e4
    for name, p in preds.items():
        rank = np.full(p.size, np.nan)
        for g in np.unique(grp):
            m = np.flatnonzero((grp == g) & np.isfinite(p))
            if m.size > 1:
                rank[m] = np.argsort(np.argsort(p[m])) / (m.size - 1)
        for h in HOLDS:
            if (rt := round_trips(panel, h)) is None:
                continue
            lt, st, lm = rt
            # pays: the predicted move (y units -> bp) clears the spread plus the fixed costs
            rules = {"top5%": (rank >= 0.95, rank <= 0.05), "pays": (p * 1e4 > spr + fee, -p * 1e4 > spr + fee)}
            for rule, (lo, sh) in rules.items():
                for b0, b1 in itertools.pairwise(SPREAD_BINS):
                    inb = (spr > b0) & (spr <= b1)
                    tk = np.concatenate([lt[lo & inb], st[sh & inb]])
                    md = np.concatenate([lm[lo & inb], -lm[sh & inb]])
                    tk, md = tk[np.isfinite(tk)], md[np.isfinite(md)]
                    out.append({"model": name, "hold_s": h, "rule": rule, "spread_bp": f"{b0:g}-{b1:g}", "trades": int(tk.size),
                                "share_universe": float(np.mean(inb[np.isfinite(spr)])),
                                "mid_gross_bp": float(md.mean()) if md.size else np.nan,
                                "taker_net_bp": float(tk.mean() - fee) if tk.size else np.nan,
                                "win": float(np.mean(tk > fee)) if tk.size else np.nan})
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="", help="comma list, or 'all' = every day in raw_quotation")
    ap.add_argument("--cv", choices=("lodo", "walk"), default="lodo",
                    help="lodo: each day predicted from all other days; walk: only from earlier days (needs --min-train)")
    ap.add_argument("--min-train", type=int, default=5, help="walk: days needed before the first predicted day")
    ap.add_argument("--from-panel", type=Path, help="reuse panel.parquet from an earlier run (must hold the x_ quote columns)")
    ap.add_argument("--drop-dates", default="", help="comma list of days left out of the model (after the label check)")
    ap.add_argument("--horizon", type=int, choices=HOLDS, default=300,
                    help="label: mid return over the next h seconds; above 300 it comes from the panel's x_mid<h>, only marks whose window ends by 14:57")
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--step", type=float, default=300.0, help="seconds between marks (30 = a dense panel, ~10x the rows)")
    ap.add_argument("--workers", type=int, default=1, help="days built in parallel")
    ap.add_argument("--threads", type=int, default=8, help="LightGBM threads")
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
        if "t" not in days[0].columns:  # panels saved before the dense grid: 5-minute marks
            days[0] = days[0].with_columns(pl.Series("t", IC.MARKS[days[0]["mark"].to_numpy()]))
        print(f"panel from {args.from_panel}: {days[0].height} rows")
    dates = [x.strip() for x in args.dates.split(",") if x.strip()]
    if dates == ["all"]:
        dates = [str(r[0]) for r in con.sql("SELECT DISTINCT date FROM raw_quotation ORDER BY 1").fetchall()]
        print(f"{len(dates)} days in raw_quotation: {dates[0]} .. {dates[-1]}")
    jobs = [(d, args.dcp_dir, args.step) for d in ([] if args.from_panel else dates)]
    if args.workers > 1 and jobs:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(args.workers) as ex:
            built = list(ex.map(build_job, jobs))
    else:
        built = [(d, build_day(con, d, args.dcp_dir, args.step)) for d, _, _ in jobs]
    for d, df in built:
        if df is not None:
            days.append(df)
            print(f"{d}: {df.height} rows, {df.width - 5} features ({time.perf_counter() - t0:.0f}s)")
    if not days or (len(days) < 2 and not args.from_panel):
        print("need at least two days")
        return
    panel = pl.concat(days, how="diagonal_relaxed")
    if args.horizon != 300:  # 15-minute label from the saved quotes, windows that end inside the continuous session
        panel = panel.filter(pl.col("t") + args.horizon <= END).with_columns((pl.col(f"x_mid{args.horizon}") / pl.col("x_mid0")).log().alias("y"))
    panel = panel.filter(pl.col("y").is_finite())
    big = panel.filter(pl.col("y").abs() > 0.03)  # a 300 bp move of an HS300 mid within the horizon: rare, or a data fault
    print(f"label check (horizon {args.horizon} s): rows with |y| > 300 bp per day")
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(big.group_by("date").agg(pl.len().alias("n"), (pl.col("y").abs().max() * 1e4).alias("max_bp")).sort("date"))
        print(big.sort(pl.col("y").abs(), descending=True).select("date", "sid", "mark", "y", "x_bid0", "x_ask0", "x_mid0",
                                                                  f"x_mid{args.horizon}").head(8))
    drop = [int(x) for x in args.drop_dates.split(",") if x.strip()]
    if drop:
        panel = panel.filter(~pl.col("date").is_in(drop))
        print(f"dropped days {drop}: {panel.height} rows left")
    panel = panel.with_columns((pl.col("y") - pl.col("y").mean().over("date", "mark")).alias("y"))
    if not args.from_panel:
        panel.write_parquet(args.out / "panel.parquet")
    feats = [c for c in panel.columns if c.startswith(("gen.", "st.", "flow."))]
    feats = [c for c in feats if panel[c].is_finite().mean() > 0.5]
    X = panel.select(feats).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    y = panel["y"].to_numpy()
    date = panel["date"].to_numpy()
    grp = date.astype(np.int64) * 100000 + panel["mark"].to_numpy()
    R = rank_by_mark(X, grp)
    print(f"panel: {panel.height} rows, {len(feats)} features ({time.perf_counter() - t0:.0f}s)")
    try:
        import lightgbm as lgb  # optional
    except ImportError:
        lgb = None
        print("lightgbm not importable: ridge only")
    preds = {"ridge": np.full(y.size, np.nan), **({"lgbm": np.full(y.size, np.nan)} if lgb else {})}
    imp = np.zeros(len(feats))
    udays = np.unique(date)
    print(f"cv {args.cv}: {udays.size} days" + (f", first {args.min_train} only train" if args.cv == "walk" else ""))
    for i, d in enumerate(udays):
        if args.cv == "walk" and i < args.min_train:
            continue
        tr, te = (date < d) if args.cv == "walk" else (date != d), date == d
        w = fit_ridge(R[tr], y[tr], args.ridge)
        preds["ridge"][te] = np.column_stack([np.ones(te.sum()), R[te]]) @ w
        if lgb:
            m = lgb.train({**LGB_PARAMS, "num_threads": args.threads}, lgb.Dataset(X[tr], y[tr]), LGB_ROUNDS)
            preds["lgbm"][te] = m.predict(X[te])
            imp += m.feature_importance("gain")
    rows = []
    for name, p in preds.items():
        for d in np.unique(date):
            m = date == d
            if not np.isfinite(p[m]).any():  # walk: a training-only day
                continue
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
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(f"\n== out of sample ({args.cv}, horizon {args.horizon} s), per day (ls_bp_side: top-minus-bottom decile return per side, bp;"
              " net after --cost-bp)")
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
        sp = pl.DataFrame(spread_backtest(panel, preds, grp, args.commission_bp, args.stamp_bp))
        sp.write_csv(args.out / "own_model_spread.csv")
        print("\n== taker round trips by quoted spread at entry (top5%: strongest 5% each side; pays: |prediction| > spread + fixed"
              " costs; mid_gross: before spread and costs; taker_net: after both)")
        with pl.Config(tbl_rows=60, tbl_cols=9, tbl_width_chars=160, float_precision=2):
            print(sp.sort("model", "hold_s", "rule"))
    print(f"tables: {args.out} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
