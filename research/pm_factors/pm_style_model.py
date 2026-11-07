r"""Stage 9: our own model built on PM's ideas -- every snapshot is a decision point, features by PM's categories.

Not PM's formulas column by column (most are not reproducible without PM's feature values): the quantities we can
compute exactly, grouped by the kinds of feature PM's 209 columns turned out to be, at a 1-in-``--keep`` sample of
each stock's own continuous-session snapshots (local arrival), over the CSI 300 + CSI 500 + CSI 1000 universe.

``build`` writes one panel per day (``<out>/panel/date=D.parquet``), one row per sampled (stock, snapshot):

* returns / reversal (ro, zl, jx): mid / microprice / last-price returns 5 s - 30 min, position vs open / previous close /
  VWAP / window mean (``ic_score.generic``);
* market (zl, rg, ro): CSI 300 / 500 / 1000 and the own exchange composite -- index returns, the stock's excess over
  them, both also vs the previous close (``ic_score.index_relative``);
* book (eu, yg, pe): level-1 / 5-level / whole-book imbalance, relative spread, one-tick share;
* trade flow (pe, ef, ie, fe, ho, op): amount / volume / count imbalance 10 s - 15 min, the day's cumulative imbalance,
  log amount, trade size, window share of the day;
* order flow and size (ug, vc): ``tick_flow`` -- ticket-size buckets, new orders, cancels, Kyle lambda, queue refill;
* volatility / range (vx, xu, rg, sa): realized volatility, window range, distance to the window / day high and low;
* EMA state (up, ab, pk, bd, eb, ba): the engine rules reproduced exactly, as log(price / EMA) and range / price;
* activity vs normal (wa, al, qo, xc, ts, ms -- CSI 300 only): the last 5 minutes' amount / volume / buy amount and the day's
  volume over PM's own samplerS median for that slot, taken from PM's file of the previous day (today's file includes
  today);
* daily context (CED): Barra exposures, volatility, 1 / 5 / 20-day returns, turnover, intraday volume over its 20-day mean;
* ``uni`` = 0 / 1 / 2 for CSI 300 / 500 / 1000.

Labels: ``y_raw`` = the next 5-minute mid log return, ``y_ex`` = that minus the stock's index over the same 5 minutes
(snapshots are not synchronous across stocks, so "demeaned" is relative to the index). ``x_*`` columns are quotes for
cost checks, not features.

``fit`` trains LightGBM walk-forward with PM's hyper-parameters (read from PM's model file), one model per label, and
reports the out-of-sample rank IC per 1-minute bucket (mean, t, per day, per universe), the gain by category and, with
``--ablate``, the IC lost when a whole category is dropped (on the last fold).

    python research/pm_factors/pm_style_model.py build --dates 20260901-20261105 --workers 8 --out data/reports/stage9
    python research/pm_factors/pm_style_model.py fit --out data/reports/stage9 --threads 16 --ablate
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog
import dcp_ema as E  # tick loading
import feature_intraday as FI  # state rebuild, at_marks
import ic_score as IC  # generic tick quantities, index series
import own_model as OM  # state features, default stop
import tick_flow as TF  # order flow

UNI = {"000300": 0, "000905": 1, "000852": 2}  # CSI 300 / 500 / 1000
UNI_INDEX = {0: "hs300", 1: "zz500", 2: "zz1000"}
T_FIRST, T_LAST = 60.0, 13920.0  # trading clock: from 09:31, and the 5-minute label still ends by 14:57
HORIZON = 300.0
PROFILE = {"ZZXC": ("n", "q"), "ZZTS": ("q",), "ZZMS": ("q",)}  # PM samplerS medians used as activity norms
CHUNK = 300  # stocks per tick load


def day_universe(d: str) -> dict[int, int] | None:
    """sid -> 0 / 1 / 2 (CSI 300 / 500 / 1000) from CED's index weights of the day; None when CED lacks it."""
    from sampler_r import ced_universe

    out: dict[int, int] = {}
    for code, u in UNI.items():
        w = ced_universe(code, d)
        if w is None:
            return None
        for s in w["symbol_id"].to_list():
            out.setdefault(int(s), u)
    return out


def prev_profiles(hstats: Path, cache: Path, d: str) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """sid -> '<FAM>.<V>' -> (bar minutes, PM samplerS values) from PM's latest file before d (no look-ahead)."""
    from same_gap import day_frame

    out: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for fam, vs in PROFILE.items():
        days = sorted(p.name for p in (hstats / fam).glob("20*") if p.name < d and (p / "samplerS.nc").exists()) if (hstats / fam).exists() else []
        if not days:
            continue
        df = day_frame(hstats, fam, days[-1], cache)
        if df is None:
            continue
        for (code, v), g in df.filter(pl.col("V").is_in(list(vs))).sort("minute").group_by(["code", "V"], maintain_order=True):
            out.setdefault(int(code), {})[f"{fam}.{v}"] = (g["minute"].to_numpy().astype(int), g["x"].to_numpy().astype(float))
    return out


def norm_features(gen: dict[str, np.ndarray], prof: dict[str, tuple[np.ndarray, np.ndarray]], T: np.ndarray) -> dict[str, np.ndarray]:
    """log(last 5 minutes' flow / PM's slot median) and log(day volume / PM's cumulative median) at the slot of each T."""
    out: dict[str, np.ndarray] = {}
    mm = IC.mark_minutes(T)

    def cell(key: str) -> np.ndarray | None:
        if key not in prof:
            return None
        mins, vals = prof[key]
        k = np.searchsorted(mins, mm, side="right") - 1  # the last bar cell at or before the minute
        return np.where(k >= 0, vals[np.clip(k, 0, mins.size - 1)], np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        for name, flow, key in (("xc_amt", "amt300", "ZZXC.n"), ("xc_vol", "vol300", "ZZXC.q"), ("ts_buy", "buy300", "ZZTS.q")):
            c = cell(key)
            if c is not None and flow in gen:
                out[f"norm.{name}"] = np.log(gen[flow] / c)
        c = cell("ZZMS.q")
        if c is not None and "lvol300" in gen and "vfrac300" in gen:  # day volume so far = window volume / its share of the day
            day_vol = np.expm1(gen["lvol300"]) / np.exp(gen["vfrac300"])
            out["norm.ms_vol"] = np.log(day_vol / c)
    return out


def build_day(d: str, keep: int, hstats: Path, cache: Path) -> pl.DataFrame | None:
    """One day's panel: features and labels at a 1-in-keep sample of each stock's snapshots."""
    import ced_match as CM

    uni = day_universe(d)
    if not uni:
        print(f"skip {d}: no CED index weights")
        return None
    sids = sorted(uni)
    stop = OM.DEFAULT_STOP
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    try:
        ix = IC.index_series(con, int(d), stop)
        dc = CM.daily_candidates(d, sids)
        prof = prev_profiles(hstats, cache, d)
        daily = {}
        if dc is not None:
            cols = [c for c, typ in dc.schema.items() if c != "sid" and typ.is_numeric()]
            daily = {int(r[0]): r[1:] for r in dc.select("sid", *cols).iter_rows()}
        rows = []
        for i in range(0, len(sids), CHUNK):
            part = sids[i:i + CHUNK]
            up = E.load(con, int(d), part, stop)
            fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
            slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
            orders = {int(g["sid"][0]): g for g in TF.load_orders(con, int(d), part, stop).partition_by("sid")}
            for sid in part:
                gf, gs = fast.get(sid), slow.get(sid)
                if gf is None or gf.height < 50 or gs is None or gs.height < 10:
                    continue
                a = E.arrays(gf, "fast")
                ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
                t = a["t_local"][ok]
                cand = np.unique(t[(t >= T_FIRST) & (t <= T_LAST)])
                rng = np.random.default_rng(int(d) * 1_000_003 + sid)
                T = np.sort(cand[rng.random(cand.size) < 1.0 / keep])
                if T.size == 0:
                    continue
                mid = a["mid"][ok]
                with np.errstate(divide="ignore", invalid="ignore"):
                    y_raw = np.log(FI.at_marks(t, mid, T + HORIZON) / FI.at_marks(t, mid, T))
                u = uni[sid]
                y_ex = np.full(T.size, np.nan)
                if UNI_INDEX[u] in ix:
                    ti, li, _ = ix[UNI_INDEX[u]]
                    y_ex = y_raw - (FI.at_marks(ti, li, T + HORIZON) - FI.at_marks(ti, li, T))
                okb = np.isfinite(a["bid1"]) & np.isfinite(a["ask1"])
                quotes = {f"x_{s}{h}": FI.at_marks(a["t_local"][okb], a[f"{s}1"][okb], T + h)
                          for s in ("bid", "ask") for h in (0, int(HORIZON))}
                gen = IC.generic(gf, gs, T)
                gen.update(IC.index_relative(gen, ix, sid, T))
                feats = {f"gen.{k}": v for k, v in gen.items() if k not in ("amt300", "vol300", "buy300")}
                feats.update({f"flow.{k}": v for k, v in TF.flow(orders.get(sid), gs, gf, T).items()})
                feats.update(OM.state_features(FI.rebuild(gf, gs, T), FI.at_marks(t, mid, T)))
                if sid in prof:
                    feats.update(norm_features(gen, prof[sid], T))
                if sid in daily:
                    feats.update({f"ced.{c}": np.full(T.size, np.nan if v is None else float(v)) for c, v in zip(cols, daily[sid], strict=True)})
                rows.append(pl.DataFrame({"date": int(d), "sid": sid, "t": T, "uni": u, "y_raw": y_raw, "y_ex": y_ex,
                                          **{k: np.asarray(v, np.float64) for k, v in quotes.items()},
                                          **{k: np.asarray(v, np.float32) for k, v in feats.items()}}))
        return pl.concat(rows, how="diagonal_relaxed") if rows else None
    finally:
        con.close()


def _build_job(job: tuple) -> str:
    d, keep, hstats, cache, out = job
    t0 = time.perf_counter()
    df = build_day(d, keep, hstats, cache)
    if df is None:
        return f"{d}: skipped"
    out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out / f"date={d}.parquet")
    return f"{d}: {df.height:,} rows, {df['sid'].n_unique()} stocks, {len([c for c in df.columns if '.' in c])} features ({time.perf_counter() - t0:.0f}s)"


def trading_days(spec: str) -> list[str]:
    """'A-B' = the days with decoded snapshots in the catalog between A and B; else a comma list."""
    if "-" not in spec:
        return [x for x in spec.split(",") if x]
    a, b = spec.split("-")
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    try:
        days = [str(r[0]) for r in con.sql(f"SELECT DISTINCT date FROM raw_quotation WHERE date BETWEEN {a} AND {b} ORDER BY 1").fetchall()]
    finally:
        con.close()
    return days


CATEGORIES = [  # (category, regex on the feature name), first match wins
    ("market", r"^gen\.(ir|xr|ipc|xpc)\d*_?"),
    ("returns", r"^gen\.([mp]?ret\d+|lpma\d+|lp_open|lp_pc|lp_vwap|stoch|zr\d+)$"),
    ("book", r"^gen\.(imb1|imb5|timb|spread|spm\d+|tick1_\d+)$"),
    ("tradeflow", r"^gen\.(s?imb[vna]\d+|s?cum_imb|s?lvol\d+|s?lamt\d+|nam\d+|nn\d+|tsz\d+|vfrac\d+)$"),
    ("volrange", r"^gen\.(rv\d+|rng\d+|ldh\d+|ldl\d+)$"),
    ("orderflow", r"^flow\."),
    ("state", r"^st\."),
    ("norm", r"^norm\."),
    ("daily", r"^ced\."),
    ("universe", r"^uni$"),
]


def category(f: str) -> str:
    """PM-style category of a feature name."""
    return next((c for c, p in CATEGORIES if re.match(p, f)), "other")


def pm_params(models: Path, threads: int) -> tuple[dict, int]:
    """LightGBM parameters and rounds from PM's newest model file (the 'parameters:' block); defaults when unreadable."""
    files = sorted(models.glob("*/lgb/livev1_fit1/model_s1.txt"), key=lambda p: p.stat().st_mtime)
    params, rounds = dict(OM.LGB_PARAMS), OM.LGB_ROUNDS
    if files:
        text = files[-1].read_text(errors="replace")
        m = re.search(r"\nparameters:\n(.*?)\nend of parameters", text, re.S)
        kv = dict(re.findall(r"^\[(\w+): (.*)\]$", m.group(1), re.M)) if m else {}
        num = {"learning_rate": float, "num_leaves": int, "max_depth": int, "min_data_in_leaf": int, "min_sum_hessian_in_leaf": float,
               "feature_fraction": float, "bagging_fraction": float, "bagging_freq": int, "lambda_l1": float, "lambda_l2": float,
               "min_gain_to_split": float, "max_bin": int}
        for k, f in num.items():
            if k in kv:
                params[k] = f(kv[k])
        if kv.get("objective"):
            params["objective"] = kv["objective"].split()[0]
        rounds = text.count("\nTree=") or rounds
        print(f"PM parameters from {files[-1]}: {params}, {rounds} trees")
    params.update({"verbose": -1, "num_threads": threads})
    return params, rounds


def ic_buckets(pred: np.ndarray, y: np.ndarray, date: np.ndarray, t: np.ndarray, min_n: int = 30) -> pl.DataFrame:
    """Rank IC per (day, 1-minute bucket of the trading clock)."""
    df = pl.DataFrame({"date": date, "b": (t // 60).astype(np.int64), "p": pred, "y": y}).filter(pl.col("p").is_finite() & pl.col("y").is_finite())
    return (df.group_by("date", "b").agg(pl.len().alias("n"), pl.corr(pl.col("p").rank(), pl.col("y").rank()).alias("ic"))
            .filter((pl.col("n") >= min_n) & pl.col("ic").is_finite()))


def summarize(ic: pl.DataFrame) -> str:
    """Mean IC, t over buckets, buckets, share of days with a positive mean."""
    v = ic["ic"].to_numpy()
    if v.size < 2:
        return "-"
    per_day = ic.group_by("date").agg(pl.col("ic").mean())["ic"].to_numpy()
    return f"IC {v.mean():.4f} (t {v.mean() / v.std(ddof=1) * np.sqrt(v.size):.1f}, {v.size} buckets, days > 0 {np.mean(per_day > 0):.2f})"


def fit(a: argparse.Namespace) -> None:
    """Walk-forward LightGBM per label, out-of-sample IC, gain by category, optional category ablation.

    Training rows are a fixed random sample of every day (``--max-rows`` in all, kept in memory); a scored day is read
    whole, predicted and dropped, so the full panel never sits in memory.
    """
    import lightgbm as lgb
    import pyarrow.parquet as pq

    files = {f.stem.split("=")[1]: f for f in sorted((a.out / "panel").glob("date=*.parquet"))}
    days = sorted(files)
    if len(days) <= a.min_train:
        raise SystemExit(f"{len(days)} panel days, need more than --min-train {a.min_train}")
    cols = sorted(set().union(*(pq.read_schema(f).names for f in files.values())))
    feats = [c for c in cols if c not in ("date", "sid", "t", "y_raw", "y_ex") and not c.startswith("x_")]
    n_rows = {d: pq.read_metadata(f).num_rows for d, f in files.items()}
    frac = min(1.0, a.max_rows / sum(n_rows.values()))

    def read(d: str) -> pl.DataFrame:
        df = pl.read_parquet(files[d])
        df = df.with_columns([pl.lit(None, pl.Float32).alias(c) for c in feats if c not in df.columns])
        return df.filter(pl.col("uni") == int(a.uni)) if a.uni != "all" else df

    samp = []
    for i, d in enumerate(days):
        df = read(d)
        samp.append(df.sample(fraction=frac, seed=i) if frac < 1 else df)
    tr_all = pl.concat(samp, how="diagonal_relaxed")
    Xs = tr_all.select(feats).to_numpy().astype(np.float32)
    ds_date = tr_all["date"].to_numpy()
    ys = {lab: tr_all[lab].to_numpy() for lab in ("y_raw", "y_ex")}
    print(f"{len(days)} panel days, {sum(n_rows.values()):,} rows; training sample {len(ds_date):,} rows ({frac:.3f}); "
          f"{len(feats)} features; universe {a.uni}")
    params, rounds = pm_params(a.models, a.threads)
    if a.rounds:
        rounds = a.rounds
        print(f"rounds capped at {rounds} (--rounds)")
    test_days = days[a.min_train:]
    meta, gains = [], {lab: np.zeros(len(feats)) for lab in ys}
    models_last = {}
    for k in range(0, len(test_days), a.refit):
        block = test_days[k:k + a.refit]
        bst = {}
        for lab, y in ys.items():
            tr = np.flatnonzero((ds_date < int(block[0])) & np.isfinite(y))
            bst[lab] = lgb.train(params, lgb.Dataset(Xs[tr], y[tr], feature_name=feats), rounds)
            gains[lab] += bst[lab].feature_importance("gain")
            print(f"  {lab}: {tr.size:,} training rows before {block[0]}", flush=True)
        models_last = bst
        for d in block:
            df = read(d)
            X = df.select(feats).to_numpy().astype(np.float32)
            meta.append(df.select("date", "sid", "t", "uni", "y_raw", "y_ex", *[c for c in df.columns if c.startswith("x_")]).with_columns(
                *[pl.Series(f"pred_{lab[2:]}", bst[lab].predict(X, num_threads=a.threads)) for lab in ys]))
        print(f"  scored {block[0]}..{block[-1]}", flush=True)
    out = pl.concat(meta, how="diagonal_relaxed")
    date, t, uni = out["date"].to_numpy(), out["t"].to_numpy(), out["uni"].to_numpy()
    for lab in ys:
        pred = out[f"pred_{lab[2:]}"].to_numpy()
        for ylab in ys:
            yy = out[ylab].to_numpy()
            print(f"== model {lab} vs {ylab}: {summarize(ic_buckets(pred, yy, date, t))}")
            for u, name in UNI_INDEX.items():
                m = uni == u
                if m.any():
                    print(f"   {name}: {summarize(ic_buckets(pred[m], yy[m], date[m], t[m]))}")
        g = pl.DataFrame({"feature": feats, "gain": gains[lab]}).with_columns(
            pl.col("feature").map_elements(category, return_dtype=pl.String).alias("cat"))
        tot = max(g["gain"].sum(), 1e-12)
        print("   gain by category: " + ", ".join(f"{c} {s / tot:.3f}" for c, s in g.group_by("cat").agg(pl.col("gain").sum()).sort("gain", descending=True).iter_rows()))
        print("   top features: " + ", ".join(f"{f} {s / tot:.3f}" for f, s, _ in g.sort("gain", descending=True).head(12).iter_rows()))
    out.write_parquet(a.out / f"preds_{a.uni}.parquet")
    print(f"out-of-sample predictions -> {a.out / f'preds_{a.uni}.parquet'} ({out.height:,} rows)")
    if a.ablate:  # the last fold again without each category: the IC it carries beyond the others
        block = test_days[-a.refit:]
        last = pl.concat([read(d) for d in block], how="diagonal_relaxed")
        Xt = last.select(feats).to_numpy().astype(np.float32)
        for lab, y in ys.items():
            tr = np.flatnonzero((ds_date < int(block[0])) & np.isfinite(y))
            yt, dt, tt = last[lab].to_numpy(), last["date"].to_numpy(), last["t"].to_numpy()
            base = ic_buckets(models_last[lab].predict(Xt, num_threads=a.threads), yt, dt, tt)["ic"].mean()
            for c in sorted({category(f) for f in feats}):
                keep = [j for j, f in enumerate(feats) if category(f) != c]
                b2 = lgb.train(params, lgb.Dataset(Xs[tr][:, keep], y[tr], feature_name=[feats[j] for j in keep]), rounds)
                ic = ic_buckets(b2.predict(Xt[:, keep], num_threads=a.threads), yt, dt, tt)["ic"].mean()
                print(f"   ablate {lab} {c:10s}: IC {ic:.4f} ({ic - base:+.4f} vs all, last {len(block)} days)", flush=True)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="one panel per day")
    b.add_argument("--dates", required=True, help="A-B (days with snapshots in the catalog) or a comma list")
    b.add_argument("--keep", type=int, default=10, help="sample 1 in keep of each stock's snapshots")
    b.add_argument("--workers", type=int, default=8)
    b.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    b.add_argument("--pm-cache", type=Path, default=Path("data/pm_export_all"))
    b.add_argument("--out", type=Path, default=Path("data/reports/stage9"))
    b.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True, help="rebuild existing day panels (default)")
    f = sub.add_parser("fit", help="walk-forward LightGBM with PM's parameters")
    f.add_argument("--out", type=Path, default=Path("data/reports/stage9"))
    f.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    f.add_argument("--uni", default="all", help="all, or 0 / 1 / 2 = CSI 300 / 500 / 1000 only")
    f.add_argument("--min-train", type=int, default=15, help="days used only for training before the first scored day")
    f.add_argument("--refit", type=int, default=5, help="days scored per model (expanding window)")
    f.add_argument("--max-rows", type=int, default=4_000_000, help="training sample over all days (random, fixed)")
    f.add_argument("--threads", type=int, default=16)
    f.add_argument("--rounds", type=int, default=600, help="trees per model (0 = PM's own count, 3000 at lr 0.03: slow)")
    f.add_argument("--ablate", action="store_true", help="last fold: IC lost without each category")
    a = ap.parse_args()
    if a.cmd == "fit":
        fit(a)
        return
    days = trading_days(a.dates)
    todo = [d for d in days if a.overwrite or not (a.out / "panel" / f"date={d}.parquet").exists()]
    print(f"{len(days)} days, {len(todo)} to build with {a.workers} workers (1 in {a.keep} snapshots)")
    t0 = time.perf_counter()
    import multiprocessing as mp

    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_build_job, (d, a.keep, a.hstats, a.pm_cache, a.out / "panel")): d for d in todo}
        for fut in as_completed(futs):
            try:
                print(fut.result(), flush=True)
            except Exception as e:  # noqa: BLE001 -- one bad day must not stop the rest
                print(f"{futs[fut]}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    print(f"built in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
