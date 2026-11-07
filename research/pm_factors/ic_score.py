r"""Stage 6: score reproduced features through PM's own LightGBM models by IC, and scan candidates.

The model files are the one end-to-end check available: if a reproduced feature is right, PM's
model should predict the next 5-minute return better than with that feature missing. So:

1. **features**: at every 5-minute mark (09:35 ... 14:50, local arrival) the state is rebuilt from
   our ticks with ``feature_intraday.rebuild`` (the solved rules), and each model column (the 209
   names of ``lgb_fn_v1_209_cte.csv``, in order) is filled from an assignment ``feature ->
   expression`` (``feature_match.candidates`` of its stem); unassigned columns stay NaN, which
   LightGBM routes down its missing-value branches;
2. **model**: PM's ``model_s*.txt`` for the latest month not after the day, predicted by
   ``lightgbm`` when importable, else by the tree walker here (numerical and categorical splits,
   LightGBM's missing-value rules);
3. **IC**: per mark, the Spearman correlation across stocks between the prediction and the mid
   log return over the next 5 minutes; averaged over marks and days (``ic``) and its t-stat;
4. **scan**: features in order of model gain; for each, its ``--shortlist`` candidates closest to
   the fingerprint (``feature_match.rank``) are tried one at a time with the others fixed, and the
   best is kept when it raises IC by more than ``--min-gain``. ``--passes`` repeats the sweep.
   A stem without state draws from one shared ``Pool``: generic tick quantities, CED daily fields and
   intraday-over-daily ratios (``--ced``), each also z-scored / ranked across stocks within its mark
   (``--xs``). Those transforms share one CDF whatever they transform, so their bases are shortlisted by
   their own |IC| (``--xs-shortlist``) and never accepted by CDF alone.

State comes from the C++ engine's ``--marks`` output (``<engine-dir>/marks_<D>.parquet``: every solved rule at
every mark) where it exists, over ``feature_intraday.rebuild``; state candidates include (x - y) / z (``--triples``).
samplerS profile cells at each mark (``--profiles``: the cell at, before and after the mark's minute) and the last
5-minute bar's amount / volume / buy amount over them join the shared pool as ``ced.<FAM>.<V>@<off>``. A candidate
with fewer than ``--min-distinct`` values (a dummy) only matches a feature whose fingerprint is a flag / count.

The assignment found is written to ``ic_assign.json`` and can be fed back with ``--assign``. At the end every assigned
column is removed once (ablation: the paired IC loss and its t) and checked against the training CDF:
``ic_evidence.parquet`` grades each strong (CDF gap <= 0.02 and t >= 2) / fits (gap <= --accept-cdf) / weak;
``ic_log.parquet`` keeps the scan log (closest candidates of the features left open).

    python research/pm_factors/ic_score.py --dates 20260923,20260929,20261009
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader
import dcp_ema as E  # tick loading
import feature_intraday as FI  # the 5-minute rebuild
import feature_match as FM  # candidate expressions, fingerprint ranking
from same_gap import day_frame  # samplerS cells of one family and day (cached parquet)

MARKS = np.arange(300.0, 13801.0, 300.0)  # 09:35 ... 14:50: the last mark still has 5 minutes of continuous session ahead
HORIZON = 300.0


def mark_grid(spec: str) -> np.ndarray:
    """'five' = the 5-minute marks; 'rand:N[:seed]' = N random trading-clock seconds in (0, 13800) (the same list in
    ``run_cpp.py``): PM scores at every snapshot, so mid-bar times sample its training distribution; else a comma list."""
    if spec == "five":
        return np.arange(300.0, 13801.0, 300.0)
    if spec.startswith("rand:"):
        n, *seed = (int(x) for x in spec[5:].split(":"))
        return np.sort(np.random.default_rng(seed[0] if seed else 0).choice(np.arange(1, 13800), n, replace=False)).astype(float)
    return np.array([float(x) for x in spec.split(",")])
KNOWN = {"zzup": "px / zzup.pf"}  # starting assignment: the fingerprint's one clean match


def _arr(kv: dict[str, str], key: str, dtype: type = float) -> np.ndarray:
    """One whitespace-separated array field of a tree block (empty when absent)."""
    v = kv.get(key, "").strip()
    return np.array(v.split(), dtype=dtype) if v else np.array([], dtype=dtype)


class TreeModel:
    """A LightGBM text model, predicted without lightgbm."""

    def __init__(self, path: Path) -> None:
        text = path.read_text(errors="replace")
        self.trees = []
        for block in re.split(r"\nTree=\d+\n", text.split("\nend of trees")[0])[1:]:
            kv = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
            t = {"leaf": _arr(kv, "leaf_value"), "feat": _arr(kv, "split_feature", int), "thr": _arr(kv, "threshold"),
                 "dt": _arr(kv, "decision_type", int), "left": _arr(kv, "left_child", int), "right": _arr(kv, "right_child", int),
                 "cat_b": _arr(kv, "cat_boundaries", int), "cat_t": _arr(kv, "cat_threshold", np.int64)}
            self.trees.append(t)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sum of the trees' leaf values for each row of X (NaN = missing)."""
        out = np.zeros(X.shape[0])
        for t in self.trees:
            if t["feat"].size == 0:
                out += t["leaf"][0]
                continue
            node = np.zeros(X.shape[0], int)
            live = np.ones(X.shape[0], bool)
            while live.any():
                n = node[live]
                x = X[live, t["feat"][n]]
                dt = t["dt"][n]
                default_left = (dt >> 1) & 1
                missing = (dt >> 2) & 3  # 0 none, 1 zero, 2 nan
                isnan = np.isnan(x)
                xz = np.where(isnan & (missing != 2), 0.0, x)  # LightGBM turns NaN into 0 unless NaN is the missing type
                go_left = xz <= t["thr"][n]
                use_default = ((missing == 2) & isnan) | ((missing == 1) & (np.abs(xz) <= 1e-35))
                go_left = np.where(use_default, default_left == 1, go_left)
                cat = (dt & 1) == 1
                if cat.any():
                    ci = t["thr"][n[cat]].astype(int)
                    xi = np.where(np.isnan(x[cat]), -1, x[cat]).astype(int)
                    word = t["cat_b"][ci] + xi // 32
                    ok = (xi >= 0) & (word < t["cat_b"][ci + 1])
                    bit = np.where(ok, (t["cat_t"][np.where(ok, word, 0)] >> (xi % 32)) & 1, 0)
                    go_left[cat] = bit == 1
                nxt = np.where(go_left, t["left"][n], t["right"][n])
                idx = np.flatnonzero(live)
                leaf = nxt < 0
                out[idx[leaf]] += t["leaf"][~nxt[leaf]]
                node[idx[~leaf]] = nxt[~leaf]
                live[idx[leaf]] = False
        return out


def load_models(root: Path, date: str, fits: list[str], subs: list[str]) -> list:
    """PM's models for the latest month folder not after the day."""
    months = sorted(p.name for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name) and p.name <= date[:6])
    if not months:
        raise SystemExit(f"no model month <= {date[:6]} under {root}")
    files = [root / months[-1] / "lgb" / f / f"model_{s}.txt" for f in fits for s in subs]
    files = [f for f in files if f.exists()]
    try:
        from tree_kernel import FlatModel  # compiled walker; also sums just the trees on one feature

        return [FlatModel.from_file(f) for f in files]
    except ImportError:
        pass
    try:
        import lightgbm as lgb  # optional: much faster than the numpy tree walker
        return [lgb.Booster(model_file=str(f)) for f in files]
    except ImportError:
        return [TreeModel(f) for f in files]


def predict(models: list, X: np.ndarray) -> np.ndarray:
    """Mean prediction of the models."""
    return np.mean([m.predict(X) for m in models], axis=0)


def predict_rows(models: list, X: np.ndarray, rows: np.ndarray, feat: int | None = None) -> np.ndarray:
    """Mean raw score on X[rows]; with ``feat`` only the trees that split on that column (FlatModel only)."""
    if not hasattr(models[0], "trees_of"):
        return predict(models, X[rows])
    none = np.zeros(0, np.int64)
    return np.mean([m.predict(X, rows, None if feat is None else m.trees_of.get(feat, none)) for m in models], axis=0)


def win_reduce(fn: np.ufunc, v: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """fn (np.fmax / np.fmin) over v[s:e] for each window, NaN when empty (one reduceat call)."""
    if v.size == 0:
        return np.full(s.shape, np.nan)
    ext = np.append(v, np.nan)  # e may equal len(v)
    idx = np.empty(2 * s.size, np.int64)
    idx[0::2], idx[1::2] = np.minimum(s, v.size), np.minimum(np.maximum(e, s), v.size)
    r = fn.reduceat(ext, idx)[0::2]
    return np.where(e > s, r, np.nan)


def win_mean(v: np.ndarray, s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Mean of the finite v[s:e] for each window, NaN when none."""
    ok = np.isfinite(v)
    cs, cn = np.concatenate([[0.0], np.cumsum(np.where(ok, v, 0.0))]), np.concatenate([[0], np.cumsum(ok)])
    n = cn[e] - cn[s]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, (cs[e] - cs[s]) / n, np.nan)


def generic(gf: pl.DataFrame | None, gs: pl.DataFrame | None, T: np.ndarray = MARKS) -> dict[str, np.ndarray]:
    """Tick quantities at each time of T (default: the 5-minute marks) for stateless families: returns, imbalances, volatility, book."""
    out: dict[str, np.ndarray] = {}
    if gf is not None and gf.height > 2:
        a = E.arrays(gf, "fast")
        t, mid = a["t_local"], a["mid"]
        ok = np.isfinite(mid) & (mid > 0)
        t, lm = t[ok], np.log(mid[ok])
        at = lambda x, v: FI.at_marks(t, v, x)  # noqa: E731 -- value received by times x
        l0 = at(T, lm)
        bv, av = gf["bv1"].to_numpy().astype(float)[ok], gf["av1"].to_numpy().astype(float)[ok]
        b1, a1 = gf["b1"].to_numpy().astype(float)[ok], gf["a1"].to_numpy().astype(float)[ok]
        with np.errstate(divide="ignore", invalid="ignore"):
            micro = np.where((bv + av > 0) & (b1 > 0) & (a1 > 0), (b1 * av + a1 * bv) / (bv + av), np.exp(lm))  # volume-weighted mid
            lmi = np.log(micro)
            lpx = np.log(np.where(a["price"][ok] > 0, a["price"][ok], np.nan))
        lm0, li0, lp0 = l0, at(T, lmi), at(T, lpx)
        for h in (5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 240, 300, 450, 600, 900, 1800):
            out[f"ret{h}"] = lm0 - at(T - h, lm)
            out[f"mret{h}"] = li0 - at(T - h, lmi)  # microprice return
            out[f"pret{h}"] = lp0 - at(T - h, lpx)  # last-price return
        if "bv5" in gf.columns:
            b5, a5 = gf["bv5"].to_numpy().astype(float), gf["av5"].to_numpy().astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out["imb5"] = FI.at_marks(a["t_local"], (b5 - a5) / (b5 + a5), T)
        c2 = np.concatenate([[0.0], np.cumsum(np.diff(lm) ** 2)])
        for h in (60, 300, 900):
            i1, i0 = np.searchsorted(t, T, side="right") - 1, np.searchsorted(t, T - h, side="right") - 1
            out[f"rv{h}"] = np.sqrt(np.where(i1 >= 0, c2[np.clip(i1, 0, None)], np.nan) - np.where(i0 >= 0, c2[np.clip(i0, 0, None)], 0.0))
        for k in ("spread", "imb1", "timb", "lp_open", "lp_pc"):
            out[k] = FI.at_marks(a["t_local"], a[k], T)
        # windowed snapshot shapes the fingerprints point at: mean / max / min over (T - w, T] of the snapshots received
        sp, tick1 = a["spread"][ok], (a["spread_abs"][ok] <= 0.0101).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            for w in (30, 60, 120, 300, 600, 900, 1800):
                s0, e0 = np.searchsorted(t, T - w, side="right"), np.searchsorted(t, T, side="right")
                hi_w, lo_w = win_reduce(np.fmax, lm, s0, e0), win_reduce(np.fmin, lm, s0, e0)
                out[f"ldh{w}"], out[f"ldl{w}"] = hi_w - lm0, lo_w - lm0  # distance to the window's high / low (log)
                out[f"rng{w}"] = hi_w - lo_w
                out[f"lpma{w}"] = lm0 - win_mean(lm, s0, e0)  # log mid over its window average
                out[f"spm{w}"] = win_mean(sp, s0, e0)  # mean relative spread
                out[f"tick1_{w}"] = win_mean(tick1, s0, e0)  # share of snapshots with a one-tick spread
            rv = out.get("rv900")
            if rv is not None:
                for h in (30, 60, 300):  # returns in units of the 15-minute realised volatility, clipped like PM's +-10 features
                    out[f"zr{h}"] = np.clip(out[f"ret{h}"] / (rv * np.sqrt(h / 900)), -10, 10)
    if gs is not None and gs.height > 2:
        b = E.arrays(gs, "slow")
        t = b["t_local"]
        cont = gs["s"].to_numpy() >= 34200
        cum = {k: np.concatenate([[0.0], np.cumsum(v * cont)]) for k, v in
               (("v", b["vol"]), ("nv", b["vol"] * b["side"]), ("n", np.ones(t.size)), ("nn", b["side"]), ("a", b["amt"]), ("na", b["amt"] * b["side"]))}
        upto = lambda x, k: cum[k][np.searchsorted(t, x, side="right")]  # noqa: E731 -- running total at times x
        with np.errstate(divide="ignore", invalid="ignore"):
            for w in (10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900):
                win = {k: upto(T, k) - upto(T - w, k) for k in cum}
                out[f"imbv{w}"] = win["nv"] / win["v"]
                out[f"imbn{w}"] = win["nn"] / win["n"]
                out[f"imba{w}"] = win["na"] / win["a"]
                out[f"lvol{w}"] = np.log1p(win["v"])
                out[f"lamt{w}"] = np.log1p(win["a"])
                out[f"nam{w}"], out[f"nn{w}"] = win["na"], win["nn"]  # raw net amount (CNY) / net trade count
                out[f"tsz{w}"] = win["v"] / win["n"]  # mean trade size (shares)
                out[f"vfrac{w}"] = np.log(win["v"] / upto(T, "v"))  # window volume against the day's so far
                if w == 300:  # the last 5-minute bar's flows, raw: the profile ratios divide them by the samplerS medians
                    out["amt300"], out["vol300"], out["buy300"] = win["a"], win["v"], (win["a"] + win["na"]) / 2
            out["cum_imb"] = upto(T, "nv") / upto(T, "v")
            px = FI.at_marks(t, b["price"], T)
            out["lp_vwap"] = np.log(px / (upto(T, "a") / upto(T, "v")))
            hi, lo = np.fmax.accumulate(np.where(cont, b["price"], np.nan)), np.fmin.accumulate(np.where(cont, b["price"], np.nan))
            out["stoch"] = (px - FI.at_marks(t, lo, T)) / (FI.at_marks(t, hi, T) - FI.at_marks(t, lo, T))
    return out


INDEXES = {"hs300": (300, 3553), "zz500": (905, 3553), "zz1000": (852, 3553), "sh": (1, 3553), "sz": (399001, 3554)}


def index_series(con: duckdb.DuckDBPyConnection, date: int, stop: float) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    """Index snapshots received by the stop: name -> (trading-clock local time, log level, log previous close); {} without raw_index."""
    want = " OR ".join(f"(symbol_id = {c} AND market_id = {m})" for c, m in INDEXES.values())
    try:
        df = con.sql(f"""SELECT symbol_id, market_id, {B.tc(B.LSOD)} AS t_local, last_price::DOUBLE AS lp, pre_close_price::DOUBLE AS pc
            FROM raw_index WHERE date = {date} AND ({want}) AND {B.LSOD} <= {stop} AND last_price > 0 ORDER BY spider_ts""").pl()
    except duckdb.Error:
        return {}
    out = {}
    for name, (c, m) in INDEXES.items():
        g = df.filter((pl.col("symbol_id") == c) & (pl.col("market_id") == m))
        pc = g.filter(pl.col("pc") > 0)["pc"]
        if g.height > 2 and pc.len():
            out[name] = (g["t_local"].to_numpy().astype(float), np.log(g["lp"].to_numpy()), float(np.log(pc[0])))
    return out


def index_relative(gen: dict[str, np.ndarray], ix: dict, sid: int, T: np.ndarray) -> dict[str, np.ndarray]:
    """Index returns and the stock's excess over them: ir<h>_<ix> / xr<h>_<ix> (mid return minus index), ipc_ / xpc_ (vs previous close);
    'own' = the stock's exchange composite."""
    out: dict[str, np.ndarray] = {}
    names = {**{n: n for n in ix if n not in ("sh", "sz")}, "own": "sh" if sid >= 600000 else "sz"}
    for tag, n in names.items():
        if n not in ix:
            continue
        ti, li, lpc = ix[n]
        i0 = FI.at_marks(ti, li, T)
        out[f"ipc_{tag}"] = i0 - lpc
        if "lp_pc" in gen:
            out[f"xpc_{tag}"] = gen["lp_pc"] - (i0 - lpc)
        for h in (30, 60, 300, 900):
            ir = i0 - FI.at_marks(ti, li, T - h)
            out[f"ir{h}_{tag}"] = ir
            if f"ret{h}" in gen:
                out[f"xr{h}_{tag}"] = gen[f"ret{h}"] - ir
    return out


def family(expr: str) -> str:
    """Drop an expression's horizon / window digits and price flavour: -gen.ret30 and -gen.mret45 share one family."""
    return re.sub(r"\d+", "#", re.sub(r"gen\.[mp]?ret", "gen.ret", expr))


def stem_frames(frames: list[pl.DataFrame], stem: str) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Per mark: (sids, atoms) -- the stem's state quantities; expressions are evaluated on demand (``FM.evaluate``)."""
    return [FM.atom_columns(FM.atoms(fr, stem)) for fr in frames]


def mark_minutes(marks: np.ndarray) -> np.ndarray:
    """Minute of day (local) of each trading-clock mark."""
    sec = np.where(marks <= 7200, 34200 + marks, 46800 + marks - 7200)
    return np.floor(sec / 60).astype(int)


def profile_marks(hstats: Path, cache: Path, fams: list[str], d: str, marks: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
    """Map sid -> ``<FAM>.<V>@<off>`` -> value at each mark: the samplerS cell at the mark's minute (0), the one before / after."""
    mm = mark_minutes(marks)
    out: dict[int, dict[str, np.ndarray]] = {}
    for fam in fams:
        df = day_frame(hstats, fam, d, cache)
        if df is None:
            continue
        mins = np.array(sorted(df["minute"].unique().to_list()))
        daily = mins.size == 1
        k0 = np.zeros(mm.size, int) if daily else np.searchsorted(mins, mm, side="right") - 1  # last cell at or before the minute
        for V in sorted(df["V"].unique().to_list()):
            w = df.filter(pl.col("V") == V).pivot(on="minute", index="code", values="x", aggregate_function="first")
            cols = [str(m) for m in mins]
            M = w.select([pl.col(c).cast(pl.Float64) if c in w.columns else pl.lit(None, pl.Float64).alias(c) for c in cols]).to_numpy()
            codes = w["code"].to_numpy()
            for off in ((0,) if daily else (-1, 0, 1)):
                k = k0 + off
                ok = (k >= 0) & (k < mins.size)
                vals = np.where(ok, M[:, np.clip(k, 0, mins.size - 1)], np.nan)
                for c, row in zip(codes, vals, strict=True):
                    out.setdefault(int(c), {})[f"{fam}.{V}@{off}"] = row
    return out


XS = ("z:", "rank:", "crank:")  # cross-sectional transforms within a mark: each has the same CDF whatever it transforms
DM = ("dm:", "dmi:")  # minus the mark's cross-sectional mean / its Barra industry's mean: units kept (peer-relative, ZZQF)


def parse(expr: str) -> tuple[str, bool, bool, float, str]:
    """``crank:-gen.ret30`` / ``|gen.timb|*1e4`` -> (transform, negate, abs, scale, base): the names ``Pool`` makes."""
    pre = next((p for p in XS + DM if expr.startswith(p)), "")
    e = expr[len(pre):]
    scale = 1e4 if e.endswith("*1e4") else 1.0
    e = e.removesuffix("*1e4")
    ab = len(e) > 1 and e.startswith("|") and e.endswith("|")
    e = e[1:-1] if ab else e
    return pre, e.startswith("-"), ab, scale, e.removeprefix("-")


def xs(pre: str, v: np.ndarray) -> np.ndarray:
    """One mark's values across stocks as is, z-scored, or ranked in [0, 1] / [-0.5, 0.5] (average ranks)."""
    if not pre:
        return v
    f = np.isfinite(v)
    out = np.full(v.size, np.nan)
    if f.sum() < 2:
        return out
    if pre == "z:":
        sd = v[f].std()
        if sd > 0:
            out[f] = (v[f] - v[f].mean()) / sd
        return out
    out[f] = (pl.Series(v[f]).rank("average").to_numpy() - 1) / (f.sum() - 1)
    return out - 0.5 if pre == "crank:" else out


def demean(v: np.ndarray, g: np.ndarray) -> np.ndarray:
    """v minus its group's mean over finite values (groups of fewer than 3, and group -1, give NaN)."""
    f = np.isfinite(v) & (g >= 0)
    out = np.full(v.size, np.nan)
    if not f.any():
        return out
    k = g[f]
    n = np.bincount(k)
    mean = np.bincount(k, weights=v[f]) / np.maximum(n, 1)
    ok = n[k] >= 3
    out[np.flatnonzero(f)[ok]] = v[f][ok] - mean[k][ok]
    return out


class Pool:
    """The candidates of features whose stem keeps no state, at every (day, mark).

    Generic tick quantities (``gen.``), CED daily fields and intraday-over-daily ratios (``ced.``).

    Only the bases are stored; a candidate name is evaluated on demand (``value``), so the unit variants and the
    cross-sectional transforms cost no memory.
    """

    def __init__(self, xs_on: bool) -> None:
        self.xs_on = xs_on
        self.sids: list[list[np.ndarray]] = []  # per block, per mark
        self.base: list[list[dict[str, np.ndarray]]] = []
        self.daily: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []  # per block: CED sids, CED columns
        self.didx: list[list[np.ndarray]] = []  # per block, per mark: each stock's row in the CED columns, -1 = none
        self.ind: list[np.ndarray] = []  # per block: each CED row's Barra industry (argmax of the 0 / 1 exposures), -1 = none
        self._samples: dict[str, np.ndarray] | None = None

    def add_block(self, frames: list[pl.DataFrame], dc: pl.DataFrame | None) -> None:
        """One day: its per-mark frames (keys ``gen.*`` / ``ced.*``) and its CED daily table (one row per sid)."""
        sids, base, didx = [], [], []
        dsid = dc["sid"].to_numpy() if dc is not None else np.array([], int)
        dcol = {f"ced.{c}": dc[c].to_numpy().astype(float) for c in dc.columns if c != "sid"} if dc is not None else {}
        for fr in frames:
            g = (fr.filter(pl.col("key").str.starts_with("gen.") | pl.col("key").str.starts_with("ced."))
                 .pivot(on="key", index="sid", values="pm", aggregate_function="first").sort("sid"))
            s = g["sid"].to_numpy()
            sids.append(s)
            base.append({c: g[c].to_numpy().astype(float) for c in g.columns if c != "sid"})
            i = np.searchsorted(dsid, s)
            ok = (i < dsid.size) & (dsid[np.clip(i, 0, max(dsid.size - 1, 0))] == s) if dsid.size else np.zeros(s.size, bool)
            didx.append(np.where(ok, i, -1))
        self.sids.append(sids)
        self.base.append(base)
        self.daily.append((dsid, dcol))
        dummies = [v for k, v in dcol.items() if k.startswith("ced.exp.") and np.isin(v[np.isfinite(v)], (0.0, 1.0)).all()
                   and 0 < np.nanmean(v) < 0.5]  # industry dummies (the country factor is 1 everywhere)
        if len(dummies) >= 5:
            st = np.nan_to_num(np.stack(dummies))
            self.ind.append(np.where(st.max(axis=0) == 1, st.argmax(axis=0), -1))
        else:
            self.ind.append(np.full(dsid.size, -1))
        self.didx.append(didx)
        self._samples = None

    def bases(self) -> list[str]:
        """Every base quantity seen on any day."""
        out = {k for blk in self.base for b in blk for k in b} | {k for _, d in self.daily for k in d}
        return sorted(out)

    def names(self) -> list[str]:
        """Every candidate: tick bases with sign / abs / bp units, CED bases with sign; each also cross-sectional."""
        out = []
        for c in self.bases():
            out += [c, f"-{c}", f"|{c}|", f"{c}*1e4", f"|{c}|*1e4"] if c.startswith("gen.") else [c, f"-{c}"]
            if self.xs_on:
                out += [f"{p}{s}{c}" for p in XS for s in ("", "-")]
            if c.startswith("gen."):
                out += [f"{p}{s}{c}" for p in DM for s in ("", "-")]
        return out

    def raw(self, b: int, m: int, base: str) -> np.ndarray | None:
        """Return a base at (block, mark), aligned with ``sids[b][m]``."""
        if base in self.base[b][m]:
            return self.base[b][m][base]
        col = self.daily[b][1].get(base)
        if col is None:
            return None
        i = self.didx[b][m]
        return np.where(i >= 0, col[np.clip(i, 0, None)], np.nan)

    def value(self, b: int, m: int, expr: str) -> np.ndarray | None:
        """Candidate ``expr`` at (block, mark), aligned with ``sids[b][m]``; None when its base is absent."""
        pre, neg, ab, scale, base = parse(expr)
        v = self.raw(b, m, base)
        if v is None:
            return None
        v = np.abs(v) if ab else v
        v = (-v if neg else v) * scale
        if pre in DM:
            i = self.didx[b][m]
            g = np.where(i >= 0, self.ind[b][np.clip(i, 0, None)], -1) if pre == "dmi:" and self.ind[b].size else np.zeros(v.size, int)
            return demean(v, g if pre == "dmi:" else np.zeros(v.size, int))
        return xs(pre, v)

    def samples(self, n: int = 20000, every: int = 3) -> dict[str, np.ndarray]:
        """Each candidate's finite values over every third mark of every day, subsampled to n: enough for a CDF to ~0.005."""
        if self._samples is None:
            rng = np.random.default_rng(0)
            out = {}
            for k in self.names():
                vs = [self.value(b, m, k) for b in range(len(self.sids)) for m in range(0, len(self.sids[b]), every)]
                v = np.concatenate([x for x in vs if x is not None] or [np.array([])])
                v = v[np.isfinite(v)]
                if v.size >= 50:
                    out[k] = rng.choice(v, n, replace=False) if v.size > n else v
            self._samples = out
        return self._samples


def ic_vec(pred: np.ndarray, ret: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-sectional Spearman IC per group (mark), NaN where it cannot be computed."""
    order = np.argsort(groups, kind="stable")  # one sort, then contiguous slices (rows keep their order inside a group)
    g, p, r = groups[order], pred[order], ret[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    out = []
    for s, e in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [g.size]]), strict=True):
        ps, rs = p[s:e], r[s:e]
        m = np.isfinite(ps) & np.isfinite(rs)
        if m.sum() < 20 or np.std(ps[m]) == 0:
            out.append(np.nan)
            continue
        r1 = np.argsort(np.argsort(ps[m])).astype(float)
        r2 = np.argsort(np.argsort(rs[m])).astype(float)
        out.append(np.corrcoef(r1, r2)[0, 1])
    return np.array(out)


def summary(v: np.ndarray) -> tuple[float, float, int]:
    """Mean, t-stat and count of the finite entries."""
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan, np.nan, int(v.size)
    return float(v.mean()), float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size))), int(v.size)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1", help="sub-models to average (s1,s2,s3,s4 for all; each costs a predict per candidate)")
    ap.add_argument("--assign", type=Path, help="json feature -> expression to start from (default: KNOWN)")
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--min-t", type=float, default=2.0, help="keep a candidate when its paired IC gain over marks has t >= this")
    ap.add_argument("--min-in-range", type=float, default=0.95, help="a candidate must sit inside the feature's training range this often")
    ap.add_argument("--max-score", type=float, default=1.5, help="and its p10 / median / p90 within this distance of the fingerprint's")
    ap.add_argument("--cdf", type=Path, default=Path("data/reports/stage5/cdf.parquet"),
                    help="root-split training-CDF points (pm_models_probe); when present a feature with >= 8 points is ranked and gated by them")
    ap.add_argument("--max-cdf", type=float, default=0.05, help="gate: mean |F(t) - p| over the CDF points (right formulas sit at 0.01-0.02 on 97)")
    ap.add_argument("--drop-gap", type=float, default=0.05, help="drop a starting assignment whose training-CDF gap exceeds this")
    ap.add_argument("--accept-cdf", type=float, default=0.03,
                    help="with no IC winner, take the closest candidate by training CDF when its gap is at most this (sign by IC)")
    ap.add_argument("--max-features", type=int, default=30, help="scan this many features, by model gain")
    ap.add_argument("--ced", action=argparse.BooleanOptionalAction, default=True,
                    help="also CED daily fields and intraday amount / volume over daily baselines (ced_match.daily_candidates)")
    ap.add_argument("--ced-xr", type=Path, help="read shtcommon's CED files from this directory instead of the crucible store")
    ap.add_argument("--xs", action=argparse.BooleanOptionalAction, default=True,
                    help="also each stateless candidate cross-sectionally z-scored / ranked within its mark")
    ap.add_argument("--max-worsen", type=float, default=None,
                    help="a feature with a pick only moves to a candidate whose training-CDF gap is at most this much worse (off by default)")
    ap.add_argument("--xs-shortlist", type=int, default=6,
                    help="z / rank candidates look alike to the CDF: try the gate-passing ones of this many bases with the largest own |IC|")
    ap.add_argument("--engine-dir", type=Path, default=Path("data/engine"), help="marks_<D>.parquet from run_cpp.py --marks <the same grid>")
    ap.add_argument("--marks", default="five", help="five | rand:N[:seed] | a,b,...: the times scored (run_cpp.py --marks must match)")
    ap.add_argument("--target", type=Path, help="pm_signals.py output: score against PM's own live prediction (the latest signal of each "
                                                "stock at the mark) instead of the forward return; days without a file are skipped")
    ap.add_argument("--target-stale", type=float, default=10.0, help="--target: a signal older than this many seconds at the mark is no value")
    ap.add_argument("--profiles", default="ZZWA,ZZAL,ZZQO,ZZXC,ZZTS", help="samplerS families whose cells join the pool ('' = none)")
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--pm-cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--triples", action=argparse.BooleanOptionalAction, default=True, help="state candidates (x - y) / z")
    ap.add_argument("--min-distinct", type=int, default=30, help="fewer distinct values = a dummy: only for flag / count features")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage6"))
    args = ap.parse_args()
    global MARKS
    MARKS = mark_grid(args.marks)
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in args.feature_csv.read_text(errors="replace").splitlines() if ln.strip()]
    names = [ln.split(",")[0].strip() for ln in lines]
    if names and names[0].lower() == "name":
        names = names[1:]
    col = {n: i for i, n in enumerate(names)}
    fp = pl.read_parquet(args.fingerprint).with_columns(pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"))
    fp_kind = dict(zip(fp["feature"].to_list(), fp["kind"].to_list(), strict=True)) if "kind" in fp.columns else {}
    pfams = [f for f in args.profiles.split(",") if f]
    cdf = {}
    if args.cdf.exists():
        for g in pl.read_parquet(args.cdf).filter(pl.col("nan_left") == 0).partition_by("feature"):  # NaN rows sit on the right: left share is exact
            if g.height >= 8:
                cdf[g["feature"][0]] = (g["t"].to_numpy(), g["p"].to_numpy())
        print(f"training-CDF points for {len(cdf)} features")
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    CM = None
    if args.ced:
        import ced_match as CM  # lazy: it reads the CED store

        if args.ced_xr:
            CM.S = CM.XRStore(args.ced_xr)
        print(f"CED candidates from {CM.S.STORE}")
    pool = Pool(args.xs)
    state_stems: set[str] = set()
    blocks = []  # per day: (models, frames per mark, sids per mark, forward returns per mark)
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        tgt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if args.target:
            tf = args.target / f"date={d}.parquet"
            if not tf.exists():
                print(f"skip {d}: no PM signals {tf}")
                continue
            for g in pl.read_parquet(tf, columns=["sid", "t", "prediction"]).sort("sid", "t").partition_by("sid"):
                tgt[int(g["sid"][0])] = (g["t"].to_numpy().astype(float), g["prediction"].to_numpy().astype(float))
        pm = B.load_state(f)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(pm["sid"].unique().to_list())
        up = E.load(con, int(d), sids, stop)
        ix = index_series(con, int(d), stop)
        print(f"{d}: index snapshots for {sorted(ix) or 'none (decode with KINDS=index research/backfill.sh)'}")
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        dc = CM.daily_candidates(d, sids) if CM is not None else None
        dbase = {r["sid"]: r for r in dc.select("sid", "adv20", "avgvol20", "freeshare").fill_null(np.nan).iter_rows(named=True)} if dc is not None else {}
        eng: dict[int, dict[str, np.ndarray]] = {}
        ef = args.engine_dir / f"marks_{d}.parquet"
        if ef.exists():  # the engine's state at the marks (every solved rule) over the Python rebuild's
            em = pl.read_parquet(ef).filter(pl.col("mark").is_in(MARKS.tolist())).sort("sid", "mark")
            keep = [c for c in em.columns if c not in ("sid", "mark") and "|" not in c]
            for g in em.partition_by("sid"):
                if g.height == MARKS.size:
                    eng[int(g["sid"][0])] = {c: g[c].to_numpy() for c in keep}
        prof = profile_marks(args.hstats, args.pm_cache, pfams, d, MARKS) if pfams else {}
        rets: dict[int, np.ndarray] = {}
        b_sid, b_key, b_val = [], [], []  # one long frame built once per day (not one small frame per key)

        def add(sid: int, key: str, v: np.ndarray, bs=b_sid, bk=b_key, bv=b_val) -> None:
            bs.append(sid)
            bk.append(key)
            bv.append(np.broadcast_to(np.asarray(v, float), (MARKS.size,)))  # a scalar fills every mark, as the old frame did
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            built = FI.rebuild(gf, gs, MARKS)
            fs = {k.split(".")[0] for k in built}  # only stems the rebuild keeps: zzvm etc. stay on the shared pool (daily ratios)
            built.update({k: v for k, v in eng.get(sid, {}).items() if k.split(".")[0] in fs})
            for k, v in built.items():
                add(sid, k, v)
            if sid in dbase and "zzqj.v" in built:  # intraday over daily: amount / volume since 09:30 over 20-day averages, volume over float
                bs = dbase[sid]
                with np.errstate(divide="ignore", invalid="ignore"):
                    for k, v in (("amt_adv20", built["zzqj.tpv"] / bs["adv20"]), ("vol_avg20", built["zzqj.v"] / bs["avgvol20"]),
                                 ("vol_float", built["zzqj.v"] / bs["freeshare"])):
                        add(sid, f"ced.{k}", np.asarray(v, float))
            if gf is not None and gf.height > 2:
                a = E.arrays(gf, "fast")
                mid = a["mid"]
                ok = np.isfinite(mid) & (mid > 0)
                m0 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS)
                m1 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS + HORIZON)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets[sid] = np.log(m1 / m0)
            if args.target:  # PM's prediction made on the stock's latest snapshot at or before the mark (exchange clock)
                ts, pv = tgt.get(sid, (np.zeros(0), np.zeros(0)))
                last_t = FI.at_marks(ts, ts, MARKS)
                rets[sid] = np.where(MARKS - last_t <= args.target_stale, FI.at_marks(ts, pv, MARKS), np.nan)
                add(sid, "px", FI.at_marks(a["t_local"], a["price"], MARKS))
            gen = generic(gf, gs, MARKS)
            gen.update(index_relative(gen, ix, sid, MARKS))
            for k, v in gen.items():
                add(sid, f"gen.{k}", v.astype(float))
            for k, v in prof.get(sid, {}).items():
                add(sid, f"ced.{k}", v)
                if not k.endswith("@1"):  # the last bar's flow over its profile cell (log): the bar as it ended, before / at the minute
                    with np.errstate(divide="ignore", invalid="ignore"):
                        for r in ("amt300", "vol300", "buy300"):
                            if r in gen:
                                add(sid, f"ced.lr.{r}.{k}", np.log(gen[r] / v))
        nm = MARKS.size
        long = pl.DataFrame({"sid": np.repeat(np.array(b_sid, np.int64), nm), "mark": np.tile(MARKS, len(b_sid)),
                             "key": np.repeat(np.array(b_key, dtype=object), nm), "pm": np.concatenate(b_val) if b_val else np.zeros(0)},
                            schema={"sid": pl.Int64, "mark": pl.Float64, "key": pl.String, "pm": pl.Float64}).filter(pl.col("pm").is_finite())
        by_mark = long.partition_by("mark", as_dict=True, include_key=False)  # one pass instead of a filter per mark
        empty = long.drop("mark").clear()
        frames = [by_mark.get((m,), empty) for m in MARKS]
        pool.add_block(frames, dc)
        state_stems |= set(long["key"].str.extract(r"^([^.]+)\.", 1).drop_nulls().unique().to_list()) - {"gen", "ced"}
        models = load_models(args.models, d, args.fits.split(","), args.subs.split(","))
        blocks.append((d, models, frames, rets))
        print(f"{d}: {len(sids)} stocks x {MARKS.size} marks, {len(models)} models, engine state for {len(eng)}, "
              f"profiles for {len(prof)} ({time.perf_counter() - t0:.0f}s)")
    if not blocks:
        return
    # the panel: one row per (day, mark, stock with a forward return)
    rows_sid, rows_grp, rows_ret, rows_blk, rows_mark = [], [], [], [], []
    for bi, (_, _, _, rets) in enumerate(blocks):
        for mi in range(MARKS.size):
            for sid, r in rets.items():
                rows_sid.append(sid)
                rows_grp.append(bi * 1000 + mi)
                rows_ret.append(r[mi])
                rows_blk.append(bi)
                rows_mark.append(mi)
    sid_a, grp, ret, blk, mk = (np.array(x) for x in (rows_sid, rows_grp, rows_ret, rows_blk, rows_mark))
    X = np.full((sid_a.size, len(names)), np.nan)
    pos = {(b, m, s): i for i, (b, m, s) in enumerate(zip(blk, mk, sid_a, strict=True))}
    cache: dict[str, list] = {}  # stateful stem -> per block, per mark: (sids, candidates)

    def rows_of(b: int, m: int, sids: np.ndarray) -> np.ndarray:
        """Panel row of each stock at (block, mark), -1 when it has no forward return."""
        return np.array([pos.get((b, m, int(s)), -1) for s in sids], int)

    pidx = [[rows_of(b, m, s) for m, s in enumerate(per)] for b, per in enumerate(pool.sids)]  # the pool's stocks -> panel rows

    def stem_cache(stem: str) -> list:
        if stem not in cache:
            cache[stem] = [[(s, c, rows_of(b, m, s)) for m, (s, c) in enumerate(stem_frames(fr, stem))]
                           for b, (_, _, fr, _) in enumerate(blocks)]
        return cache[stem]

    pooled_cache: dict[str, dict[str, np.ndarray]] = {}

    def stem_pool(stem: str) -> dict[str, np.ndarray]:
        """Every candidate of a stem over days and marks, as finite samples (atoms pooled, then 30000 rows, then expressions)."""
        if stem not in state_stems:
            return pool.samples()
        if stem not in pooled_cache:
            per = [(s, c) for blk in stem_cache(stem) for s, c, _ in blk]
            names = sorted({k for _, c in per for k in c})
            cols = {k: np.concatenate([c.get(k, np.full(s.size, np.nan)) for s, c in per]) for k in names}
            n = next(iter(cols.values())).size if cols else 0
            if n > 30000:
                idx = np.sort(np.random.default_rng(0).choice(n, 30000, replace=False))
                cols = {k: v[idx] for k, v in cols.items()}
            ex = FM.expressions(cols, args.triples)
            pooled_cache[stem] = {k: v[np.isfinite(v)] for k, v in ex.items() if np.isfinite(v).sum() >= 50}
        return pooled_cache[stem]

    def column(feature: str, expr: str) -> np.ndarray:
        """One model column filled with expression ``expr`` of the feature's stem."""
        stem = feature.split(".")[0]
        v = np.full(sid_a.size, np.nan)
        for b in range(len(blocks)):
            for m in range(MARKS.size):
                if stem in state_stems:
                    _, atoms, ix = stem_cache(stem)[b][m]
                    x = FM.evaluate(expr, atoms)
                else:
                    x, ix = pool.value(b, m, expr), pidx[b][m]
                if x is not None:
                    ok = ix >= 0
                    v[ix[ok]] = x[ok]
        return v

    base_ic: dict[str, float] = {}

    def own_ic(base: str) -> float:
        """Return a stateless base's own IC against the forward return: the same for its z-score and rank (Spearman, per mark)."""
        if base not in base_ic:
            base_ic[base] = summary(ic_vec(column("", base), ret, grp))[0]
        return base_ic[base]

    brows = [np.flatnonzero(blk == b).astype(np.int64) for b in range(len(blocks))]
    flat = all(hasattr(ms[0], "trees_of") for _, ms, _, _ in blocks)

    def panel(Xm: np.ndarray) -> np.ndarray:
        """Every panel row's prediction (each day with its own models)."""
        pred = np.empty(sid_a.size)
        for b, (_, models, _, _) in enumerate(blocks):
            pred[brows[b]] = predict_rows(models, Xm, brows[b])
        return pred

    def score(Xm: np.ndarray) -> np.ndarray:
        return ic_vec(panel(Xm), ret, grp)

    class Swap:
        """Column j's candidates without re-running the whole model: the prediction minus the trees on j, plus them anew."""

        def __init__(self, j: int, pred: np.ndarray | None = None) -> None:
            self.j, self.pred = j, panel(X) if pred is None else pred
            self.old = [predict_rows(ms, X, brows[b], j) for b, (_, ms, _, _) in enumerate(blocks)] if flat else None

        def score(self, colv: np.ndarray) -> np.ndarray:
            X[:, self.j] = colv
            if not flat:
                return score(X)
            pred = self.pred.copy()
            for b, (_, ms, _, _) in enumerate(blocks):
                pred[brows[b]] += predict_rows(ms, X, brows[b], self.j) - self.old[b]
            return ic_vec(pred, ret, grp)

    assign = dict(KNOWN)
    if args.assign and args.assign.exists():
        assign = json.loads(args.assign.read_text())
    for feat, expr in list(assign.items()):
        if feat in col:
            X[:, col[feat]] = column(feat, expr)
            v = X[:, col[feat]]
            v = v[np.isfinite(v)]
            if feat in cdf and v.size and (g := FM.cdf_gap(v, cdf[feat])) > args.drop_gap:  # a start pick off the training CDF is dropped
                print(f"drop {feat} = {expr}: training-CDF gap {g:.3f} > {args.drop_gap}")
                X[:, col[feat]] = np.nan
                del assign[feat]
            elif fp_kind.get(feat) not in ("flag", "count", "const") and np.unique(v).size < args.min_distinct:
                print(f"drop {feat} = {expr}: {np.unique(v).size} distinct values on a {fp_kind.get(feat)} feature")
                X[:, col[feat]] = np.nan
                del assign[feat]
    base = summary(score(X))
    print(f"\nall-missing except the assignment {assign}: IC {base[0]:.4f} (t {base[1]:.2f}, {base[2]} marks) ({time.perf_counter() - t0:.0f}s)")
    log = []
    feats = [f for f in fp.sort("gain_share", descending=True)["feature"].to_list() if f in col][: args.max_features]
    for ps in range(args.passes):
        for feat in feats:
            stem = feat.split(".")[0]
            cand_all = stem_pool(stem)
            if not cand_all:
                continue
            ranked = FM.rank(fp.filter(pl.col("feature") == feat), cand_all, len(cand_all), cdf)
            cut = args.max_cdf if feat in cdf else args.max_score
            # gate on the fingerprint: right units and range first, IC second -- IC alone rewards any useful signal in any column
            rk = {r["candidate"]: r for r in ranked}
            dummy_ok = fp_kind.get(feat) in ("flag", "count", "const")
            n_distinct: dict[str, int] = {}

            def varied(c: str, ok: bool = dummy_ok, pool_: dict = cand_all, nd: dict = n_distinct) -> bool:
                if ok or c not in pool_:
                    return True
                if c not in nd:
                    nd[c] = int(np.unique(pool_[c][:5000]).size)
                return nd[c] >= args.min_distinct

            in_scale = lambda r: (not np.isfinite(r["in_range"]) or r["in_range"] >= args.min_in_range) and varied(r["candidate"])  # noqa: E731
            passes = [r["candidate"] for r in ranked if in_scale(r) and r["score"] <= cut]
            short = [c for c in passes if not c.startswith(XS)][: args.shortlist]
            by_base: dict[str, dict[bool, str]] = {}  # z / rank: the CDF cannot tell what is transformed, so bases go by own |IC|
            for c in passes:
                if c.startswith(XS):
                    _, neg, _, _, b = parse(c)
                    by_base.setdefault(b, {}).setdefault(neg, c)  # per sign, the best-CDF transform
            top_b = sorted(by_base, key=lambda b: -np.nan_to_num(abs(own_ic(b))))[: args.xs_shortlist]
            short += [c for b in top_b for c in by_base[b].values()]
            if assign.get(feat):  # refine: every gate-passing neighbour of the current pick (other horizons / windows, mid / micro / last price)
                fam = family(assign[feat])
                short = list(dict.fromkeys(short + [c for c in passes if family(c) == fam and c != assign[feat]]))
            if not short:
                log.append({"pass": ps, "feature": feat, "kept": assign.get(feat), "ic_after": np.nan, "t_gain": 0.0,
                            "top": "no candidate passes the fingerprint gate; closest: "
                                   + "; ".join(f"{r['candidate']} (score {r['score']:.2f}, in {r['in_range']:.2f})" for r in ranked[:2])})
                print(f"pass {ps} {feat}: no candidate passes the fingerprint gate ({time.perf_counter() - t0:.0f}s)")
                continue
            cur = assign.get(feat)
            if args.max_worsen is not None and feat in cdf and cur in rk:  # IC may not trade a better-fitting formula for a worse one
                short = [c for c in short if rk[c]["score"] <= rk[cur]["score"] + args.max_worsen]
            j = col[feat]
            keep_col = X[:, j].copy()
            sw = Swap(j)
            cur_v = np.nan_to_num(ic_vec(sw.pred, ret, grp), nan=0.0)  # a constant prediction (everything missing) ranks nothing: IC 0
            best = (summary(cur_v)[0], cur, 0.0)
            tried = []
            for expr in short:
                v = np.nan_to_num(sw.score(column(feat, expr)), nan=0.0)
                dv = v - cur_v  # paired over marks: the same marks, the same other columns
                _, t_gain, _ = summary(dv)
                tried.append((expr, summary(v)[0], t_gain))
                if np.isfinite(t_gain) and t_gain >= args.min_t and summary(v)[0] > best[0]:
                    best = (summary(v)[0], expr, t_gain)
            # accept by the training CDF: the closest candidate within --accept-cdf, its sign picked by IC, unless IC drops significantly
            plain = [r for r in ranked if not r["candidate"].startswith(XS) and in_scale(r)]  # a z / rank shape names no variable
            if feat in cdf and best[1] == cur and plain and plain[0]["score"] <= args.accept_cdf and cur is None:
                top = plain[0]["candidate"]
                # the other sign only when it fits too: negating a one-sided quantity (volatility, amount) moves its whole CDF
                pair = [e for e in (top, top[1:] if top.startswith("-") else "-" + top)
                        if e in rk and rk[e]["score"] <= args.accept_cdf and in_scale(rk[e])]
                opts = [(e, ic, tg) for e, ic, tg in tried if e in pair]
                for e in pair:
                    if e not in {o[0] for o in opts} and e in cand_all:
                        v = np.nan_to_num(sw.score(column(feat, e)), nan=0.0)
                        opts.append((e, summary(v)[0], summary(v - cur_v)[1]))
                opts = [o for o in opts if np.isfinite(o[1])]
                if opts:
                    e, ic, tg = max(opts, key=lambda o: o[1])
                    if not (np.isfinite(tg) and tg <= -args.min_t):
                        best = (ic, e, tg)
                        print(f"  accept {feat} = {e} by training CDF (gap {plain[0]['score']:.3f}), IC {ic:.4f} t {tg:.1f}")
            if best[1] != cur and best[1] is not None:
                assign[feat] = best[1]
                X[:, j] = column(feat, best[1])
            else:
                X[:, j] = keep_col
            tried.sort(key=lambda t: -t[2] if np.isfinite(t[2]) else 1e9)
            log.append({"pass": ps, "feature": feat, "kept": assign.get(feat), "ic_after": best[0], "t_gain": best[2],
                        "top": "; ".join(f"{e} {ic:.4f} (t {tg:.1f})" for e, ic, tg in tried[:3])})
            print(f"pass {ps} {feat}: kept {assign.get(feat)} IC {best[0]:.4f} t_gain {best[2]:.1f} ({time.perf_counter() - t0:.0f}s)")
    full_pred = panel(X)
    final = summary(ic_vec(full_pred, ret, grp))
    full_v = np.nan_to_num(ic_vec(full_pred, ret, grp), nan=0.0)
    evid = []
    for feat, expr in assign.items():  # ablation: the IC lost when the column goes back to missing, and the training-CDF gap
        if feat not in col:
            continue
        j = col[feat]
        keep_col = X[:, j].copy()
        loss, t_loss, _ = summary(full_v - np.nan_to_num(Swap(j, full_pred).score(np.full(sid_a.size, np.nan)), nan=0.0))
        X[:, j] = keep_col
        v = keep_col[np.isfinite(keep_col)]
        gap = FM.cdf_gap(v, cdf[feat]) if feat in cdf and v.size else np.nan
        src = ("state" if feat.split(".")[0] in state_stems else "xs" if expr.startswith(XS)
               else expr.lstrip("-|").split(".")[0])
        grade = ("strong" if gap <= 0.02 and t_loss >= 2 else "fits" if gap <= args.accept_cdf else "weak") if np.isfinite(gap) \
            else ("ic" if t_loss >= 2 else "weak")
        evid.append({"feature": feat, "formula": expr, "source": src, "cdf_gap": gap, "ic_loss": loss, "t_loss": t_loss, "grade": grade})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ic_assign.json").write_text(json.dumps(assign, indent=1))
    if evid:
        ev = pl.DataFrame(evid).sort("grade", "t_loss", descending=[False, True])
        ev.write_parquet(args.out / "ic_evidence.parquet")
        with pl.Config(tbl_rows=80, tbl_cols=7, tbl_width_chars=200, fmt_str_lengths=40, float_precision=4):
            print("\n== evidence per assigned column: training-CDF gap, IC lost without it (t); strong = gap <= 0.02 and t >= 2")
            print(ev)
    if log:
        pl.DataFrame(log).write_parquet(args.out / "ic_log.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=6, tbl_width_chars=230, fmt_str_lengths=90, float_precision=4):
        if log:
            print("\n== scan: per feature, the expression kept and the three with the largest paired gain (IC, t of the gain over marks)")
            print(pl.DataFrame(log))
    print(f"\nIC: start {base[0]:.4f} -> end {final[0]:.4f} (t {final[1]:.2f}, {final[2]} marks); "
          f"{len(assign)} of {len(names)} columns assigned -> {args.out / 'ic_assign.json'}")


if __name__ == "__main__":
    main()
