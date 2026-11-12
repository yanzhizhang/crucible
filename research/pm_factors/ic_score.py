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
import hashlib
import inspect
import json
import re
import shutil
import sys
import time
from collections import Counter
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
STS_MAP: dict[str, str] = {}  # --pm-sts-map: trading day -> the stsfile day 224's PM loaded


def mark_grid(spec: str) -> np.ndarray:
    """'five' = the 5-minute marks; 'rand:N[:seed]' = N random trading-clock seconds in (0, 13800) (the same list in
    ``run_cpp.py``): PM scores at every snapshot, so mid-bar times sample its training distribution; else a comma list.
    """
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


ROW_T = np.zeros(0)  # --sub-by-time: each panel row's mark on the trading clock (set once the panel is built)


class _JKey(dict):
    """trees_of of a SegModel: every column maps to itself, each sub-model looks its own trees up."""

    def get(self, k, default=None):  # noqa: ARG002 -- signature of dict.get
        return ("col", k)


class SegModel:
    """PM's sub-models as time-of-day segments: row r is scored by the sub-model whose (lo, hi] holds ROW_T[r]."""

    def __init__(self, segs: list[tuple[float, float, object]]):
        self.segs = segs
        self.trees_of = _JKey()

    def predict(self, X: np.ndarray, rows: np.ndarray, trees=None) -> np.ndarray:
        t = ROW_T[rows]
        out = np.full(rows.size, np.nan)
        for lo, hi, m in self.segs:
            sel = np.flatnonzero((t > lo) & (t <= hi))
            if sel.size:
                tr = None if trees is None else m.trees_of.get(trees[1], np.zeros(0, np.int64))
                out[sel] = m.predict(X, rows[sel], tr)
        return out


def load_seg_models(root: Path, date: str, fits: list[str], spec: str) -> list:
    """--sub-by-time "s1:600,s2:1800,s3:12600,s4:14220": per fit one SegModel, segment k = (end of k-1, end of k] seconds."""
    from tree_kernel import FlatModel  # the segment dispatch needs the row-indexed walker

    months = sorted(p.name for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name) and p.name <= date[:6])
    out, lo = [], -np.inf
    parts = [(x.split(":")[0], float(x.split(":")[1])) for x in spec.split(",") if x]
    for f in fits:
        segs, lo = [], -np.inf
        for s_, hi in parts:
            fn = root / months[-1] / "lgb" / f / f"model_{s_}.txt"
            segs.append((lo, hi if hi < parts[-1][1] else np.inf, FlatModel.from_file(fn)))
            lo = hi
        out.append(SegModel(segs))
    print(f"{date}: sub-models by time of day {spec} ({months[-1]}, fits {fits})")
    return out


SEED_KEYS = ("zzeg.c.high", "zzeg.c.low", "zzeg.c.span", "zzeg.c.rEma.sum", "zzeg.c.rEma.wgt", "zzxu.last", "zzpk.high", "zzpk.low",
             "zzpk.open", "zzxo.r.29", "zzxo.r.0", "zzui.r.9", "zzmh.hm")


def seed_state(args, d: str) -> dict[int, dict[str, float]]:
    """--seed-dir: the .jz PM started day d from (--seed-map D=YYYYMMDD, else the latest file before d): sid -> SEED_KEYS."""
    if not args.seed_dir:
        return {}
    import gzip

    mp = dict(x.split("=") for x in args.seed_map.split(",") if x.strip())
    if d in mp:
        f = args.seed_dir / f"{mp[d]}.jz"
    else:
        fs = sorted(p for p in args.seed_dir.glob("2*.jz") if p.stem < d)
        f = fs[-1] if fs else None
    if f is None or not f.exists():
        print(f"{d}: no seed state in {args.seed_dir}")
        return {}
    with gzip.open(f) as fh:
        raw = json.loads(fh.read())
    out = {int(k): {x: float(kv[x]) for x in SEED_KEYS if isinstance(kv.get(x), (int, float))} for k, kv in raw.items()}
    print(f"{d}: seed state from {f.name} for {len(out)} stocks")
    return out


def seed_feats(sd: dict[str, float], a: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """gen.sd_*: the zzeg channel (c.high / c.low carried across days, widened by today's range so far), its EMA, and
    yesterday's late prices, each against the price at the mark."""
    t, px = a["t_local"], a["price"]
    ok = np.isfinite(px) & (px > 0)
    if ok.sum() < 3:
        return {}
    t, px = t[ok], px[ok]
    p0 = FI.at_marks(t, px, MARKS)
    hi_d = FI.at_marks(t, np.maximum.accumulate(px), MARKS)
    lo_d = FI.at_marks(t, np.minimum.accumulate(px), MARKS)
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        hi, lo, sp = sd.get("zzeg.c.high", np.nan), sd.get("zzeg.c.low", np.nan), sd.get("zzeg.c.span", np.nan)
        if np.isfinite(hi) and np.isfinite(lo):
            hit, lot = np.fmax(hi_d, hi), np.fmin(lo_d, lo)
            out.update({"sd_eg_pos": (p0 - lo) / (hi - lo), "sd_eg_px_hi": p0 / hi, "sd_eg_px_lo": p0 / lo, "sd_eg_hi_px": hi / p0,
                        "sd_eg_lo_px": lo / p0, "sd_eg_posw": (p0 - lot) / (hit - lot), "sd_eg_px_hiw": p0 / hit, "sd_eg_px_low": p0 / lot,
                        "sd_eg_rng": (hi - lo) / p0, "sd_eg_rngw": (hit - lot) / p0})
        if np.isfinite(sp):
            out["sd_eg_span_px"] = sp / p0
            out["sd_eg_px_span"] = p0 / sp
        w = sd.get("zzeg.c.rEma.wgt", np.nan)
        if np.isfinite(w) and w > 0:
            em = sd.get("zzeg.c.rEma.sum", np.nan) / w
            out["sd_eg_ema_px"] = em / p0
            out["sd_eg_px_ema"] = p0 / em
        for k, nm in (("zzxu.last", "last"), ("zzxo.r.29", "xo29"), ("zzxo.r.0", "xo0"), ("zzui.r.9", "ui9"), ("zzpk.high", "pkhi"),
                      ("zzpk.low", "pklo"), ("zzpk.open", "pkop"), ("zzmh.hm", "hm")):
            v = sd.get(k, np.nan)
            if np.isfinite(v) and v > 0:
                out[f"sd_lp_{nm}"] = np.log(p0 / v)
    return {k: np.asarray(v, float) for k, v in out.items()}


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
        # flows from snapshots alone (a chrono plugin only sees lv2: no trade stream, no BS flag): volume / amount growth
        # between snapshots, signed by the interval's average price against the previous mid (tick rule on a tie)
        dv, da = a["dvol"][ok], a["damt"][ok]
        pmid = np.concatenate([[np.nan], np.exp(lm)[:-1]])
        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.where(dv > 0, da / dv, np.nan)
            sd = np.sign(vw - pmid)
            sd = np.where(np.isfinite(sd) & (sd != 0), sd, np.sign(np.diff(lpx, prepend=np.nan)))
            sd = np.where(dv > 0, np.nan_to_num(sd), 0.0)
            sc = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in
                  (("v", dv), ("sv", dv * sd), ("a", da), ("sa", da * sd), ("n", (dv > 0).astype(float)), ("sn", sd))}
            sup = lambda x, k: sc[k][np.searchsorted(t, x, side="right")]  # noqa: E731 -- running total at times x
            for w in (10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900):
                win = {k: sup(T, k) - sup(T - w, k) for k in sc}
                out[f"simbv{w}"], out[f"simba{w}"], out[f"simbn{w}"] = win["sv"] / win["v"], win["sa"] / win["a"], win["sn"] / win["n"]
                out[f"slvol{w}"], out[f"slamt{w}"] = np.log1p(win["v"]), np.log1p(win["a"])
            out["scum_imb"] = sup(T, "sv") / sup(T, "v")
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
            # PM's state is EMAs (weight 1 per update, half-lives in trading-clock seconds): price over its EMA, log and
            # ratio (zzeg / zzup-like values near 1), on the mid and on the last price; the ratio of window means too
            from ic_kernel import ema_irregular

            mid_, lpx_ok = np.exp(lm), np.isfinite(lpx)
            dlm = np.concatenate([[np.nan], np.diff(lm)])  # snapshot-to-snapshot log return
            ems = {}
            for h in (5, 10, 15, 20, 25, 30, 40, 45, 50, 60, 75, 90, 100, 120, 150, 200, 240, 300, 450, 600, 900, 1200, 1800, 2400, 3600, 5400, 7200, 10800, 14400):
                em = ems[h] = at(T, np.log(ema_irregular(t, mid_, float(h))))
                out[f"lpe{h}"], out[f"rpe{h}"] = lm0 - em, np.exp(lm0 - em)
                out[f"lmie{h}"] = li0 - at(T, np.log(ema_irregular(t, np.exp(lmi), float(h))))  # microprice over its EMA
                out[f"rete{h}"] = at(T, ema_irregular(t, dlm, float(h)))  # EMA of returns (zzbd.rEma-like)
                if lpx_ok.sum() > 2:
                    ep = FI.at_marks(t[lpx_ok], np.log(ema_irregular(t[lpx_ok], np.exp(lpx[lpx_ok]), float(h))), T)
                    out[f"lppe{h}"] = lp0 - ep
            for w in (60, 300, 1800):
                out[f"rpma{w}"] = np.exp(out[f"lpma{w}"])
            hs = (30, 60, 120, 300, 600, 1200, 1800, 3600, 7200, 14400)
            for i, ha in enumerate(hs):  # fast EMA over slow EMA (log, MACD-like): lpe_a - lpe_b as one column
                for hb in hs[i + 1:]:
                    out[f"lee{ha}_{hb}"] = ems[ha] - ems[hb]
            from ic_kernel import ema_seeded

            pc = gf["pc"].to_numpy().astype(float)[ok] if "pc" in gf.columns else np.full(t.size, np.nan)
            pc0 = pc[np.isfinite(pc) & (pc > 0)]
            if pc0.size and t.size > 2:  # an EMA carried over from yesterday: starts at the open from pre-close at steady-state weight
                dt = float(np.median(np.diff(t))) if t.size > 10 else 3.0
                for h in (600, 900, 1200, 1800, 2400, 3600, 7200, 14400, 28800):
                    w0 = 1.0 / (1.0 - 2.0 ** (-max(dt, 0.1) / h))  # an EMA's weight after many updates every dt
                    out[f"lpes{h}"] = lm0 - at(T, np.log(ema_seeded(t, mid_, float(h), float(pc0[0]), w0, 0.0)))
            rv = out.get("rv900")
            if rv is not None:
                for h in (30, 60, 300):  # returns in units of the 15-minute realised volatility, clipped like PM's +-10 features
                    out[f"zr{h}"] = np.clip(out[f"ret{h}"] / (rv * np.sqrt(h / 900)), -10, 10)
                for h in (60, 200, 600, 1800, 3600, 14400):  # price over its EMA in units of the 15-minute volatility
                    out[f"lpev{h}"] = np.clip(out[f"lpe{h}"] / rv, -10, 10)
            tw = np.where(t > 7200, t + 5400, t)  # wall clock: the lunch break back in (t is the trading clock)
            ix_ = np.arange(t.size, dtype=float)  # per snapshot: a classic EMA counted in updates, not seconds
            for h in (60, 200, 600, 1800, 3600, 14400):
                out[f"lpew{h}"] = lm0 - at(T, np.log(ema_irregular(tw, mid_, float(h))))
            for n in (10, 20, 50, 100, 200, 600, 1200, 2400):
                out[f"lpek{n}"] = lm0 - at(T, np.log(ema_irregular(ix_, mid_, float(n))))
            for h in (60, 200, 600, 1800, 3600, 14400):  # Bollinger-style z: (log mid - its EMA) / EMA std of log mid
                m1 = ema_irregular(t, lm, float(h))
                sd = np.sqrt(np.maximum(ema_irregular(t, lm * lm, float(h)) - m1 * m1, 0.0))
                out[f"zpe{h}"] = np.clip(at(T, (lm - m1) / np.where(sd > 1e-9, sd, np.nan)), -10, 10)
    if gs is not None and gs.height > 2:
        b = E.arrays(gs, "slow")
        t = b["t_local"]
        cont = gs["s"].to_numpy() >= 34200
        cum = {k: np.concatenate([[0.0], np.cumsum(v * cont)]) for k, v in
               (("v", b["vol"]), ("nv", b["vol"] * b["side"]), ("n", np.ones(t.size)), ("nn", b["side"]), ("a", b["amt"]), ("na", b["amt"] * b["side"]))}
        upto = lambda x, k: cum[k][np.searchsorted(t, x, side="right")]  # noqa: E731 -- running total at times x
        with np.errstate(divide="ignore", invalid="ignore"):
            for w in (10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900, 1800, 3600):
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
            out["lcum_vol"], out["lcum_amt"], out["lcum_n"] = np.log1p(upto(T, "v")), np.log1p(upto(T, "a")), np.log1p(upto(T, "n"))
            out["cum_imba"], out["cum_imbn"] = upto(T, "na") / upto(T, "a"), upto(T, "nn") / upto(T, "n")
            px = FI.at_marks(t, b["price"], T)
            out["lp_vwap"] = np.log(px / (upto(T, "a") / upto(T, "v")))
            hi, lo = np.fmax.accumulate(np.where(cont, b["price"], np.nan)), np.fmin.accumulate(np.where(cont, b["price"], np.nan))
            out["stoch"] = (px - FI.at_marks(t, lo, T)) / (FI.at_marks(t, hi, T) - FI.at_marks(t, lo, T))
    from lit_factors import literature  # textbook factors, self-describing names (lit_factors.DESCRIBE)

    out.update(literature(gf, gs, T))
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


BUILD_REV = 3  # bump when main()'s per-day build loop changes: every cached build is then rebuilt


def build_dir(args, d: str, state: Path, target: Path | None) -> Path | None:
    """Where one day's build is cached: keyed by the build code, the build arguments and the input files' size / mtime.
    Market data in the catalog is not in the key: --rebuild when it is in doubt."""
    if not args.cache or not args.cache_dir.parent.parent.exists():
        return None
    import ced_match
    import same_gap

    def stamp(f: Path | None) -> tuple:
        return (str(f), f.stat().st_size, f.stat().st_mtime_ns) if f is not None and f.exists() else (str(f),)

    import ic_kernel
    import lit_factors

    import tick_flow

    mods = [B, E, FI, same_gap, ic_kernel, lit_factors] + ([ced_match] if args.ced else []) + ([tick_flow] if args.order_flow else [])
    code = [inspect.getsource(m) for m in mods] + [inspect.getsource(fn) for fn in (generic, index_relative, index_series, pm_daily, profile_marks, seed_state, seed_feats)]
    parts = [BUILD_REV, HORIZON, MARKS.tolist(), args.profiles, str(args.hstats), str(args.pm_cache), str(args.pm_stsfile), args.pm_sts_day, STS_MAP.get(d), str(args.seed_dir), args.seed_map, args.ced,
             str(args.ced_xr), args.target_stale, stamp(args.engine_dir / f"marks_{d}.parquet"), args.order_flow]
    if args.full_book is not None:  # whole-book factors: their file is part of the build
        parts.append(stamp(args.full_book / f"date={d}.parquet"))
    if str(B.CATALOG) != "/work/crucible_data/catalog.duckdb":  # another market-data source (CRUCIBLE_SOURCE): its own builds
        parts.append(str(B.CATALOG))
    key = hashlib.sha256(repr(code + parts).encode()).hexdigest()[:16]
    day = hashlib.sha256(repr((stamp(state), stamp(target))).encode()).hexdigest()[:8]
    return args.cache_dir / key / f"date={d}.{day}"


def pm_daily(root: Path, d: str, sids: list[int], rule: str = "before", pick: str | None = None) -> pl.DataFrame | None:
    """PM's own morning inputs for day d (/tmp/s/stsfile): stats/sod msr / mb / vr, index weights (SH50 HS300 A500 ZZ500
    ZZ1000 ZZ2000) and the exchange dummies. Which file: ``pick`` (a day, e.g. the one 224's log shows it loaded) -> the latest
    on or before it; else rule ``before`` = the latest before d, ``on`` = the latest on or before d."""
    import xarray as xr

    def latest(sub: str, prefix: str = "") -> Path | None:
        lim = pick or d
        ok = (lambda x: x <= lim) if (pick or rule == "on") else (lambda x: x < lim)
        fs = sorted(p for p in (root / sub).glob(f"{prefix}2*.xr") if ok(p.stem.removeprefix(prefix)))
        return fs[-1] if fs else None

    out = pl.DataFrame({"sid": sids}, schema={"sid": pl.Int64})
    found = False
    for sub, prefix, tag in (("stats/sod", "", "pm"), ("index", "index-", "pmw"), ("ex", "ex-", "pmx")):
        f = latest(sub, prefix)
        if f is None:
            continue
        print(f"{d}: stsfile {sub}/{f.name}")
        with xr.open_dataset(f) as ds:
            x = ds["x"].squeeze(drop=True)
            cols = {f"{tag}.{v}": x.sel(V=v).values.astype(float) for v in ds["V"].values}
            t = pl.DataFrame({"sid": np.array([int(v) for v in ds["S"].values], np.int64), **cols})
        out = out.join(t, on="sid", how="left")
        found = True
    return out if found else None


def family(expr: str) -> str:
    """Drop an expression's horizon / window digits and price flavour: -gen.ret30 and -gen.mret45 share one family."""
    return re.sub(r"\d+", "#", re.sub(r"gen\.[mp]?ret", "gen.ret", expr))


# quantity classes of gen.* bases, first full match wins: horizons, windows, price flavours and index tags are variants of one class
CORE_CLASSES = (
    ("ret", r"[mp]?ret\d*|rete\d*|zr\d*|xr\d*_\w+|ir\d*\w*"),
    ("ema_dev", r"lpp?e[swkv]?\d*|rpe\d*|zpe\d*|lee\d*|lmie\d*"),
    ("ma_dev", r"[lr]pma\d*"),
    ("ref_px", r"lp_\w+|xpc_\w+|ipc_\w+|lpd\w*|ld[hl]\d*|stoch\w*|spm\d*"),
    ("imb", r"s?imb\w*|timb\w*|s?cum_imb\w*|buy\w*"),
    ("volat", r"rv\d*|rng\d*"),
    ("activity", r"s?l?amt\d*|s?l?vol\d*|lcum_\w+|nam\d*|nn\d*|tick\w*|tsz\d*|vfrac\w*"),
)


def core(expr: str) -> str:
    """The quantity class an expression measures: sign, |.|, scale and every transform prefix dropped (--family-lock)."""
    m = re.search(r"\b(gen|ced)\.([A-Za-z_0-9.]+)", expr)
    if not m:  # a stateful stem's own atoms: the expression shape without the stem and its digits
        return "state:" + re.sub(r"\d+", "#", re.sub(r"\bzz[a-z0-9]+\.", "", re.sub(r"^(cal:|dm:|z:|s?rank:|crank:|-)+", "", expr)))
    src, b = m.group(1), m.group(2)
    if src == "ced":
        parts = b.split(".")
        return "ced." + (parts[0] if len(parts) > 1 else re.sub(r"\d+", "#", b))
    for name, rx in CORE_CLASSES:
        if re.fullmatch(rx, b):
            return name
    return "gen." + re.sub(r"\d+", "#", b)


def pm_family(feat: str) -> str:
    """PM's family of a model column: zzro2.a and zzro.e are both zzro."""
    return re.match(r"[a-z]+", feat)[0]


def model_files(root: Path, date: str, fits: list[str]) -> list[Path]:
    """Every sub-model file of the fits in the month folder load_models picks for the day."""
    months = sorted(p.name for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name) and p.name <= date[:6])
    return sorted(f for fit in fits for f in (root / months[-1] / "lgb" / fit).glob("model_s*.txt")) if months else []


def generation_cdf(files: list[Path], names: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Root-split training-CDF points of one model generation (pm_models_probe's parser), features with >= 8 exact points."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import pm_models_probe as PP  # noqa: PLC0415 -- LightGBM text parser

    roots = pl.concat([PP.parse_model(f)[3] for f in files])
    if names:  # Column_i -> the feature csv's name i
        roots = roots.with_columns(pl.col("feature").map_elements(
            lambda c: names[int(c[7:])] if c.startswith("Column_") and int(c[7:]) < len(names) else c, return_dtype=pl.String))
    out = {}
    for g in roots.filter(pl.col("nan_left") == 0).partition_by("feature"):  # NaN rows sit on the right: left share is exact
        if g.height >= 8:
            out[g["feature"][0]] = (g["t"].to_numpy(), g["p"].to_numpy())
    return out


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


XS = ("z:", "rank:", "crank:", "srank:")  # srank = rank in [-1, 1]; cross-sectional transforms within a mark: each has the same CDF whatever it transforms
DM = ("dm:", "dmi:", "neu:")  # neu: = residual on that day's Barra exposures (industries + styles), per mark  # minus the mark's cross-sectional mean / its Barra industry's mean: units kept (peer-relative, ZZQF)


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
    return out - 0.5 if pre == "crank:" else 2 * out - 1 if pre == "srank:" else out


CAL = "cal:"


def calibrate(v: np.ndarray, pts: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """v's pooled mid-rank ECDF mapped through PM's training CDF points (t, p): same order, PM's scale (clamped at the end points)."""
    if pts is None:
        return v
    o = np.argsort(pts[0], kind="mergesort")  # every point, tails included: they place the extreme thresholds
    t, p = pts[0][o], np.maximum.accumulate(pts[1][o])
    f = np.isfinite(v)
    out = np.full(v.size, np.nan)
    if f.sum() < 2 or t.size < 2:
        return out
    s = np.sort(v[f])
    u = (np.searchsorted(s, v[f], side="left") + np.searchsorted(s, v[f], side="right")) / (2.0 * s.size)
    out[f] = np.interp(u, p, t)
    return out


def neutralize(v: np.ndarray, E: np.ndarray) -> np.ndarray:
    """v minus its least-squares fit on [1, E] over the rows where v and every exposure are finite (NaN elsewhere)."""
    out = np.full(v.size, np.nan)
    if E.size == 0:
        return out
    f = np.isfinite(v) & np.isfinite(E).all(axis=1)
    k = E.shape[1] + 1
    if f.sum() < k + 10:
        return out
    X = np.column_stack([np.ones(f.sum()), E[f]])
    b, *_ = np.linalg.lstsq(X, v[f], rcond=None)
    out[f] = v[f] - X @ b
    return out


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

    def __init__(self, xs_on: bool, neu_on: bool = False) -> None:
        self.xs_on = xs_on
        self.neu_on = neu_on
        self.expo: list[np.ndarray] = []  # per block: each CED row's Barra exposures (industries + styles), for neu:
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
        ek = sorted(k for k in dcol if k.startswith("ced.exp.") and np.nanstd(dcol[k]) > 0)
        self.expo.append(np.column_stack([dcol[k] for k in ek]) if ek else np.zeros((dsid.size, 0)))
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
                out += [f"{p}{s}{c}" for p in DM if p != "neu:" or self.neu_on for s in ("", "-")]
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
        if pre == "neu:":
            i = self.didx[b][m]
            E = np.where((i >= 0)[:, None], self.expo[b][np.clip(i, 0, None)], np.nan) if self.expo[b].shape[1] else np.zeros((v.size, 0))
            return neutralize(v, E)
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


_SEG: dict[int, tuple] = {}  # id(groups) -> (groups, order, starts, ends): the grouping is sorted once per array


def segments(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(order, starts, ends) of ``groups``: rows sorted by group (stable) and each group's slice in that order."""
    hit = _SEG.get(id(groups))
    if hit is None or hit[0] is not groups:
        order = np.argsort(groups, kind="stable")
        g = groups[order]
        cuts = np.flatnonzero(np.diff(g)) + 1
        hit = (groups, order.astype(np.int64), np.concatenate([[0], cuts]).astype(np.int64), np.concatenate([cuts, [g.size]]).astype(np.int64))
        _SEG[id(groups)] = hit
    return hit[1], hit[2], hit[3]


def ic_vec(pred: np.ndarray, ret: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Cross-sectional Spearman IC per group (mark), NaN where it cannot be computed (compiled, parallel over groups)."""
    from ic_kernel import seg_spearman

    order, st, en = segments(groups)
    return seg_spearman(np.ascontiguousarray(pred[order], np.float64), np.ascontiguousarray(ret[order], np.float64), st, en, 20)


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
    ap.add_argument("--state-fallback", action="store_true",
                    help="a day with no state file of its own takes the stock list and stop time from the latest earlier one")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--seed-dir", type=Path, help="dir of PM .jz files: yesterday's carried state as gen.sd_* candidates (zzeg channel ...)")
    ap.add_argument("--seed-map", default="", help="--seed-dir: D=YYYYMMDD,... the file day D started from (default: the latest before D)")
    ap.add_argument("--sub-by-time", default="", help='sub-model per time of day, "s1:600,s2:1800,s3:12600,s4:14220" = segment ends '
                    "on the trading clock (s); the last runs to the close; overrides --subs")
    ap.add_argument("--subs", default="s1", help="sub-models to average (s1,s2,s3,s4 for all; each costs a predict per candidate)")
    ap.add_argument("--assign", type=Path, help="json feature -> expression to start from (default: KNOWN)")
    ap.add_argument("--shortlist", type=int, default=10)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--siblings", action="store_true",
                    help="also try every gate-passing candidate of the families its stem siblings are assigned (same formula, other parameters)")
    ap.add_argument("--gate-against", type=Path, default=None,
                    help="an assignment to gate this run's against, on this run's days: each column that differs keeps the new formula "
                         "only when it beats the old one (paired over marks, t >= --gate-t); writes ic_gate.parquet + ic_assign_gated.json")
    ap.add_argument("--gate-t", type=float, default=1.0)
    ap.add_argument("--cdf-by-generation", action=argparse.BooleanOptionalAction, default=True,
                    help="training-CDF points from each day's own model files (cal: per generation, gates on the run's generations); "
                         "--no-cdf-by-generation reads --cdf")
    ap.add_argument("--family-lock-soft", action="store_true",
                    help="with --family-lock: off-class formulas are rescanned in class but kept unless beaten, and the gate treats them as any change")
    ap.add_argument("--family-lock", action="store_true",
                    help="every column of one PM family (zzro, zzro2 ...) uses one quantity class (core): off-class start picks are dropped "
                         "and rescanned, the scan and the gate only take in-class formulas")
    ap.add_argument("--dump-pred", action="store_true", help="write <out>/pred.parquet: date, sid, mark, our prediction, PM's, per row")
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
    ap.add_argument("--sid-mod", default="", help="n:k = score the IC only over stocks with sid %% n == k; features still use every stock (a stock split for out-of-sample checks)")
    ap.add_argument("--neutral", action="store_true", help="also neu:<base>: each tick base's residual on that day's Barra industry + style exposures, per mark")
    ap.add_argument("--xs", action=argparse.BooleanOptionalAction, default=True,
                    help="also each stateless candidate cross-sectionally z-scored / ranked within its mark")
    ap.add_argument("--max-worsen", type=float, default=None,
                    help="a feature with a pick only moves to a candidate whose training-CDF gap is at most this much worse (off by default)")
    ap.add_argument("--xs-shortlist", type=int, default=6,
                    help="z / rank candidates look alike to the CDF: try the gate-passing ones of this many bases with the largest own |IC|")
    ap.add_argument("--full-book", type=Path, default=None,
                    help="full_book.py output dir (date=D.parquet on the same --marks): whole-book factors as gen.<name>")
    ap.add_argument("--engine-dir", type=Path, default=Path("data/engine"), help="marks_<D>.parquet from run_cpp.py --marks <the same grid>")
    ap.add_argument("--marks", default="five", help="five | rand:N[:seed] | a,b,...: the times scored (run_cpp.py --marks must match)")
    ap.add_argument("--target", type=Path, help="pm_signals.py output: score against PM's own live prediction (the latest signal of each "
                                                "stock at the mark) instead of the forward return; days without a file are skipped")
    ap.add_argument("--pm-stsfile", type=Path, default=Path("/work/crucible_data/pm_stsfile"),
                    help="PM's morning inputs (rsync of 2.23:/data/share/s): msr / mb / vr, index weights, exchange dummies")
    ap.add_argument("--pm-sts-day", choices=("before", "on"), default="before",
                    help="stsfile day: the latest before the trading day (before) or on or before it (on)")
    ap.add_argument("--pm-sts-map", type=Path, help="json {day: stsfile day}: the day 224's PM loaded (sts_days.py), overrides --pm-sts-day")
    ap.add_argument("--target-stale", type=float, default=10.0, help="--target: a signal older than this many seconds at the mark is no value")
    ap.add_argument("--implied", type=int, default=0, help="--target only: invert PM's model for this many open features (by gain)")
    ap.add_argument("--implied-rows", type=int, default=30000, help="rows sampled for --implied")
    ap.add_argument("--carriers", type=int, default=6, help="--implied: this many top residual groups (every scale of each) tried in each open column")
    ap.add_argument("--freeze-strong", action=argparse.BooleanOptionalAction, default=True,
                    help="skip features whose current formula was graded strong in <out>/ic_evidence.parquet")
    ap.add_argument("--profiles", default="ZZWA,ZZAL,ZZQO,ZZXC,ZZTS", help="samplerS families whose cells join the pool ('' = none)")
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--pm-cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--triples", action=argparse.BooleanOptionalAction, default=True, help="state candidates (x - y) / z")
    ap.add_argument("--min-distinct", type=int, default=30, help="fewer distinct values = a dummy: only for flag / count features")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage6"))
    ap.add_argument("--order-flow", action=argparse.BooleanOptionalAction, default=True,
                    help="load the order stream: gen.flow_* (tick_flow.flow) and lit_factors.order_stream candidates")
    ap.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="read / write each day's build under --cache-dir")
    ap.add_argument("--cache-dir", type=Path, default=Path("/work/crucible_data/cache/ic_build"))
    ap.add_argument("--rebuild", action="store_true", help="build every day again (market data in doubt) and overwrite the cache")
    ap.add_argument("--override", default="", help="feat=expr;feat=expr: replace these assigned formulas (each alone and together reported)")
    ap.add_argument("--scan-include", default="", help="regex: candidates the scan always tries on top of its shortlist")
    ap.add_argument("--scan-feats", default="", help="scan these features (comma-separated) instead of the top --max-features")
    ap.add_argument("--scan-match", default="", help="also scan every feature whose name matches this regex")
    ap.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=False,
                    help="scan: also every shortlisted candidate quantile-mapped onto PM's training CDF (cal:<expr>)")
    ap.add_argument("--scale-report", default="", help="regex: print PM's training CDF vs our values for matching assigned columns")
    ap.add_argument("--drop-harmful", action=argparse.BooleanOptionalAction, default=True,
                    help="at the start, drop the assigned formulas the IC rises without (each alone and all together)")
    ap.add_argument("--implied-feats", default="", help="--implied: only these features (comma-separated)")
    ap.add_argument("--carrier-assigned", action=argparse.BooleanOptionalAction, default=True,
                    help="--implied: also try the residual candidates in every assigned column, strong ones included")
    args = ap.parse_args()
    global MARKS
    MARKS = mark_grid(args.marks)
    if args.pm_sts_map:
        STS_MAP.update({str(k): str(v) for k, v in json.loads(args.pm_sts_map.read_text()).items()})
        print(f"stsfile days from {args.pm_sts_map}: {STS_MAP}")
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
    pool = Pool(args.xs, args.neutral)
    state_stems: set[str] = set()
    stem_parts: list[list[dict[str, pl.DataFrame]]] = []  # per block, per mark: stem -> its state rows
    empty_st = pl.DataFrame(schema={"sid": pl.Int64, "key": pl.String, "pm": pl.Float64})
    blocks = []  # per day: (models, frames per mark, sids per mark, forward returns per mark)
    gen_cdf: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}  # model generation -> its training-CDF points
    block_gen: list[str] = []  # per day: its model generation ("" = none found)
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists() and args.state_fallback:  # the state file only gives the stock list and the stop time: an earlier day's will do
            prev = sorted(p for p in args.dcp_dir.glob("2*.jz") if p.stem < d)
            if prev:
                f = prev[-1]
                print(f"{d}: no state file, stock list and stop time from {f.name}")
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        tf = args.target / f"date={d}.parquet" if args.target else None
        if tf is not None and not tf.exists():
            print(f"skip {d}: no PM signals {tf}")
            continue
        t_day = time.perf_counter()
        cdir = build_dir(args, d, f, tf)
        if cdir is not None and not args.rebuild and (cdir / "long.parquet").exists():  # this day's build, unchanged inputs: read it back
            long = pl.read_parquet(cdir / "long.parquet")
            dc = pl.read_parquet(cdir / "dc.parquet") if (cdir / "dc.parquet").exists() else None
            rets = {int(g["sid"][0]): g["ret"].to_numpy() for g in pl.read_parquet(cdir / "rets.parquet").partition_by("sid", maintain_order=True)}
            eng, prof = {}, {}
            print(f"{d}: build read from cache {cdir.parent.name}")
        else:
            tgt: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            if args.target:
                for g in pl.read_parquet(tf, columns=["sid", "t", "prediction"]).sort("sid", "t").partition_by("sid"):
                    tgt[int(g["sid"][0])] = (g["t"].to_numpy().astype(float), g["prediction"].to_numpy().astype(float))
            pm = B.load_state(f)
            tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
            stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
            sids = sorted(pm["sid"].unique().to_list())
            up = E.load(con, int(d), sids, stop, extra=True)
            ix = index_series(con, int(d), stop)
            print(f"{d}: index snapshots for {sorted(ix) or 'none (decode with KINDS=index research/backfill.sh)'}")
            fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
            slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
            if args.order_flow:
                import tick_flow as TF

                orders = {int(g["sid"][0]): g for g in TF.load_orders(con, int(d), sids, stop).partition_by("sid")}
            dc = CM.daily_candidates(d, sids) if CM is not None else None
            dbase = {r["sid"]: r for r in dc.select("sid", "adv20", "avgvol20", "freeshare").fill_null(np.nan).iter_rows(named=True)} if dc is not None else {}
            lca_c = [c for c in (dc.columns if dc is not None else []) if c.startswith("lca")]  # multi-day close EMAs over yesterday's close (log)
            danch = {r["sid"]: r for r in dc.select("sid", *lca_c).fill_null(np.nan).iter_rows(named=True)} if lca_c else {}
            pmd = pm_daily(args.pm_stsfile, d, sids, args.pm_sts_day, STS_MAP.get(d)) if args.pm_stsfile.exists() else None
            seed = seed_state(args, d)
            pmsm = {}
            if pmd is not None:  # PM's daily inputs join the CED table as ced.pm.* / ced.pmw.* / ced.pmx.*
                dc = pmd if dc is None else dc.join(pmd.with_columns(pl.col("sid").cast(dc.schema["sid"])), on="sid", how="left")
                if "pm.msr" in pmd.columns:
                    pmsm = {int(r[0]): (float(r[1]), float(r[2])) for r in pmd.select("sid", "pm.msr", "pm.mb").iter_rows()
                            if r[1] is not None and r[2] is not None and r[1] > 0 and r[2] > 0}  # stocks PM has no stats for: no ratio
                print(f"{d}: PM daily inputs {[c for c in pmd.columns if c != 'sid']}")
            eng: dict[int, dict[str, np.ndarray]] = {}
            ef = args.engine_dir / f"marks_{d}.parquet"
            if ef.exists():  # the engine's state at the marks (every solved rule) over the Python rebuild's
                em = pl.read_parquet(ef).filter(pl.col("mark").is_in(MARKS.tolist())).sort("sid", "mark")
                keep = [c for c in em.columns if c not in ("sid", "mark") and "|" not in c]
                for g in em.partition_by("sid"):
                    if g.height == MARKS.size:
                        eng[int(g["sid"][0])] = {c: g[c].to_numpy() for c in keep}
            fbk: dict[int, dict[str, np.ndarray]] = {}
            fbf = args.full_book / f"date={d}.parquet" if args.full_book is not None else None
            if fbf is not None and fbf.exists():  # whole-book factors from our replay (full_book.py), on this mark grid only
                fm = pl.read_parquet(fbf).filter(pl.col("mark").is_in(MARKS.tolist())).sort("sid", "mark")
                for g in fm.partition_by("sid"):
                    if g.height == MARKS.size:
                        fbk[int(g["sid"][0])] = {c: g[c].to_numpy() for c in fm.columns if c not in ("sid", "mark")}
                print(f"{d}: full-book factors for {len(fbk)} stocks from {fbf}")
            elif fbf is not None:
                print(f"{d}: no {fbf}: no full-book factors")
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
                if args.order_flow:  # order stream as tick bases: tick_flow as gen.flow_<name>, lit_factors.order_stream under its own names
                    from lit_factors import order_stream

                    go = orders.get(sid)
                    gen.update({f"flow_{k}": v for k, v in TF.flow(go, gs, gf, MARKS).items()})
                    gen.update(order_stream(go, gf, gs, MARKS))
                gen.update(fbk.get(sid, {}))
                gen.update(index_relative(gen, ix, sid, MARKS))
                if sid in danch and "lp_pc" in gen:  # price over a multi-day EMA of closes (an EMA state carried across days)
                    for c in lca_c:
                        gen[f"lpd{c[3:]}"] = gen["lp_pc"] - float(danch[sid][c])
                if sid in pmsm and gf is not None and gf.height > 2:  # spread and depth over PM's own daily normalisers
                    msr, mb = pmsm[sid]
                    af = E.arrays(gf, "fast")
                    dep = FI.at_marks(af["t_local"], (gf["bv1"].to_numpy().astype(float) + gf["av1"].to_numpy().astype(float)) / 2, MARKS)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        add(sid, "ced.r.sp_msr", gen["spread"] / (2 * msr))
                        for w in (60, 300, 900):
                            if f"spm{w}" in gen:
                                add(sid, f"ced.r.spm{w}_msr", gen[f"spm{w}"] / (2 * msr))
                        add(sid, "ced.r.ldep_mb", np.log(dep / mb))
                        add(sid, "ced.r.dep_mb", dep / mb)
                if sid in seed and gf is not None and gf.height > 2:  # yesterday's carried state (zzeg channel, closes) against today's price
                    gen.update(seed_feats(seed[sid], E.arrays(gf, "fast")))
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
            if cdir is not None:  # written whole or not at all: a half-written day is never read back
                tmp = cdir.with_name(cdir.name + ".tmp")
                tmp.mkdir(parents=True, exist_ok=True)
                long.write_parquet(tmp / "long.parquet")
                if dc is not None:
                    dc.write_parquet(tmp / "dc.parquet")
                pl.DataFrame({"sid": np.repeat(np.array(list(rets), np.int64), MARKS.size),
                              "ret": np.concatenate(list(rets.values())) if rets else np.zeros(0)}).write_parquet(tmp / "rets.parquet")
                if cdir.exists():
                    shutil.rmtree(cdir)
                tmp.rename(cdir)
        by_mark = long.partition_by("mark", as_dict=True, include_key=False)  # one pass instead of a filter per mark
        empty = long.drop("mark").clear()
        frames = [by_mark.get((m,), empty) for m in MARKS]
        st_ = pl.col("key").str.extract(r"^([^.]+)\.", 1)
        stem_parts.append([{k[0]: v for k, v in fr.filter(~pl.col("key").str.starts_with("gen.") & ~pl.col("key").str.starts_with("ced."))
                            .with_columns(st_.alias("_st")).partition_by("_st", as_dict=True, include_key=False).items()}
                           for fr in frames])  # per mark, the state rows split by stem once (not one full-frame filter per stem)
        pool.add_block(frames, dc)
        state_stems |= set(long["key"].str.extract(r"^([^.]+)\.", 1).drop_nulls().unique().to_list()) - {"gen", "ced"}
        models = (load_seg_models(args.models, d, args.fits.split(","), args.sub_by_time) if args.sub_by_time
                  else load_models(args.models, d, args.fits.split(","), args.subs.split(",")))
        blocks.append((d, models, frames, rets))
        if args.cdf_by_generation:  # identical model files = one generation (202510..202604 reuse 202509's models)
            mf = model_files(args.models, d, args.fits.split(","))
            key = hashlib.md5(b"".join(hashlib.md5(f.read_bytes()).digest() for f in mf)).hexdigest()[:8] if mf else ""
            if key and key not in gen_cdf:
                gen_cdf[key] = generation_cdf(mf, names)
                print(f"{d}: model generation {key} ({mf[0].parent.parent.parent.name}): training-CDF points for {len(gen_cdf[key])} features")
            block_gen.append(key)
        print(f"{d}: {len(rets)} stocks x {MARKS.size} marks, {len(models)} models, engine state for {len(eng)}, "
              f"profiles for {len(prof)} ({time.perf_counter() - t0:.0f}s)")
        print(f"time build_{d} {time.perf_counter() - t_day:.0f}s")
    if not blocks:
        return
    print(f"time build {time.perf_counter() - t0:.0f}s")
    t_st = time.perf_counter()
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
    global ROW_T
    ROW_T = MARKS[mk].astype(float) if mk.size else np.zeros(0)
    if args.sid_mod:  # a stock split: features over the whole cross-section, the IC only over sid % n == k (others' targets NaN)
        n_, k_ = (int(x) for x in args.sid_mod.split(":"))
        ret = np.where(sid_a % n_ == k_, ret, np.nan)
        print(f"--sid-mod {args.sid_mod}: IC over {int(np.isfinite(ret).sum()):,} of {ret.size:,} panel rows")
    X = np.full((sid_a.size, len(names)), np.nan)
    pos = {(b, m, s): i for i, (b, m, s) in enumerate(zip(blk, mk, sid_a, strict=True))}
    cache: dict[str, list] = {}  # stateful stem -> per block, per mark: (sids, candidates)

    def rows_of(b: int, m: int, sids: np.ndarray) -> np.ndarray:
        """Panel row of each stock at (block, mark), -1 when it has no forward return."""
        return np.array([pos.get((b, m, int(s)), -1) for s in sids], int)

    pidx = [[rows_of(b, m, s) for m, s in enumerate(per)] for b, per in enumerate(pool.sids)]  # the pool's stocks -> panel rows

    def stem_cache(stem: str) -> list:
        if stem not in cache:
            def part(p: dict) -> pl.DataFrame:  # the stem's rows, plus zzxu.last (FM.atoms takes the price from it)
                own = p.get(stem, empty_st)
                if stem == "zzxu" or "zzxu" not in p:
                    return own
                return pl.concat([own, p["zzxu"].filter(pl.col("key") == "zzxu.last")])

            cache[stem] = [[(s, c, rows_of(b, m, s)) for m, (s, c) in enumerate(stem_frames([part(p) for p in stem_parts[b]], stem))]
                           for b in range(len(blocks))]
        return cache[stem]

    pooled_cache: dict[str, dict[str, np.ndarray]] = {}

    def stem_pool(stem: str) -> dict[str, np.ndarray]:
        """Every candidate of a stem over days and marks, as finite samples (atoms pooled, then 30000 rows, then expressions)."""
        if stem not in state_stems:
            return pool_samples()
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

    import ic_kernel as K

    assert np.all(np.diff(grp) >= 0), "panel rows must be grouped by mark"
    _, seg_st, seg_en = segments(grp)
    ind_panel = np.full(sid_a.size, -1, np.int64)  # each panel row's Barra industry (for dmi:)
    for b in range(len(pool.sids)):
        for m in range(len(pool.sids[b])):
            ix, i = pidx[b][m], pool.didx[b][m]
            if pool.ind[b].size:
                iv = np.where(i >= 0, pool.ind[b][np.clip(i, 0, None)], -1)
                ok = ix >= 0
                ind_panel[ix[ok]] = iv[ok]
    n_ind = int(ind_panel.max()) + 1 if (ind_panel >= 0).any() else 1
    n_exp = max((e.shape[1] for e in pool.expo), default=0)
    exp_panel = np.full((sid_a.size, n_exp), np.nan)  # each panel row's Barra exposures (for neu:), the day's own file
    if n_exp:
        for b in range(len(pool.sids)):
            if pool.expo[b].shape[1] != n_exp:
                continue
            for m in range(len(pool.sids[b])):
                ix, i = pidx[b][m], pool.didx[b][m]
                ok = (ix >= 0) & (i >= 0)
                exp_panel[ix[ok]] = pool.expo[b][i[ok]]
    zeros_sub = np.zeros(sid_a.size, np.int64)
    base_vec: dict[str, np.ndarray | None] = {}

    def pool_base(base: str) -> np.ndarray | None:
        """A pool base laid out on the panel rows (once per base)."""
        if base not in base_vec:
            v, seen = np.full(sid_a.size, np.nan), False
            for b in range(len(pool.sids)):
                for m in range(len(pool.sids[b])):
                    x = pool.raw(b, m, base)
                    if x is not None:
                        ix = pidx[b][m]
                        ok = ix >= 0
                        v[ix[ok]] = x[ok]
                        seen = True
            base_vec[base] = v.astype(np.float32) if seen else None  # float32 halves the memory of ~1000 cached bases
        b_ = base_vec[base]
        return None if b_ is None else b_.astype(np.float64)

    def pool_col(expr: str) -> np.ndarray | None:
        """A pool candidate over the whole panel at once: sign / abs / bp unit, then the per-mark transform (compiled)."""
        pre, neg, ab, scale, base = parse(expr)
        v = pool_base(base)
        if v is None:
            return None
        v = np.abs(v) if ab else v.copy()
        v = (-v if neg else v) * scale
        if pre == "z:":
            return K.seg_z(v, seg_st, seg_en)
        if pre in ("rank:", "crank:"):
            return K.seg_rank(v, seg_st, seg_en, pre == "crank:")
        if pre == "srank:":
            return 2 * K.seg_rank(v, seg_st, seg_en, False) - 1
        if pre == "dm:":
            return K.seg_demean(v, zeros_sub, seg_st, seg_en, 1)
        if pre == "dmi:":
            return K.seg_demean(v, ind_panel, seg_st, seg_en, n_ind)
        if pre == "neu:":
            out = np.full(v.size, np.nan)
            for a_, e_ in zip(seg_st, seg_en, strict=True):
                out[a_:e_] = neutralize(v[a_:e_], exp_panel[a_:e_])
            return out
        return v

    samples_cache: dict[str, np.ndarray] = {}

    def pool_samples(n: int = 20000, every: int = 3) -> dict[str, np.ndarray]:
        """Every pool candidate's finite values on every third mark, subsampled to n -- from the whole-panel columns
        (one compiled transform per candidate) instead of a Python loop over (candidate, day, mark)."""
        if not samples_cache:
            t1 = time.perf_counter()
            rows = np.flatnonzero(mk % every == 0)
            rng = np.random.default_rng(0)
            for k in pool.names():
                x = pool_col(k)
                if x is None:
                    continue
                v = x[rows]
                v = v[np.isfinite(v)]
                if v.size >= 50:
                    samples_cache[k] = rng.choice(v, n, replace=False) if v.size > n else v
            print(f"candidate samples: {len(samples_cache):,} candidates ({time.perf_counter() - t1:.0f}s)", flush=True)
        return samples_cache

    def column(feature: str, expr: str) -> np.ndarray:
        """One model column filled with expression ``expr`` of the feature's stem (``cal:`` = quantile-mapped onto PM's training CDF)."""
        if expr.startswith(CAL):
            raw = column(feature, expr[len(CAL):])
            if not gen_rows:
                return calibrate(raw, cdf.get(feature))
            out = np.full(raw.size, np.nan)
            for k, rows in gen_rows.items():  # each generation's days ranked among themselves, mapped through its own CDF
                out[rows] = calibrate(raw[rows], gen_cdf[k].get(feature))
            return out
        stem = feature.split(".")[0]
        if stem not in state_stems:
            x = pool_col(expr)
            return x if x is not None else np.full(sid_a.size, np.nan)
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
    gen_rows: dict[str, np.ndarray] = {}  # generation -> the panel rows of its days
    if args.cdf_by_generation and gen_cdf:
        for b, k in enumerate(block_gen):
            if k:
                gen_rows[k] = np.concatenate([gen_rows[k], brows[b]]) if k in gen_rows else brows[b]
        pooled: dict[str, list] = {}
        for k in gen_rows:  # the gates see the run's generations pooled (one generation: exactly its points)
            for f_, (t_, p_) in gen_cdf[k].items():
                pooled.setdefault(f_, []).append((t_, p_))
        cdf = {f_: (np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])) for f_, v in pooled.items()}
        print(f"training CDF by generation: {len(gen_rows)} generation(s) {sorted(gen_rows)}, {len(cdf)} features")
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

        def pred_with(self, colv: np.ndarray) -> np.ndarray:
            X[:, self.j] = colv
            if not flat:
                return panel(X)
            pred = self.pred.copy()
            for b, (_, ms, _, _) in enumerate(blocks):
                pred[brows[b]] += predict_rows(ms, X, brows[b], self.j) - self.old[b]
            return pred

        def score(self, colv: np.ndarray) -> np.ndarray:
            return ic_vec(self.pred_with(colv), ret, grp)

    assign = dict(KNOWN)
    if args.assign and args.assign.exists():
        assign = json.loads(args.assign.read_text())
    print(f"time panel_rows {time.perf_counter() - t_st:.0f}s")
    ovr = dict(kv.split("=", 1) for kv in args.override.split(";") if "=" in kv)
    ovr = {f.strip(): e.strip() for f, e in ovr.items() if f.strip() in col}
    col_t = []
    for feat, expr in list(assign.items()):
        if feat in col:
            t1 = time.perf_counter()
            X[:, col[feat]] = column(feat, expr)
            col_t.append((time.perf_counter() - t1, feat, expr))
            v = X[:, col[feat]]
            v = v[np.isfinite(v)]
            if feat in ovr:  # replaced below: no gate on the formula it replaces
                pass
            elif feat in cdf and v.size and (g := FM.cdf_gap(v, cdf[feat])) > args.drop_gap:  # a start pick off the training CDF is dropped
                print(f"drop {feat} = {expr}: training-CDF gap {g:.3f} > {args.drop_gap}")
                X[:, col[feat]] = np.nan
                del assign[feat]
            elif fp_kind.get(feat) not in ("flag", "count", "const") and np.unique(v).size < args.min_distinct:
                print(f"drop {feat} = {expr}: {np.unique(v).size} distinct values on a {fp_kind.get(feat)} feature")
                X[:, col[feat]] = np.nan
                del assign[feat]
    col_t.sort(reverse=True)
    print("time panel_slowest_columns " + ", ".join(f"{f}={e} {s:.0f}s" for s, f, e in col_t[:5]))
    if ovr:  # --override: each alone (the others as assigned), then all together; kept as the start (no CDF gate)
        p_a = panel(X)
        ic_a = summary(ic_vec(p_a, ret, grp))
        for f, e in ovr.items():
            kc = X[:, col[f]].copy()
            v1 = summary(Swap(col[f], p_a).score(column(f, e)))
            X[:, col[f]] = kc
            print(f"override {f}: {assign.get(f, '-')} -> {e}: IC {ic_a[0]:.4f} -> {v1[0]:.4f} alone")
        for f, e in ovr.items():
            X[:, col[f]] = column(f, e)
            assign[f] = e
        print(f"override all {len(ovr)}: IC {ic_a[0]:.4f} -> {summary(score(X))[0]:.4f}")
    lock: dict[str, str] = {}  # PM family -> the one quantity class its columns may use
    relock: list[str] = []
    if args.family_lock:  # anchor = the class with the most gain behind it (strong x3, fits x2 by the start's evidence); the rest is rescanned
        gsh = dict(zip(fp["feature"].to_list(), fp["gain_share"].to_list(), strict=False))
        evp = args.assign.parent / "ic_evidence.parquet" if args.assign else None
        grd = dict(pl.read_parquet(evp).select("feature", "grade").iter_rows()) if evp is not None and evp.exists() else {}
        wt: dict[str, Counter] = {}
        for f, e in assign.items():
            if f in col and e:
                wt.setdefault(pm_family(f), Counter())[core(e)] += gsh.get(f, 0.0) * {"strong": 3, "fits": 2}.get(grd.get(f), 1)
        for fam_, w in wt.items():
            lock[fam_] = w.most_common(1)[0][0]
            for f, e in list(assign.items()):
                if f in col and e and pm_family(f) == fam_ and core(e) != lock[fam_]:
                    if args.family_lock_soft:  # kept as the baseline: replaced only by an in-class formula that scores better
                        print(f"family lock {fam_} = {lock[fam_]}: rescan {f} = {e} ({core(e)}) in class, kept unless beaten")
                    else:
                        print(f"family lock {fam_} = {lock[fam_]}: drop {f} = {e} ({core(e)}), rescanned in class")
                        X[:, col[f]] = np.nan
                        del assign[f]
                    relock.append(f)
        print(f"family lock: {len(lock)} families anchored, {len(relock)} columns dropped to rescan: "
              + ", ".join(f"{k}={v}" for k, v in sorted(lock.items())))
    print(f"time panel {time.perf_counter() - t_st:.0f}s")
    t_st = time.perf_counter()
    pred0 = panel(X)
    v0 = ic_vec(pred0, ret, grp)
    base = summary(v0)
    ef0 = args.out / "ic_evidence.parquet"
    ev0 = pl.read_parquet(ef0) if ef0.exists() else None
    if args.drop_harmful:  # formulas whose removal raises the IC, each alone (t <= -2), on this run's data: dropped only if dropping them together helps too
        v0 = np.nan_to_num(v0, nan=0.0)
        bad = []
        for f in [f for f in assign if f in col]:
            kc = X[:, col[f]].copy()
            if summary(v0 - np.nan_to_num(Swap(col[f], pred0).score(np.full(sid_a.size, np.nan)), nan=0.0))[1] <= -args.min_t:
                bad.append(f)
            X[:, col[f]] = kc
        if bad:
            kept_cols = {f: X[:, col[f]].copy() for f in bad}
            for f in bad:
                X[:, col[f]] = np.nan
            b2 = summary(score(X))
            if b2[0] > base[0]:
                for f in bad:
                    del assign[f]
                print(f"dropped (IC rises without each; together {base[0]:.4f} -> {b2[0]:.4f}): {bad}")
                base = b2
            else:
                for f, v in kept_cols.items():
                    X[:, col[f]] = v
                print(f"kept {bad}: the IC rises without each alone, but together {base[0]:.4f} -> {b2[0]:.4f}")
    print(f"\nall-missing except the assignment {assign}: IC {base[0]:.4f} (t {base[1]:.2f}, {base[2]} marks) ({time.perf_counter() - t0:.0f}s)")
    print(f"time base {time.perf_counter() - t_st:.0f}s")
    t_st = time.perf_counter()
    log = []
    feats = [f for f in fp.sort("gain_share", descending=True)["feature"].to_list() if f in col][: args.max_features]
    if args.scan_feats:
        feats = [f for f in args.scan_feats.split(",") if f in col]
    if args.scan_match:
        base_f = feats if args.scan_feats else []  # without --scan-feats, feats is still the top --max-features list here
        feats = base_f + [f for f in names if f in col and re.search(args.scan_match, f) and f not in base_f]
        print(f"scan: {feats}")
    feats += [f for f in relock if f not in feats]  # --family-lock drops are always rescanned
    frozen: set[str] = set()
    if args.freeze_strong and ev0 is not None:  # a formula graded strong last time and still assigned is settled: not scanned again
        frozen = {f for f, e, g in ev0.select("feature", "formula", "grade").iter_rows() if g == "strong" and assign.get(f) == e}
        print(f"frozen (strong last round, not rescanned): {sorted(frozen)}")
    sorted_pools: dict[int, dict[str, np.ndarray]] = {}  # id(candidate pool) -> its samples sorted once
    ranked_cache: dict[str, list[dict]] = {}
    fp_rows = {r["feature"]: r for r in fp.iter_rows(named=True)}
    tm = {"rank": 0.0, "column": 0.0, "score": 0.0}

    def fast_rank(feat: str, cs: dict[str, np.ndarray]) -> list[dict]:
        """FM.rank on presorted samples: quantiles by index, in-range and the training-CDF gap by binary search."""
        f = fp_rows[feat]
        pre = "q" if f.get("q_med") is not None else "t"
        if f.get(f"{pre}_med") is None:
            return []
        tq = np.array([f[f"{pre}_p10"], f[f"{pre}_med"], f[f"{pre}_p90"]])
        span = max(tq[2] - tq[0], 1e-12)
        pts = FM.interior(cdf[feat]) if feat in cdf else None
        out = []
        for k, s in cs.items():
            n = s.size
            cq = s[[int(0.1 * (n - 1)), (n - 1) // 2, int(0.9 * (n - 1))]]
            inr = (np.searchsorted(s, f["hi"], side="right") - np.searchsorted(s, f["lo"], side="left")) / n if f["lo"] is not None else np.nan
            sc = float(np.mean(np.abs(np.searchsorted(s, pts[0], side="right") / n - pts[1]))) if pts is not None else float(np.abs(cq - tq).sum() / span)
            out.append({"candidate": k, "score": sc, "in_range": float(inr)})
        out.sort(key=lambda r: r["score"])
        return out

    cur_pred = panel(X)  # the full prediction with every assigned column, kept up to date column by column
    for ps in range(args.passes):
        for feat in feats:
            if feat in frozen:
                continue
            stem = feat.split(".")[0]
            cand_all = stem_pool(stem)
            if not cand_all:
                continue
            t1 = time.perf_counter()
            if id(cand_all) not in sorted_pools:
                sorted_pools[id(cand_all)] = {k: np.sort(v) for k, v in cand_all.items()}
            cs = sorted_pools[id(cand_all)]
            if feat not in ranked_cache:
                ranked_cache[feat] = fast_rank(feat, cs)
            ranked = ranked_cache[feat]
            tm["rank"] += time.perf_counter() - t1
            cut = args.max_cdf if feat in cdf else args.max_score
            # gate on the fingerprint: right units and range first, IC second -- IC alone rewards any useful signal in any column
            rk = {r["candidate"]: r for r in ranked}
            dummy_ok = fp_kind.get(feat) in ("flag", "count", "const")

            def varied(c: str, ok: bool = dummy_ok, cs_: dict = cs) -> bool:
                if ok or c not in cs_:
                    return True
                return 1 + int(np.count_nonzero(np.diff(cs_[c]))) >= args.min_distinct

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
            if args.siblings:  # the stem's siblings' formulas with other windows / levels: siblings often share one formula
                sib = {family(e) for g_, e in assign.items() if e and g_ != feat and g_.split(".")[0] == stem}
                short = list(dict.fromkeys(short + [c for c in passes if family(c) in sib and c != assign.get(feat)]))
            if args.scan_include:  # every gate-passing candidate the pattern names (other families the scan should try too)
                short = list(dict.fromkeys(short + [c for c in passes if re.search(args.scan_include, c) and c != assign.get(feat)]))
            fam_ = pm_family(feat)
            if fam_ in lock:  # --family-lock: in-class candidates only, topped up with the class's best-fitting ones
                in_cls = lambda c, k=lock[fam_]: core(c) == k  # noqa: E731
                short = list(dict.fromkeys([c for c in short if in_cls(c)] + [c for c in passes if in_cls(c)][: args.shortlist]))
                passes = [c for c in passes if in_cls(c)]
            if args.calibrate and feat in cdf:  # the same candidates on PM's scale: right information, wrong units (zzzl2 a = 3-17)
                c0 = assign.get(feat)
                extra = [] if not c0 else [c0[len(CAL):]] if c0.startswith(CAL) else [CAL + c0]  # the pick itself, the other way
                short = [c for c in dict.fromkeys(short + [CAL + c for c in short if not c.startswith(CAL)] + extra) if c != c0]
            if not short:
                log.append({"pass": ps, "feature": feat, "kept": assign.get(feat), "ic_after": np.nan, "t_gain": 0.0,
                            "top": "no candidate passes the fingerprint gate; closest: "
                                   + "; ".join(f"{r['candidate']} (score {r['score']:.2f}, in {r['in_range']:.2f})" for r in ranked[:2])})
                print(f"pass {ps} {feat}: no candidate passes the fingerprint gate ({time.perf_counter() - t0:.0f}s)")
                continue
            cur = assign.get(feat)
            if args.max_worsen is not None and feat in cdf and cur in rk:  # IC may not trade a better-fitting formula for a worse one
                short = [c for c in short if c not in rk or rk[c]["score"] <= rk[cur]["score"] + args.max_worsen]
            j = col[feat]
            keep_col = X[:, j].copy()
            sw = Swap(j, cur_pred)
            cur_v = np.nan_to_num(ic_vec(sw.pred, ret, grp), nan=0.0)  # a constant prediction (everything missing) ranks nothing: IC 0
            best = (summary(cur_v)[0], cur, 0.0)
            tried = []
            for expr in short:
                t1 = time.perf_counter()
                cv = column(feat, expr)
                t2 = time.perf_counter()
                v = np.nan_to_num(sw.score(cv), nan=0.0)
                tm["column"] += t2 - t1
                tm["score"] += time.perf_counter() - t2
                dv = v - cur_v  # paired over marks: the same marks, the same other columns
                _, t_gain, _ = summary(dv)
                tried.append((expr, summary(v)[0], t_gain))
                if np.isfinite(t_gain) and t_gain >= args.min_t and summary(v)[0] > best[0]:
                    best = (summary(v)[0], expr, t_gain)
            # accept by the training CDF: the closest candidate within --accept-cdf, its sign picked by IC, unless IC drops significantly
            plain = [r for r in ranked if not r["candidate"].startswith(XS) and in_scale(r)  # a z / rank shape names no variable
                     and (fam_ not in lock or core(r["candidate"]) == lock[fam_])]
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
                cur_pred = sw.pred_with(column(feat, best[1]))
                if args.family_lock and fam_ not in lock:  # the family's first pick (scanned by gain) sets its class
                    lock[fam_] = core(best[1])
                    print(f"  family lock {fam_} = {lock[fam_]} (from {feat})")
            else:
                X[:, j] = keep_col
            tried.sort(key=lambda t: -t[2] if np.isfinite(t[2]) else 1e9)
            log.append({"pass": ps, "feature": feat, "kept": assign.get(feat), "ic_after": best[0], "t_gain": best[2],
                        "top": "; ".join(f"{e} {ic:.4f} (t {tg:.1f})" for e, ic, tg in tried[:3])})
            print(f"pass {ps} {feat}: kept {assign.get(feat)} IC {best[0]:.4f} t_gain {best[2]:.1f} ({time.perf_counter() - t0:.0f}s)")
        print(f"pass {ps} time: fingerprint ranking {tm['rank']:.0f}s, candidate columns {tm['column']:.0f}s, model + IC {tm['score']:.0f}s")
    print(f"time scan {time.perf_counter() - t_st:.0f}s")
    fam_cores: dict[str, Counter] = {}
    for f, e in assign.items():
        if f in col and e:
            fam_cores.setdefault(pm_family(f), Counter())[core(e)] += 1
    mixed = {k: v for k, v in fam_cores.items() if len(v) > 1}
    print(f"families with more than one quantity class: {len(mixed)} of {len(fam_cores)}"
          + "".join(f"\n  {k}: " + ", ".join(f"{c} x{n}" for c, n in v.most_common()) for k, v in sorted(mixed.items())))
    t_st = time.perf_counter()
    full_pred = panel(X)
    final = summary(ic_vec(full_pred, ret, grp))
    full_v = np.nan_to_num(ic_vec(full_pred, ret, grp), nan=0.0)
    evid = []
    prev = {r["feature"]: r for r in ev0.iter_rows(named=True)} if ev0 is not None else {}
    for feat, expr in assign.items():  # ablation: the IC lost when the column goes back to missing, and the training-CDF gap
        if feat not in col:
            continue
        j = col[feat]
        keep_col = X[:, j].copy()
        if feat in frozen and feat in prev:  # settled (strong, same formula): last round's ablation stands
            loss, t_loss = prev[feat]["ic_loss"], prev[feat]["t_loss"]
        else:
            loss, t_loss, _ = summary(full_v - np.nan_to_num(Swap(j, full_pred).score(np.full(sid_a.size, np.nan)), nan=0.0))
            X[:, j] = keep_col
        v = keep_col[np.isfinite(keep_col)]
        gap = FM.cdf_gap(v, cdf[feat]) if feat in cdf and v.size else np.nan
        e_ = expr.removeprefix(CAL)
        src = ("state" if feat.split(".")[0] in state_stems else "xs" if e_.startswith(XS)
               else e_.lstrip("-|").split(".")[0]) + (" cal" if expr.startswith(CAL) else "")
        grade = ("strong" if gap <= 0.02 and t_loss >= 2 else "fits" if gap <= args.accept_cdf else "weak") if np.isfinite(gap) \
            else ("ic" if t_loss >= 2 else "weak")
        evid.append({"feature": feat, "formula": expr, "source": src, "cdf_gap": gap, "ic_loss": loss, "t_loss": t_loss, "grade": grade})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ic_assign.json").write_text(json.dumps(assign, indent=1))
    if args.dump_pred:  # per row: our prediction with the final assignment and PM's, for residual breakdowns
        days_ = np.array([int(blocks[b][0]) for b in range(len(blocks))])
        pl.DataFrame({"date": days_[blk], "sid": sid_a, "mark": MARKS[mk], "ours": full_pred, "pm": ret}).write_parquet(args.out / "pred.parquet")
        print(f"rows -> {args.out / 'pred.parquet'}")
    if args.gate_against is not None:  # each changed column, new formula against the old one on these days
        gb = json.loads(args.gate_against.read_text())
        gated, grow = dict(gb), []
        for feat in sorted(set(assign) | set(gb)):
            new_e, old_e = assign.get(feat), gb.get(feat)
            if new_e == old_e or feat not in col:
                continue
            if args.family_lock and not args.family_lock_soft and old_e and pm_family(feat) in lock and core(old_e) != lock[pm_family(feat)]:  # off-class: never restored
                if new_e:
                    gated[feat] = new_e
                else:
                    gated.pop(feat, None)
                grow.append({"feature": feat, "old": old_e, "new": new_e, "gain": np.nan, "t": np.nan, "kept_new": True})
                continue
            j = col[feat]
            keep_col = X[:, j].copy()
            try:
                alt = column(feat, old_e) if old_e else np.full(sid_a.size, np.nan)
            except Exception:  # noqa: BLE001 -- an old formula this run's pool cannot build: compare with missing
                alt = np.full(sid_a.size, np.nan)
            v_alt = np.nan_to_num(Swap(j, full_pred).score(alt), nan=0.0)
            X[:, j] = keep_col
            gain, t_g, _ = summary(full_v - v_alt)
            ok = bool(np.isfinite(t_g) and t_g >= args.gate_t)
            if ok:
                if new_e:
                    gated[feat] = new_e
                else:
                    gated.pop(feat, None)
            grow.append({"feature": feat, "old": old_e, "new": new_e, "gain": gain, "t": t_g, "kept_new": ok})
        if grow:
            gt = pl.DataFrame(grow).sort("t", descending=True)
            gt.write_parquet(args.out / "ic_gate.parquet")
            with pl.Config(tbl_rows=120, tbl_cols=6, tbl_width_chars=230, fmt_str_lengths=70, float_precision=4):
                print(f"\n== gate against {args.gate_against}: {gt['kept_new'].sum()} of {gt.height} changes kept (t >= {args.gate_t})")
                print(gt)
        (args.out / "ic_assign_gated.json").write_text(json.dumps(gated, indent=1))
        print(f"gated: {len(gated)} columns -> {args.out / 'ic_assign_gated.json'}")
    if evid:
        ev = pl.DataFrame(evid).sort("grade", "t_loss", descending=[False, True])
        ev.write_parquet(args.out / "ic_evidence.parquet")
        with pl.Config(tbl_rows=80, tbl_cols=7, tbl_width_chars=200, fmt_str_lengths=40, float_precision=4):
            print("\n== evidence per assigned column: training-CDF gap, IC lost without it (t); strong = gap <= 0.02 and t >= 2")
            print(ev)
    if args.scale_report:  # per assigned column: PM's training-CDF points vs our quantiles at the same p, and the affine map between
        print("\n== scale: feature | formula | gap | PM t = a x ours + b | gap after | PM t @p ~ .1/.5/.9 | ours @ same p")
        for feat, expr in assign.items():
            if feat not in col or feat not in cdf or not re.search(args.scale_report, feat):
                continue
            v = X[:, col[feat]]
            v = v[np.isfinite(v)]
            if v.size < 100:
                continue
            tp, pp = FM.interior(cdf[feat])
            qo = np.quantile(v, pp)
            a_, b_ = np.polyfit(qo, tp, 1) if np.ptp(qo) > 0 else (np.nan, np.nan)
            g0, g1 = FM.cdf_gap(v, cdf[feat]), FM.cdf_gap(a_ * v + b_, cdf[feat]) if np.isfinite(a_) else np.nan
            ix = [int(np.argmin(np.abs(pp - p))) for p in (0.1, 0.5, 0.9)]
            print(f"  {feat} | {expr} | {g0:.3f} | a {a_:.4g} b {b_:.4g} | {g1:.3f} | "
                  + "/".join(f"{tp[i]:.4g}@{pp[i]:.2f}" for i in ix) + " | " + "/".join(f"{qo[i]:.4g}" for i in ix))
    if log:
        pl.DataFrame(log).write_parquet(args.out / "ic_log.parquet")
    with pl.Config(tbl_rows=100, tbl_cols=6, tbl_width_chars=230, fmt_str_lengths=90, float_precision=4):
        if log:
            print("\n== scan: per feature, the expression kept and the three with the largest paired gain (IC, t of the gain over marks)")
            print(pl.DataFrame(log))
    print(f"\nIC: start {base[0]:.4f} -> end {final[0]:.4f} (t {final[1]:.2f}, {final[2]} marks); "
          f"{len(assign)} of {len(names)} columns assigned -> {args.out / 'ic_assign.json'}")
    print(f"time ablation {time.perf_counter() - t_st:.0f}s")
    if args.implied and flat and args.target:
        t_st = time.perf_counter()
        implied(args, X, col, blocks, brows, ret, grp, sid_a, blk, mk, full_pred, fp, assign, ev if evid else None, pool, pool_col)
        print(f"time implied {time.perf_counter() - t_st:.0f}s")
    print(f"time total {time.perf_counter() - t0:.0f}s")


def implied(args, X, col, blocks, brows, ret, grp, sid_a, blk, mk, full_pred, fp, assign, ev, pool, pool_col) -> None:
    """Invert PM's model one open feature at a time: the value of column j that brings our prediction closest to PM's.

    With every other column at our values, the model output is piecewise constant in x_j, the pieces cut at the split
    thresholds the trees use on j. Per sampled row, each piece's representative value is tried (only the trees that split
    on j re-run) and the piece whose prediction is nearest PM's (after a linear map of PM's prediction onto our model's
    raw scale) gives the implied interval; rows where the best piece is not clearly better than the second, or where x_j
    does not move the prediction, carry no information. From the informative rows: where the value lives (stock-day mean
    / mark mean shares of variance -> a daily or a market-wide quantity), its range, and the candidates closest to it.
    Written to <out>/implied.parquet (feature, row, lo, hi, mid)."""
    strong = set(ev.filter(pl.col("grade") == "strong")["feature"].to_list()) if ev is not None else set()
    feats = [f for f in fp.sort("gain_share", descending=True)["feature"].to_list() if f in col and f not in strong][: args.implied]
    if args.implied_feats:
        feats = [f for f in args.implied_feats.split(",") if f in col]
    tm = Counter()  # seconds per stage
    t1 = time.perf_counter()
    rng = np.random.default_rng(1)
    ok_rows = np.flatnonzero(np.isfinite(ret) & np.isfinite(full_pred))
    rows = np.sort(rng.choice(ok_rows, min(args.implied_rows, ok_rows.size), replace=False)).astype(np.int64)
    a_, b_ = np.polyfit(ret[rows], full_pred[rows], 1)  # PM's prediction on our model's raw scale
    target = a_ * ret[rows] + b_
    names = pool.names()
    cand = {}
    for k in names:  # plain candidates (no cross-sectional transform) on the sampled rows, for the matching below
        if k.startswith(XS) or k.startswith(DM):
            continue
        x = pool_col(k)
        if x is not None:
            cand[k] = x[rows]
    n_all = len(cand)
    seen: dict[bytes, str] = {}
    for k in sorted(cand, key=lambda s: (len(s), s)):  # identical values (|x| of a positive x, ...): keep the simplest name
        seen.setdefault(hashlib.blake2b(cand[k].tobytes(), digest_size=16).digest(), k)
    cand = {k: cand[k] for k in seen.values()}
    group: dict[bytes, list[str]] = {}  # same ranks up to sign (x, -x, x * 1e4): one candidate for every rank statistic
    for k in sorted(cand, key=lambda s: (len(s), s)):
        x = cand[k]
        g = np.isfinite(x)
        r = pl.Series(x[g]).rank().to_numpy()
        if r.size and r[0] > (r.size + 1) / 2:
            r = r.size + 1 - r
        group.setdefault(hashlib.blake2b(g.tobytes() + r.tobytes(), digest_size=16).digest(), []).append(k)
    reps_c = [v[0] for v in group.values()]
    members = {v[0]: v for v in group.values()}
    tm["dedup"] += time.perf_counter() - t1
    print(f"\n== implied values ({rows.size:,} rows, PM pred mapped as {a_:.3f} x + {b_:.2e}; {n_all} plain candidates, {len(cand)} distinct, {len(reps_c)} distinct by rank)")
    t1 = time.perf_counter()
    # per-mark ranks of ours and PM's; "misses" = PM's rank beyond what ours explains, "extra" = ours beyond PM's
    rk = pl.DataFrame({"g": grp[rows], "p": full_pred[rows], "t": ret[rows]}).with_columns(
        ((pl.col(c).rank() - 0.5) / pl.len() - 0.5).over("g").alias(c) for c in ("p", "t"))
    pr, tr = rk["p"].to_numpy(), rk["t"].to_numpy()
    res_miss = tr - np.polyfit(pr, tr, 1)[0] * pr
    res_extra = pr - np.polyfit(tr, pr, 1)[0] * tr
    rc, rx = [], []
    for k in reps_c:
        x = cand[k]
        g = np.isfinite(x)
        if g.sum() > 1000:
            xr = pl.Series(x[g]).rank().to_numpy()
            c, c2 = np.corrcoef(xr, res_miss[g])[0, 1], np.corrcoef(xr, res_extra[g])[0, 1]
            if np.isfinite(c):
                rc.append((abs(c), c, k))
            if np.isfinite(c2):
                rx.append((abs(c2), c2, k))
    rc.sort(reverse=True)
    rx.sort(reverse=True)
    print("  residual (PM beyond ours, per-mark ranks) closest: " + ", ".join(f"{k} {c:+.2f}" for _, c, k in rc[:6]))
    print("  residual (ours beyond PM, per-mark ranks) closest: " + ", ".join(f"{k} {c:+.2f}" for _, c, k in rx[:6]))
    dummy = {k for k, x in cand.items() if np.unique(x[np.isfinite(x)]).size < args.min_distinct}  # dummies cannot carry a level match
    top_g, per_fam = [], Counter()
    for _, _, r in rc:  # the top residual groups, at most two per family (rpe900 / rpe1200 are one family): more kinds of quantity
        fam = re.sub(r"\d+", "#", parse(r)[4])
        if per_fam[fam] < 2:
            per_fam[fam] += 1
            top_g.append(r)
        if len(top_g) >= args.carriers:
            break
    carriers = [k for r in top_g for k in members[r]]  # every scale of the top groups: the trees cut on levels
    tm["residual"] += time.perf_counter() - t1
    g_rows = grp[rows]

    def xs_ic(p: np.ndarray) -> float:  # mean per-mark rank IC with PM, the agreement score on these rows
        d = pl.DataFrame({"g": g_rows, "p": p, "t": target}).with_columns(pl.col("p").rank().over("g"), pl.col("t").rank().over("g"))
        return float(d.group_by("g").agg(pl.corr("p", "t")).filter(pl.col("p").is_finite())["p"].mean())

    def pj(j: int) -> np.ndarray:  # the trees that split on column j, at the current X, on the sampled rows
        o = np.zeros(rows.size)
        for b, (_, ms, _, _) in enumerate(blocks):
            sel = np.flatnonzero(blk[rows] == b)
            if sel.size:
                o[sel] = np.mean([m.predict(X, rows[sel], m.trees_of.get(j, np.zeros(0, np.int64))) for m in ms], axis=0)
        return o

    ic0 = xs_ic(full_pred[rows])
    print(f"  agreement on these rows {ic0:.3f}; carrier probe: the top {args.carriers} residual groups ({len(carriers)} candidates) "
          f"tried in each open column, against constants (below / inside / above its split points, missing)")
    best_all = []
    out = []
    for f in feats:
        j = col[f]
        thr = np.unique(np.concatenate([m.thr[(m.feat == j) & ((m.dt & 1) == 0)] for b in range(len(blocks)) for m in blocks[b][1]]))
        if thr.size < 2:
            continue
        reps = np.concatenate([[thr[0] - max(abs(thr[0]), 1.0) * 1e-3], (thr[:-1] + thr[1:]) / 2, [thr[-1] + max(abs(thr[-1]), 1.0) * 1e-3]])
        keep = X[:, j].copy()
        base_rows = full_pred[rows].copy()
        t1 = time.perf_counter()
        old = pj(j)
        tried, const = [], []
        for nm_, v in (("lo", reps[0]), ("mid", reps[reps.size // 2]), ("hi", reps[-1]), ("nan", np.nan)):  # what any constant gives
            X[rows, j] = v
            const.append((xs_ic(base_rows - old + pj(j)) - ic0, nm_))
        for k in carriers:  # put the residual's candidate in column j: does agreement with PM go up, beyond the best constant?
            X[rows, j] = cand[k]
            tried.append((xs_ic(base_rows - old + pj(j)) - ic0, k))
        X[:, j] = keep
        tried.sort(reverse=True)
        const.sort(reverse=True)
        cb = const[0][0]
        best_all += [(d - cb, f, k, d) for d, k in tried[:3]]
        tm["carrier"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        err = np.empty((reps.size, rows.size))
        resp = np.empty(reps.size)  # the model's mean response to x_j (shape: monotone, U, flat)
        for k, v in enumerate(reps):
            X[:, j] = v
            new = pj(j)
            err[k] = np.abs(base_rows - old + new - target)
            resp[k] = np.mean(new - old)
        X[:, j] = keep
        tm["reps"] += time.perf_counter() - t1
        e1 = err.min(axis=0)
        spread = err.max(axis=0) - e1
        near = err <= e1 + 0.1 * spread  # pieces within 10% of the error range of the best: the implied stretch of x_j
        idx = np.arange(reps.size)[:, None]
        kmin = np.where(near, idx, reps.size).min(axis=0)
        kmax = np.where(near, idx, -1).max(axis=0)
        info = (spread > 1e-9) & (e1 < 0.2 * spread) & (kmax - kmin < 0.3 * reps.size)  # x_j matters, it fits, the stretch is narrow
        lo = np.where(kmin > 0, thr[np.clip(kmin - 1, 0, thr.size - 1)], -np.inf)
        hi = np.where(kmax < thr.size, thr[np.clip(kmax, 0, thr.size - 1)], np.inf)
        best = (kmin + kmax) // 2
        mid = reps[best]
        out.append(pl.DataFrame({"feature": f, "row": rows, "lo": lo, "hi": hi, "mid": mid, "info": info}))
        m = info
        if m.sum() < 200:
            print(f"  {f}: only {int(m.sum())} informative rows")
            print(f"  {f} carrier: const {const[0][1]} {cb:+.4f}; " + ", ".join(f"{k} {d:+.4f} ({d - cb:+.4f})" for d, k in tried[:3]))
            continue
        t1 = time.perf_counter()
        d = pl.DataFrame({"y": mid[m], "sd": blk[rows][m] * 10_000_000 + sid_a[rows][m], "mk": grp[rows][m]})
        tot = d["y"].var()
        sd_share = d.group_by("sd").agg(pl.col("y").mean().alias("g"), pl.len()).filter(pl.col("len") > 1)
        r_sd = 1 - d.join(sd_share, on="sd").select((pl.col("y") - pl.col("g")).var()).item() / tot if tot and tot > 0 else np.nan
        mk_mean = d.group_by("mk").agg(pl.col("y").mean().alias("g"))
        r_mk = 1 - d.join(mk_mean, on="mk").select((pl.col("y") - pl.col("g")).var()).item() / tot if tot and tot > 0 else np.nan
        y = mid[m]
        sc = []
        for k in reps_c:
            xv = cand[k][m]
            g = np.isfinite(xv)
            if g.sum() < 200:
                continue
            c = np.corrcoef(pl.Series(xv[g]).rank().to_numpy(), pl.Series(y[g]).rank().to_numpy())[0, 1]
            if np.isfinite(c):
                sc.append((abs(c), c, k))
        sc.sort(reverse=True)
        tm["closest"] += time.perf_counter() - t1
        t1 = time.perf_counter()
        mb = m & np.isfinite(lo) & np.isfinite(hi)  # bounded intervals only: an open-ended edge piece says little about the level
        lv = []  # level match: how often the candidate's own value falls in the implied interval, over chance (shuffled within its rows)
        for k, x in cand.items():
            if k in dummy:
                continue
            xv = x[mb]
            g = np.isfinite(xv)
            if g.sum() < 200:
                continue
            xg, lg, hg = xv[g], lo[mb][g], hi[mb][g]
            hit = np.mean((xg > lg) & (xg <= hg))
            xs = rng.permutation(xg)
            lv.append((hit - np.mean((xs > lg) & (xs <= hg)), hit, k))
        lv.sort(reverse=True)
        tm["level"] += time.perf_counter() - t1
        cur = assign.get(f, "-")
        cv = keep[rows][m]
        own = (np.corrcoef(pl.Series(cv[np.isfinite(cv)]).rank().to_numpy(), pl.Series(y[np.isfinite(cv)]).rank().to_numpy())[0, 1]
               if np.isfinite(cv).sum() > 200 else np.nan)
        q = np.linspace(0, resp.size - 1, 5).astype(int)
        mono = np.corrcoef(np.arange(resp.size), resp)[0, 1]
        shape = "up" if mono > 0.7 else "down" if mono < -0.7 else ("U" if resp[resp.size // 2] < min(resp[0], resp[-1]) else "hump" if resp[resp.size // 2] > max(resp[0], resp[-1]) else "mixed")
        inv = {v: k for k, v in col.items()}
        pairs: Counter = Counter()
        for b in range(len(blocks)):
            for mm in blocks[b][1]:
                at_j = np.flatnonzero(mm.feat == j)
                for kk in at_j:
                    t_ = np.searchsorted(mm.node_off, kk, side="right") - 1
                    for ch in (mm.left[kk], mm.right[kk]):
                        if ch >= 0:
                            pairs[inv.get(int(mm.feat[mm.node_off[t_] + ch]), "?")] += 1
        print(f"  {f} response {shape} ({mono:+.2f}; " + "/".join(f"{resp[i]:+.1e}" for i in q) + "); split together with "
              + ", ".join(f"{k} {n}" for k, n in pairs.most_common(4) if k != f) + f"; implied vs current formula {own:+.2f}")
        print(f"  {f} (now {cur}): informative {m.mean():.2f}; implied p10/p50/p90 {np.percentile(y, 10):.4g}/{np.median(y):.4g}/{np.percentile(y, 90):.4g}; "
              f"variance explained by stock-day {r_sd:.2f}, by mark {r_mk:.2f}; closest: "
              + ", ".join(f"{k} {c:+.2f}" for _, c, k in sc[:4]))
        print(f"  {f} level match ({mb.mean():.2f} bounded; in interval / lift over chance): " + ", ".join(f"{k} {h:.2f}/{d:+.2f}" for d, h, k in lv[:4]))
        print(f"  {f} carrier: const {const[0][1]} {cb:+.4f}; " + ", ".join(f"{k} {d:+.4f} ({d - cb:+.4f})" for d, k in tried[:3]))
    best_all.sort(reverse=True)
    print("  carrier best overall (gain over the column's best constant; raw gain), column <- candidate: "
          + ", ".join(f"{f}<-{k} {dc:+.4f} ({d:+.4f})" for dc, f, k, d in best_all[:10]))
    if args.carrier_assigned:  # the assigned columns too (strong ones included): a formula the residual's quantity beats is suspect
        t1 = time.perf_counter()
        base_rows = full_pred[rows].copy()
        beat, worse = [], []
        for f in [f for f in assign if f in col and f not in feats]:
            j = col[f]
            keep = X[:, j].copy()
            old = pj(j)
            res = []
            thr = np.unique(np.concatenate([m.thr[(m.feat == j) & ((m.dt & 1) == 0)] for b in range(len(blocks)) for m in blocks[b][1]]))
            cs_ = [("const", np.nanmedian(keep[rows]) if np.isfinite(keep[rows]).any() else 0.0), ("missing", np.nan)]
            if thr.size:  # below / above every split point: a candidate whose values all sit on one side is only a constant
                cs_ += [("const_lo", thr[0] - max(abs(thr[0]), 1.0)), ("const_hi", thr[-1] + max(abs(thr[-1]), 1.0))]
            for nm_, v in cs_:
                X[rows, j] = v
                res.append((xs_ic(base_rows - old + pj(j)) - ic0, nm_))
            cbest = max(d for d, k in res)
            for k in carriers:
                X[rows, j] = cand[k]
                res.append((xs_ic(base_rows - old + pj(j)) - ic0, k))
            X[:, j] = keep
            res.sort(reverse=True)
            beat.append((res[0][0], f, res[0][1], assign[f], res[0][0] - cbest if res[0][1] not in dict(cs_) else 0.0))
            worse.append((-next(d for d, k in res if k == "missing"), f, assign[f]))
        beat.sort(reverse=True)
        worse.sort()
        tm["carrier_assigned"] += time.perf_counter() - t1
        print(f"  assigned columns ({len(beat)}), the best replacement (residual candidate, constants, missing) over the current formula "
              f"[a candidate's margin over the best constant]: "
              + ", ".join(f"{f}<-{k} {d:+.4f} [{dc:+.4f}] (now {e})" for d, f, k, e, dc in beat[:10]))
        print("  assigned columns worth least (agreement lost when missing): " + ", ".join(f"{f} {d:+.4f} ({e})" for d, f, e in worse[:8]))
    for k, v in tm.items():
        print(f"time implied_{k} {v:.0f}s")
    if out:
        pl.concat(out).write_parquet(args.out / "implied.parquet")


if __name__ == "__main__":
    main()
