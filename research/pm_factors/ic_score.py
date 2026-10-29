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

The assignment found is written to ``ic_assign.json`` and can be fed back with ``--assign``.

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

MARKS = np.arange(300.0, 13801.0, 300.0)  # 09:35 ... 14:50: the last mark still has 5 minutes of continuous session ahead
HORIZON = 300.0
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
        import lightgbm as lgb  # optional: much faster than the tree walker
        return [lgb.Booster(model_file=str(f)) for f in files]
    except ImportError:
        return [TreeModel(f) for f in files]


def predict(models: list, X: np.ndarray) -> np.ndarray:
    """Mean prediction of the models."""
    return np.mean([m.predict(X) for m in models], axis=0)


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
            out["cum_imb"] = upto(T, "nv") / upto(T, "v")
            px = FI.at_marks(t, b["price"], T)
            out["lp_vwap"] = np.log(px / (upto(T, "a") / upto(T, "v")))
            hi, lo = np.fmax.accumulate(np.where(cont, b["price"], np.nan)), np.fmin.accumulate(np.where(cont, b["price"], np.nan))
            out["stoch"] = (px - FI.at_marks(t, lo, T)) / (FI.at_marks(t, hi, T) - FI.at_marks(t, lo, T))
    return out


def family(expr: str) -> str:
    """Drop an expression's horizon / window digits and price flavour: -gen.ret30 and -gen.mret45 share one family."""
    return re.sub(r"\d+", "#", re.sub(r"gen\.[mp]?ret", "gen.ret", expr))


def stem_frames(frames: list[pl.DataFrame], stem: str) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Per mark: (sids, candidates) -- the stem's state expressions."""
    out = []
    for fr in frames:
        a = FM.atoms(fr, stem)
        sids = np.array(sorted(a["sid"].unique().to_list()))
        out.append((sids, FM.candidates(a)))
    return out


XS = ("z:", "rank:", "crank:")  # cross-sectional transforms within a mark: each has the same CDF whatever it transforms


def parse(expr: str) -> tuple[str, bool, bool, float, str]:
    """``crank:-gen.ret30`` / ``|gen.timb|*1e4`` -> (transform, negate, abs, scale, base): the names ``Pool`` makes."""
    pre = next((p for p in XS if expr.startswith(p)), "")
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
        return xs(pre, (-v if neg else v) * scale)

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
    out = []
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(pred) & np.isfinite(ret)
        if m.sum() < 20 or np.std(pred[m]) == 0:
            out.append(np.nan)
            continue
        r1 = np.argsort(np.argsort(pred[m])).astype(float)
        r2 = np.argsort(np.argsort(ret[m])).astype(float)
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
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage6"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    lines = [ln.strip() for ln in args.feature_csv.read_text(errors="replace").splitlines() if ln.strip()]
    names = [ln.split(",")[0].strip() for ln in lines]
    if names and names[0].lower() == "name":
        names = names[1:]
    col = {n: i for i, n in enumerate(names)}
    fp = pl.read_parquet(args.fingerprint).with_columns(pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"))
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
        pm = B.load_state(f)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(pm["sid"].unique().to_list())
        up = E.load(con, int(d), sids, stop)
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        dc = CM.daily_candidates(d, sids) if CM is not None else None
        dbase = {r["sid"]: r for r in dc.select("sid", "adv20", "avgvol20", "freeshare").fill_null(np.nan).iter_rows(named=True)} if dc is not None else {}
        rows, rets = [], {}
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            built = FI.rebuild(gf, gs, MARKS)
            for k, v in built.items():
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": k, "pm": v}))
            if sid in dbase and "zzqj.v" in built:  # intraday over daily: amount / volume since 09:30 over 20-day averages, volume over float
                bs = dbase[sid]
                with np.errstate(divide="ignore", invalid="ignore"):
                    for k, v in (("amt_adv20", built["zzqj.tpv"] / bs["adv20"]), ("vol_avg20", built["zzqj.v"] / bs["avgvol20"]),
                                 ("vol_float", built["zzqj.v"] / bs["freeshare"])):
                        rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": f"ced.{k}", "pm": np.asarray(v, float)}))
            if gf is not None and gf.height > 2:
                a = E.arrays(gf, "fast")
                mid = a["mid"]
                ok = np.isfinite(mid) & (mid > 0)
                m0 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS)
                m1 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS + HORIZON)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets[sid] = np.log(m1 / m0)
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": "px", "pm": FI.at_marks(a["t_local"], a["price"], MARKS)}))
            for k, v in generic(gf, gs).items():
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": f"gen.{k}", "pm": v.astype(float)}))
        long = pl.concat(rows).filter(pl.col("pm").is_finite()).with_columns(pl.col("sid").cast(pl.Int64))
        frames = [long.filter(pl.col("mark") == m).drop("mark") for m in MARKS]
        pool.add_block(frames, dc)
        state_stems |= set(long["key"].str.extract(r"^([^.]+)\.", 1).drop_nulls().unique().to_list()) - {"gen", "ced"}
        models = load_models(args.models, d, args.fits.split(","), args.subs.split(","))
        blocks.append((d, models, frames, rets))
        print(f"{d}: {len(sids)} stocks x {MARKS.size} marks, {len(models)} models ({time.perf_counter() - t0:.0f}s)")
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

    def stem_pool(stem: str) -> dict[str, np.ndarray]:
        """Pool a stem's candidates over days and marks (the stateless pool: subsampled)."""
        if stem not in state_stems:
            return pool.samples()
        pooled: dict[str, list[np.ndarray]] = {}
        for per_mark in stem_cache(stem):
            for _, cand, _ in per_mark:
                for k, v in cand.items():
                    pooled.setdefault(k, []).append(v)
        return FM.pool(pooled)

    def column(feature: str, expr: str) -> np.ndarray:
        """One model column filled with expression ``expr`` of the feature's stem."""
        stem = feature.split(".")[0]
        v = np.full(sid_a.size, np.nan)
        for b in range(len(blocks)):
            for m in range(MARKS.size):
                if stem in state_stems:
                    _, cand, ix = stem_cache(stem)[b][m]
                    x = cand.get(expr)
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

    def score(Xm: np.ndarray) -> np.ndarray:
        pred = np.empty(sid_a.size)
        for b, (_, models, _, _) in enumerate(blocks):
            m = blk == b
            pred[m] = predict(models, Xm[m])
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
            in_scale = lambda r: not np.isfinite(r["in_range"]) or r["in_range"] >= args.min_in_range  # noqa: E731
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
            cur_v = np.nan_to_num(score(X), nan=0.0)  # a constant prediction (everything missing) ranks nothing: IC 0
            best = (summary(cur_v)[0], cur, 0.0)
            tried = []
            for expr in short:
                X[:, j] = column(feat, expr)
                v = np.nan_to_num(score(X), nan=0.0)
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
                        X[:, j] = column(feat, e)
                        v = np.nan_to_num(score(X), nan=0.0)
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
    final = summary(score(X))
    if cdf:  # what the model's own training distribution says about each assigned column
        rep = []
        for feat, expr in assign.items():
            if feat not in cdf:
                continue
            allc = stem_pool(feat.split(".")[0])
            gaps = sorted(((FM.cdf_gap(v, cdf[feat]), k) for k, v in allc.items()), key=lambda t: t[0])
            cur_gap = FM.cdf_gap(allc[expr], cdf[feat]) if expr in allc else np.nan
            rep.append({"feature": feat, "kept": expr, "kept_cdf_gap": cur_gap, "cdf_points": cdf[feat][0].size,
                        "closest_by_cdf": "; ".join(f"{k} {g:.3f}" for g, k in gaps[:5])})
        with pl.Config(tbl_rows=60, tbl_cols=5, tbl_width_chars=230, fmt_str_lengths=120, float_precision=4):
            print("\n== assigned columns against the model's training CDF (mean |F(t) - p| over root splits; 0 = same distribution)")
            print(pl.DataFrame(rep))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ic_assign.json").write_text(json.dumps(assign, indent=1))
    with pl.Config(tbl_rows=100, tbl_cols=6, tbl_width_chars=230, fmt_str_lengths=90, float_precision=4):
        if log:
            print("\n== scan: per feature, the expression kept and the three with the largest paired gain (IC, t of the gain over marks)")
            print(pl.DataFrame(log))
    print(f"\nIC: start {base[0]:.4f} -> end {final[0]:.4f} (t {final[1]:.2f}, {final[2]} marks); "
          f"{len(assign)} of {len(names)} columns assigned -> {args.out / 'ic_assign.json'}")


if __name__ == "__main__":
    main()
