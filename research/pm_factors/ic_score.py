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


def stem_frames(frames: list[pl.DataFrame], stem: str) -> list[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Per mark: (sids, candidate expressions of the stem) -- feature_match.candidates, keeping the sid order."""
    out = []
    for fr in frames:
        a = FM.atoms(fr, stem)
        if a.is_empty():
            out.append((np.array([], int), {}))
            continue
        sids = np.array(sorted(a["sid"].unique().to_list()))
        out.append((sids, FM.candidates(a)))
    return out


def rank_ic(pred: np.ndarray, ret: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    """Mean cross-sectional Spearman IC over groups (marks), its t-stat and the number of groups."""
    ics = []
    for g in np.unique(groups):
        m = (groups == g) & np.isfinite(pred) & np.isfinite(ret)
        if m.sum() < 20 or np.std(pred[m]) == 0:
            continue
        r1 = np.argsort(np.argsort(pred[m])).astype(float)
        r2 = np.argsort(np.argsort(ret[m])).astype(float)
        ics.append(np.corrcoef(r1, r2)[0, 1])
    if not ics:
        return np.nan, np.nan, 0
    a = np.array(ics)
    return float(a.mean()), float(a.mean() / (a.std(ddof=1) / np.sqrt(a.size))) if a.size > 1 else np.nan, int(a.size)


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
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--max-features", type=int, default=30, help="scan this many features, by model gain")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage6"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    lines = [ln.strip() for ln in args.feature_csv.read_text(errors="replace").splitlines() if ln.strip()]
    names = [ln.split(",")[0].strip() for ln in lines]
    if names and names[0].lower() == "name":
        names = names[1:]
    col = {n: i for i, n in enumerate(names)}
    fp = pl.read_parquet(args.fingerprint).with_columns(pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"))
    con = duckdb.connect(str(B.CATALOG), read_only=True)
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
        rows, rets = [], {}
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            built = FI.rebuild(gf, gs, MARKS)
            for k, v in built.items():
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": k, "pm": v}))
            if gf is not None and gf.height > 2:
                a = E.arrays(gf, "fast")
                mid = a["mid"]
                ok = np.isfinite(mid) & (mid > 0)
                m0 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS)
                m1 = FI.at_marks(a["t_local"][ok], mid[ok], MARKS + HORIZON)
                with np.errstate(divide="ignore", invalid="ignore"):
                    rets[sid] = np.log(m1 / m0)
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": "px", "pm": FI.at_marks(a["t_local"], a["price"], MARKS)}))
        long = pl.concat(rows).filter(pl.col("pm").is_finite()).with_columns(pl.col("sid").cast(pl.Int64))
        frames = [long.filter(pl.col("mark") == m).drop("mark") for m in MARKS]
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
    cache: dict[str, list] = {}  # stem -> per block, per mark: (sids, candidates)

    def column(feature: str, expr: str) -> np.ndarray:
        """One model column filled with expression ``expr`` of the feature's stem."""
        stem = feature.split(".")[0]
        if stem not in cache:
            cache[stem] = [stem_frames(fr, stem) for _, _, fr, _ in blocks]
        v = np.full(sid_a.size, np.nan)
        for b, per_mark in enumerate(cache[stem]):
            for m, (sids, cand) in enumerate(per_mark):
                if expr in cand:
                    for s, x in zip(sids, cand[expr], strict=True):
                        i = pos.get((b, m, int(s)))
                        if i is not None:
                            v[i] = x
        return v

    def score(Xm: np.ndarray) -> tuple[float, float, int]:
        pred = np.empty(sid_a.size)
        for b, (_, models, _, _) in enumerate(blocks):
            m = blk == b
            pred[m] = predict(models, Xm[m])
        return rank_ic(pred, ret, grp)

    assign = dict(KNOWN)
    if args.assign and args.assign.exists():
        assign = json.loads(args.assign.read_text())
    for feat, expr in assign.items():
        if feat in col:
            X[:, col[feat]] = column(feat, expr)
    base = score(X)
    print(f"\nall-missing except the assignment {assign}: IC {base[0]:.4f} (t {base[1]:.2f}, {base[2]} marks) ({time.perf_counter() - t0:.0f}s)")
    log = []
    feats = [f for f in fp.sort("gain_share", descending=True)["feature"].to_list() if f in col][: args.max_features]
    for ps in range(args.passes):
        for feat in feats:
            stem = feat.split(".")[0]
            if stem not in cache:
                cache[stem] = [stem_frames(fr, stem) for _, _, fr, _ in blocks]
            pooled: dict[str, list[np.ndarray]] = {}
            for per_mark in cache[stem]:
                for _, cand in per_mark:
                    for k, v in cand.items():
                        pooled.setdefault(k, []).append(v)
            cand_all = FM.pool(pooled)
            if not cand_all:
                continue
            short = [r["candidate"] for r in FM.rank(fp.filter(pl.col("feature") == feat), cand_all, args.shortlist)]
            cur = assign.get(feat)
            j = col[feat]
            keep_col = X[:, j].copy()
            best = (score(X)[0], cur)
            tried = []
            for expr in short:
                X[:, j] = column(feat, expr)
                ic = score(X)[0]
                tried.append((expr, ic))
                if np.isfinite(ic) and ic > best[0] + args.min_gain:
                    best = (ic, expr)
            if best[1] != cur and best[1] is not None:
                assign[feat] = best[1]
                X[:, j] = column(feat, best[1])
            else:
                X[:, j] = keep_col
            tried.sort(key=lambda t: -t[1] if np.isfinite(t[1]) else 1e9)
            log.append({"pass": ps, "feature": feat, "kept": assign.get(feat), "ic_after": best[0],
                        "top": "; ".join(f"{e} {ic:.4f}" for e, ic in tried[:3])})
            print(f"pass {ps} {feat}: kept {assign.get(feat)} IC {best[0]:.4f} ({time.perf_counter() - t0:.0f}s)")
    final = score(X)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ic_assign.json").write_text(json.dumps(assign, indent=1))
    with pl.Config(tbl_rows=100, tbl_cols=6, tbl_width_chars=230, fmt_str_lengths=90, float_precision=4):
        if log:
            print("\n== scan: per feature, the expression kept and the best three tried (IC)")
            print(pl.DataFrame(log))
    print(f"\nIC: start {base[0]:.4f} -> end {final[0]:.4f} (t {final[1]:.2f}, {final[2]} marks); "
          f"{len(assign)} of {len(names)} columns assigned -> {args.out / 'ic_assign.json'}")


if __name__ == "__main__":
    main()
