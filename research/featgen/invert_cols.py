r"""Walk our model input toward PM's: per-row implied values of the open columns, then whether our bases predict them.

PM's model files are in hand, so what still separates our prediction from PM's is the inputs -- mostly the open columns
(no settled formula). Stage 1 (inversion, in-sample): for each open column in turn, every row may take any of the
column's split intervals in PM's trees (a point inside each interval, below the lowest, above the highest, NaN, or its
current value); all candidates are scored in one lgbpred --batch run and each row keeps the one whose score lands
closest to PM's prediction (PM's scale mapped onto ours by a least-squares line per day). A few passes over the columns
give the implied inputs: how close the model can get to PM when only the open columns move (an upper bound, and the
answer to "is anything outside the open columns wrong"). Stage 2 (the honest test): a LightGBM regressor learns each
column's implied value from our bases and the settled columns on one day and fills the column on the other
day; the similarity gain there, against the NaN baseline, says whether the implied values are something our quantities
carry, and the regressor's split gain names which ones. Stage 3: the walk again, each column a regressor of our bases
fitted to the rows' best values on the first day (rows weighted by how much their choice matters), kept when that day
improves, judged on the others. Synthetic check: with one open column the inversion finds it (the regressor's top split
is the true base); with ten open at once the per-row values are not identified (rank corr .06 with the truth) though
held-out similarity still rises -- read stage 1 as an upper bound, the held-out numbers of stages 2-3 as the honest ones.

    python research/featgen/invert_cols.py --days 20261112,20260430 --x-dir "data/reports/stage6_r70_{mmdd}" \
        --base "data/featgen_r67/base_{date}.parquet" --models $DATA/models_gen/B --out data/invert_r96
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from implied_scan import Day  # noqa: E402
from missing_damage import sim  # noqa: E402

BASELINE = HERE.parent / "engine" / "baseline_r41c.txt"


def thresholds(files: list[Path], n_cols: int) -> list[np.ndarray]:
    """Every numeric split threshold of every column over the model files (categorical splits are skipped)."""
    th: list[set[float]] = [set() for _ in range(n_cols)]
    for f in files:
        tree: dict[str, list[str]] = {}
        for ln in f.read_text().splitlines():
            for key in ("split_feature", "threshold", "decision_type"):
                if ln.startswith(key + "="):
                    tree[key] = ln[len(key) + 1:].split()
            if len(tree) == 3:  # bit 0 of decision_type = categorical split
                for j, t, dt in zip(tree["split_feature"], tree["threshold"], tree["decision_type"], strict=True):
                    if not int(dt) & 1:
                        th[int(j)].add(float(t))
                tree = {}
            elif ln.startswith("Tree="):
                tree = {}
    return [np.array(sorted(s)) for s in th]


def points(t: np.ndarray, cap: int) -> np.ndarray:
    """One value inside each interval between consecutive thresholds, one below and one above (at most cap)."""
    if t.size > cap - 1:
        t = t[np.unique(np.linspace(0, t.size - 1, cap - 1).round().astype(int))]
    pad = max(1.0, float(np.abs(t).max()) if t.size else 1.0)
    return np.concatenate([[t[0] - pad] if t.size else [0.0], (t[:-1] + t[1:]) / 2, [t[-1] + pad] if t.size else []])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--target", default="open", help="comma list of model columns, or 'open' = every column without a --formulas line")
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--cap", type=int, default=128, help="most candidate values per column")
    ap.add_argument("--out", type=Path, default=Path("data/invert_cols"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-fit", action="store_true", help="stage 1 only")
    ap.add_argument("--with-settled", action="store_true", help="regressors may also read the settled model columns (they then partly re-learn PM's trees; off = our bases only)")
    a = ap.parse_args()

    names = P.feature_names(a.feature_csv)
    settled = {ln.split()[0] for ln in a.formulas.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    targets = [n for n in names if n not in settled] if a.target == "open" else [t for t in a.target.split(",") if t]
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names) for d in ds}
    files = {d: P.model_files(a.models, d, a.fits.split(","), a.subs.split(",")) for d in ds}
    a.out.mkdir(parents=True, exist_ok=True)
    work = a.out / "work"
    work.mkdir(exist_ok=True)
    exe = P.ensure_binary()

    def score(d: str, M: np.ndarray) -> np.ndarray:
        P.write_x(work / "x.bin", M)
        subprocess.run([str(exe), str(work / "x.bin"), str(work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
        return P.read_scores(work / "p.bin").copy()

    def batch(d: str, M: np.ndarray, col: int, cols: list[np.ndarray]) -> np.ndarray:
        P.write_x(work / "x.bin", M)
        with (work / "c.bin").open("wb") as f:
            f.write(b"CRULC001")
            f.write(np.array([len(cols), M.shape[0]], "<u8").tobytes())
            for v in cols:
                f.write(np.array([col], "<u8").tobytes())
                f.write(np.ascontiguousarray(v, "<f8").tobytes())
        subprocess.run([str(exe), str(work / "x.bin"), str(work / "b.bin"), *map(str, files[d]), "--batch", str(work / "c.bin"), "--threads", str(a.threads)],
                       check=True, stdout=subprocess.DEVNULL)
        raw = (work / "b.bin").read_bytes()
        n, r = (int(x) for x in np.frombuffer(raw, "<u8", 2, 8))
        return np.frombuffer(raw, "<f8", n * r, 24).reshape(n, r)

    th = {d: thresholds(files[d], len(names)) for d in ds}
    X = {d: D.X.copy() for d, D in days.items()}
    base_X = {d: D.X.copy() for d, D in days.items()}  # the open columns as scored today (NaN where no formula)
    s0 = {}
    line = {}
    for d, D in days.items():
        sc = score(d, X[d])
        s0[d] = sim(sc, D.pm, D.marks)
        ok = np.isfinite(sc) & np.isfinite(D.pm)
        b1, b0 = np.polyfit(D.pm[ok], sc[ok], 1)  # PM's prediction on our score's scale
        line[d] = (b0, b1)
    print(f"==== stage 1: {len(targets)} columns, {a.passes} passes; start " + ", ".join(f"{d} {v:.4f} (bases at {days[d].found:.3f} of rows)" for d, v in s0.items()), flush=True)
    cand: dict[tuple[str, str], np.ndarray] = {}
    for p in range(a.passes):
        for t in targets:
            if t not in ix:
                continue
            j = ix[t]
            t0 = time.perf_counter()
            msg = []
            for d, D in days.items():
                pts = np.concatenate([points(th[d][j], a.cap), [np.nan]]) if th[d][j].size else np.array([np.nan])
                cand[(d, t)] = pts
                cols = [np.full(D.X.shape[0], v) for v in pts] + [X[d][:, j].copy()]
                S = batch(d, X[d], j, cols)
                want = line[d][0] + line[d][1] * D.pm
                err = (S - want[None, :]) ** 2
                err[:, ~np.isfinite(want)] = np.inf
                err[-1] -= 1e-15  # keep the current value on a tie
                k = np.argmin(err, axis=0)
                k[~np.isfinite(want)] = len(cols) - 1
                X[d][:, j] = np.where(k == len(cols) - 1, X[d][:, j], pts[np.minimum(k, pts.size - 1)])
                msg.append(f"{d} {sim(S[np.minimum(k, len(cols) - 1), np.arange(S.shape[1])], D.pm, D.marks):.4f}")
            print(f"  pass {p + 1} {t:<10} {len(cand[(ds[0], t)]):>4} values  sim " + "  ".join(msg) + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    s1 = {d: sim(score(d, X[d]), D.pm, D.marks) for d, D in days.items()}
    print("  stage 1 end: " + ", ".join(f"{d} {s0[d]:.4f} -> {s1[d]:.4f}" for d in ds), flush=True)
    for d, D in days.items():
        key = pl.read_parquet(Path(a.x_dir.format(mmdd=d[4:], date=d)) / "x.parquet", columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").sort("sid", "mark")
        pl.DataFrame({"sid": key["sid"].cast(pl.Int64), "mark": key["mark"], **{t: X[d][:, ix[t]] for t in targets if t in ix}}).write_parquet(a.out / f"implied_{d}.parquet")
    if a.no_fit or len(ds) < 2:
        return

    import lightgbm as lgb

    print("==== stage 2: regress each column's implied value on our bases (+ settled columns with --with-settled) of one day, fill it on the other", flush=True)
    bnames = sorted(set.intersection(*[set(D.bases) for D in days.values()]))
    snames = sorted(settled & set(ix)) if a.with_settled else []
    feat_names = [f"gen.{b}" for b in bnames] + snames
    F = {d: np.column_stack([days[d].bases[b] for b in bnames] + [base_X[d][:, ix[s]] for s in snames]) for d in ds}
    rows = []
    for t in targets:
        if t not in ix:
            continue
        j = ix[t]
        out = {}
        for tr, te in ((ds[0], ds[1]), (ds[1], ds[0])):
            y = X[tr][:, j]
            ok = np.isfinite(y)
            if ok.sum() < 1000 or np.unique(y[ok]).size < 2:
                continue
            bst = lgb.train({"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "verbose": -1,
                             "num_threads": a.threads}, lgb.Dataset(F[tr][ok], y[ok]), num_boost_round=200)
            M = base_X[te].copy()
            M[:, j] = bst.predict(F[te])
            g = sim(score(te, M), days[te].pm, days[te].marks) - s0[te]
            imp = bst.feature_importance("gain")
            top = ", ".join(f"{feat_names[i]} {imp[i] / imp.sum():.2f}" for i in np.argsort(-imp)[:4])
            out[te] = (g, top)
        if len(out) == 2:
            rows.append((min(g for g, _ in out.values()), t, out))
    rows.sort(key=lambda r: -r[0])
    for mn, t, out in rows:
        print(f"  {t:<10} min {mn:+.4f}  " + "  ".join(f"{d} {g:+.4f} [{top}]" for d, (g, top) in out.items()), flush=True)
    print(f"  columns helping on both days held out: {sum(r[0] > 0 for r in rows)} of {len(rows)}")

    # stage 3: the same walk with every column a function of our quantities, learnt on the first day, judged on the others
    tr, tes = ds[0], ds[1:]
    print(f"==== stage 3: each column = regressor(our bases), fitted on {tr} to the rows' best values, kept if {tr} "
          f"improves; held out {', '.join(tes)}", flush=True)
    C = {d: base_X[d].copy() for d in ds}
    cur = {d: s0[d] for d in ds}
    for p in range(a.passes):
        for t in targets:
            if t not in ix or not th[tr][ix[t]].size:
                continue
            j = ix[t]
            D = days[tr]
            pts = np.concatenate([points(th[tr][j], a.cap), [np.nan]])
            S = batch(tr, C[tr], j, [np.full(D.X.shape[0], v) for v in pts])
            want = line[tr][0] + line[tr][1] * D.pm
            err = (S - want[None, :]) ** 2
            err[:, ~np.isfinite(want)] = np.inf
            y = pts[np.argmin(err, axis=0)]
            ok = np.isfinite(y) & np.isfinite(want)
            if ok.sum() < 1000:
                continue
            fe = err[:-1]  # the finite candidates
            w = fe.mean(axis=0) - fe.min(axis=0)  # how much this row's choice matters: rows where every value scores alike carry no information
            ok &= np.isfinite(w) & (w > 0)
            bst = lgb.train({"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "verbose": -1,
                             "num_threads": a.threads}, lgb.Dataset(F[tr][ok], y[ok], weight=w[ok] / w[ok].mean()), num_boost_round=200)
            new = {}
            for d in ds:
                M = C[d].copy()
                M[:, j] = bst.predict(F[d])
                new[d] = (M, sim(score(d, M), days[d].pm, days[d].marks))
            keep = new[tr][1] > cur[tr]
            if keep:
                for d in ds:
                    C[d], cur[d] = new[d]
            imp = bst.feature_importance("gain")
            top = ", ".join(f"{feat_names[i]} {imp[i] / imp.sum():.2f}" for i in np.argsort(-imp)[:3])
            print(f"  pass {p + 1} {t:<10} {'kept' if keep else 'drop'}  " + "  ".join(f"{d} {new[d][1]:.4f}" for d in ds) + f"  [{top}]", flush=True)
    print("  stage 3 end: " + ", ".join(f"{d} {s0[d]:.4f} -> {cur[d]:.4f}{' (fit day)' if d == tr else ' (held out)'}" for d in ds), flush=True)
    for f in ("x.bin", "p.bin", "c.bin", "b.bin"):
        (work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
