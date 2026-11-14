r"""Which missing model columns move the prediction most: each open column, filled with draws from PM's own training
distribution instead of NaN, through the C++ predictor.

Gain share (gain_share.py) says how much the model used a column in training; this says how much our NaN in it changes
today's scores. For each open column with training-CDF points (cal json): draw values by inverse-CDF interpolation of
the points, independently per row (so it carries no information, only the column's scale), score, and compare with the
all-NaN baseline -- the pooled correlation, the mean |change| over the scores' spread, and the similarity to PM's
prediction (per-mark rank IC, as everywhere) with the column filled. A column whose random fill moves the scores a lot is
one the model leans on and where our NaN takes the tree down its default branch every time.

    python research/featgen/missing_damage.py --date 20261112 --x $R/stage6_r70_1112/x.parquet --pred $R/stage6_r70_1112/pred.parquet \
        --cal $R/stage6_r70_1112/cal_20261112.json --models $DATA/models_gen/B --top 25
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402
from gain_share import gains  # noqa: E402

BASELINE = HERE.parent / "engine" / "baseline_r41c.txt"


def sim(pred: np.ndarray, pm: np.ndarray, marks: np.ndarray) -> float:
    ics = []
    for mk in np.unique(marks):
        m = (marks == mk) & np.isfinite(pred) & np.isfinite(pm)
        if m.sum() >= 30:
            a, b = pl.Series(pred[m]).rank().to_numpy(), pl.Series(pm[m]).rank().to_numpy()
            if a.std() > 0 and b.std() > 0:
                ics.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(ics)) if ics else float("nan")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True, help="ic_score --dump-x's x.parquet (date, sid, mark, columns)")
    ap.add_argument("--pred", type=Path, required=True, help="ic_score --dump-pred's pred.parquet (date, sid, mark, ours, pm)")
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--top", type=int, default=25, help="open columns tried, heaviest by gain first")
    ap.add_argument("--work", type=Path, default=Path("data/missing_damage_work"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    have = {ln.split(" ", 1)[0] for ln in a.formulas.read_text().splitlines() if ln and not ln.startswith("#")}
    files = P.model_files(a.models, a.date, a.fits.split(","), a.subs.split(","))
    g = np.zeros(len(names))
    for f in files:
        g += gains(f, len(names))[0]
    cal = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.loads(a.cal.read_text()).items()}
    x = pl.read_parquet(a.x)
    if "date" in x.columns:
        x = x.filter(pl.col("date") == int(a.date)).drop("date")
    x = x.with_columns(pl.col("sid").cast(pl.Int64)).sort("sid", "mark")
    pr = pl.read_parquet(a.pred).filter(pl.col("date") == int(a.date)).with_columns(pl.col("sid").cast(pl.Int64))
    j = x.select("sid", "mark").join(pr.select("sid", "mark", "pm"), on=["sid", "mark"], how="left")
    pm, marks = j["pm"].to_numpy().astype(float), j["mark"].to_numpy()
    ix = {n: i for i, n in enumerate(names)}
    X = np.full((x.height, len(names)), np.nan)
    for c in x.columns:
        if c in ix:
            X[:, ix[c]] = x[c].to_numpy().astype(float)
    open_ = [i for i in np.argsort(-g) if names[i] not in have][: a.top]
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()

    def score(M: np.ndarray) -> np.ndarray:
        P.write_x(a.work / "x.bin", M)
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files), "--threads", str(a.threads)], check=True,
                       stdout=subprocess.DEVNULL)
        return P.read_scores(a.work / "p.bin").copy()

    base = score(X)
    spread = float(np.nanstd(base))
    s0 = sim(base, pm, marks)
    tot = g.sum()
    print(f"{a.date}: {x.height:,} rows, {sum(c in ix for c in x.columns)} columns filled; similarity with every open column NaN {s0:.4f}; score spread {spread:.4g}")
    print("open column (gain rank, gain share): pooled corr with the NaN scores | mean |change| / spread | similarity with the column filled at random (change)")
    rng = np.random.default_rng(5)
    for i in open_:
        nm = names[i]
        rank = int((g > g[i]).sum()) + 1
        if nm not in cal:
            print(f"  {nm:<14} ({rank:>3}, {g[i] / tot:.4f}): no training-CDF points, skipped")
            continue
        t, p = cal[nm]
        o = np.argsort(p, kind="mergesort")
        t, p = t[o], np.maximum.accumulate(p[o])
        M = X.copy()
        M[:, i] = np.interp(rng.random(X.shape[0]), p, t)  # inverse CDF through the points (clamped at the end points)
        sc = score(M)
        f = np.isfinite(sc) & np.isfinite(base)
        c = float(np.corrcoef(sc[f], base[f])[0, 1])
        s1 = sim(sc, pm, marks)
        print(f"  {nm:<14} ({rank:>3}, {g[i] / tot:.4f}): corr {c:.4f} | change {np.nanmean(np.abs(sc - base)) / spread:.4f} | similarity {s1:.4f} ({s1 - s0:+.4f})")
    (a.work / "x.bin").unlink(missing_ok=True)
    (a.work / "p.bin").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
