r"""Scale check of every settled column: does OUR value's range and scale class agree with the model's training range?

The model file gives each column's training [min, max] and split thresholds (model_atlas units). A stand-in whose own
values fall in another scale class, sit outside the training range, or cover only a sliver of it is wrong whatever its
similarity score. Per settled column, pooled over the days: our [min, max] and 0.1% / 99.9% quantiles, the share of our
values outside the training range, the part of the training range our central 99.8% covers, the share of the model's
thresholds inside our central 99.8% (thresholds we never cross are dead splits), and both scale classes. Read-only.

    python research/featgen/unit_check.py --days 20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" --models /work/crucible_data/models_gen/C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402
from invert_cols import BASELINE, thresholds  # noqa: E402
from model_atlas import SUBS, infos, settled_formulas, unit_class  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--x-dir", required=True, help="dir with x.parquet (filled scoring matrix), {mmdd} / {date} allowed")
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/C"))
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--out", type=Path, default=Path("data/reports/unit_check"))
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    formulas = settled_formulas(a.formulas)
    ds = [x for x in a.days.split(",") if x]
    files = P.model_files(a.models, ds[0], [a.fit], SUBS)
    th = thresholds(files, len(names))
    seg = [infos(f) for f in files]
    xs = []
    for d in ds:
        x = pl.read_parquet(Path(a.x_dir.format(mmdd=d[4:], date=d)) / "x.parquet")
        if "date" in x.columns:
            x = x.filter(pl.col("date") == int(d))
        xs.append(x)
    rows = []
    for i, n in enumerate(names):
        rg = [s[i] for s in seg if s[i]]
        if n not in formulas or not rg or not all(n in x.columns for x in xs):
            continue
        v = np.concatenate([x[n].cast(pl.Float64).to_numpy() for x in xs])
        v = v[np.isfinite(v)]
        if v.size < 1000:
            continue
        lo, hi = min(r[0] for r in rg), max(r[1] for r in rg)
        t = th[i]
        q0, q1 = np.quantile(v, [0.001, 0.999])
        tol = 1e-9 * max(abs(lo), abs(hi), 1e-12)
        out_share = float(((v < lo - tol) | (v > hi + tol)).mean())
        cover = (min(q1, hi) - max(q0, lo)) / (hi - lo) if hi > lo else np.nan
        live = float(((t >= q0) & (t <= q1)).mean()) if t.size else np.nan
        cm, co = unit_class(lo, hi, t), unit_class(float(v.min()), float(v.max()), t)
        bad = (cm != co) + (out_share > 0.01) + (cover < 0.2) + (live < 0.5 if t.size else 0)
        rows.append({"col": n, "bad": int(bad), "model_lo": lo, "model_hi": hi, "our_lo": float(v.min()), "our_hi": float(v.max()), "q001": q0, "q999": q1,
                     "outside": out_share, "cover": cover, "thr_live": live, "n_thr": int(t.size), "model_cls": cm, "our_cls": co, "formula": formulas[n]})
    t = pl.DataFrame(rows).sort(["bad", "outside"], descending=True)
    a.out.mkdir(parents=True, exist_ok=True)
    t.write_csv(a.out / "unit_check.csv")
    print(f"== unit_check: {t.height} settled columns, days {ds}, model {a.models.name}; flags: class differs / >1% outside / covers <20% / <50% thresholds live")
    print(f"   flagged {t.filter(pl.col('bad') > 0).height}: by count of flags {dict(sorted(t.group_by('bad').len().iter_rows()))}")
    for r in t.filter(pl.col("bad") > 0).iter_rows(named=True):
        print(f"   {r['col']:<14} f{r['bad']} model [{r['model_lo']:.4g}, {r['model_hi']:.4g}] ours [{r['q001']:.4g}, {r['q999']:.4g}] out {r['outside']:.3f} cover {r['cover']:.2f}"
              f" live {r['thr_live']:.2f} | {r['model_cls'].split(' (')[0]} / {r['our_cls'].split(' (')[0]} | {r['formula'][:44]}")
    print(f"[done] csv {a.out / 'unit_check.csv'}")


if __name__ == "__main__":
    main()
