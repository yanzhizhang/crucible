r"""Check the column order: each model column's training range (feature_infos) against our values under the name we map it to.

The model header names its inputs Column_0..Column_208 and we name them by the row order of lgb_fn_v1_209_cte.csv
(pred_day.feature_names). Indirect evidence says the order is right (PM's trees on our matrix reach 0.63 similarity with
PM), but it was never checked column by column. Per column: feature_infos [min:max], our values' 1% / 99% quantiles and
the share inside [min, max] on one scoring matrix; settled columns far outside are flagged. Also prints the csv's first
lines (what else it holds besides the name). Read-only.

    python research/range_check.py --x data/reports/stage6_r99_1113/x.parquet --models /work/crucible_data/models_gen/B --date 20261113
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "featgen"))
from pred_day import feature_names, model_files  # noqa: E402

BASELINE = HERE / "engine" / "baseline_r41c.txt"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--feature-csv", type=Path, default=Path("/work/crucible_data/prod_models/lgb_fn_v1_209_cte.csv"))
    a = ap.parse_args()
    txt = a.feature_csv.read_text(errors="replace").splitlines()
    print(f"{a.feature_csv}: {len(txt)} lines; first 5: {txt[:5]}; last 2: {txt[-2:]}")
    names = feature_names(a.feature_csv)
    f = model_files(a.models, a.date, ["livev1_fit1"], ["s1"])[0]
    head = f.read_text(errors="replace").split("\nTree=0", 1)[0]
    info = dict(ln.split("=", 1) for ln in head.splitlines() if "=" in ln)
    rng = info.get("feature_infos", "").split()
    fn = info.get("feature_names", "").split()
    print(f"model {f}: {len(fn)} feature_names (first {fn[:3]}), {len(rng)} feature_infos; csv names {len(names)}")
    settled = {ln.split()[0] for ln in BASELINE.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    x = pl.read_parquet(a.x)
    bad, rows = [], []
    for i, n in enumerate(names):
        m = re.match(r"\[(.+):(.+)\]", rng[i]) if i < len(rng) else None
        if not m or n not in x.columns:
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        v = x[n].cast(pl.Float64).to_numpy()
        v = v[np.isfinite(v)]
        if v.size < 100:
            continue
        q1, q99 = np.quantile(v, [0.01, 0.99])
        inside = float(((v >= lo - 1e-9 * abs(lo)) & (v <= hi + 1e-9 * abs(hi))).mean())
        span = hi - lo if hi > lo else 1.0
        off = max(0.0, (lo - q1) / span, (q99 - hi) / span)
        rows.append((off, i, n, lo, hi, q1, q99, inside, n in settled))
        if n in settled and (inside < 0.9 or off > 0.5):
            bad.append(n)
    print(f"{len(rows)} columns with values; settled columns flagged (inside < 0.9 or 1-99% beyond the range by > half its width): {len(bad)}")
    for off, i, n, lo, hi, q1, q99, inside, st in sorted(rows, reverse=True)[:25]:
        print(f"  #{i:<3} {n:<10} {'settled' if st else 'open   '} range [{lo:.5g}, {hi:.5g}]  ours 1%..99% [{q1:.5g}, {q99:.5g}]  inside {inside:.3f}")
    ok = [r for r in rows if r[8] and r[7] >= 0.99]
    print(f"settled columns with >= 99% of our values inside the training range: {len(ok)} of {sum(r[8] for r in rows)}")


if __name__ == "__main__":
    main()
