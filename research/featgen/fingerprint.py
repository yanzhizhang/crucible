r"""Units fingerprint: does a raw candidate's distribution sit where PM's trees put the column's thresholds?

Single-column similarity scans have little power (R96 stage 2 / R105: even good fills move one day by ~0.001), so this is
a target-free second ruler. LightGBM cuts a column only at its histogram bin edges (quantiles of the training data), so
the thresholds it used span the training distribution: a candidate in the right units puts few rows below the lowest /
above the highest threshold and lands in most of the threshold intervals; and where the trees split at the root, the
share of training rows left of the threshold P(x <= t) is known exactly. Per open column x candidate x raw transform:
tail shares below / above, the share of threshold intervals the candidate occupies, and the gap to every root CDF point;
ranked by a combined score (tails + uncovered intervals + root gaps). Read-only, no scoring runs.

    python research/featgen/fingerprint.py --days 20260430,20261112 --x-dir "data/reports/stage6_r99_{mmdd}" \
        --base "data/featgen_r99/base_{date}.parquet" --extra "data/side_r105/ratio_{date}.parquet" --models $DATA/models_gen/B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
import pm_models_probe as PP  # noqa: E402
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from invert_cols import BASELINE, thresholds  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--target", default="open", help="comma list, or 'open' = every column without a --formulas line")
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--transforms", default="id,-id,1/b,dm,-dm,srank,abs4")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    settled = {ln.split()[0] for ln in a.formulas.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    targets = [n for n in names if n not in settled] if a.target == "open" else [t for t in a.target.split(",") if t]
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    ext = [x for x in a.extra.split(",") if x]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
                   [Path(e.format(date=d, mmdd=d[4:])) for e in ext]) for d in ds}
    files = P.model_files(a.models, ds[0], a.fits.split(","), a.subs.split(","))
    th = thresholds(files, len(names))
    roots = {}
    for f in files:
        try:
            r = PP.parse_model(f)[3]
        except KeyError:  # a model text without split_gain (the synthetic test): no root points
            continue
        for feat, t, p, nl in r.iter_rows():
            j = int(feat[7:]) if feat.startswith("Column_") else (ix.get(feat, -1))
            if j >= 0 and not nl:
                roots.setdefault(names[j], set()).add((round(t, 9), round(p, 4)))
    bases = sorted(set.intersection(*[set(D.bases) for D in days.values()]))
    trs = [t for t in a.transforms.split(",") if t in RAW]
    print(f"{len(bases)} candidates x {len(trs)} transforms, days {ds}")
    tg = [x for x in targets if x in ix and th[ix[x]].size]
    rows: dict[str, list] = {x: [] for x in tg}
    for b in bases:  # each candidate transformed once per day, then held against every target
        for tr in trs:
            vs = []
            for D in days.values():
                v = D.bases[b]
                with np.errstate(divide="ignore", invalid="ignore"):
                    v = v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
                v = np.sort(v[np.isfinite(v)])
                if v.size < 1000:
                    break
                vs.append(v)
            if len(vs) != len(days):
                continue
            for x in tg:
                t = th[ix[x]]
                per = []
                for v in vs:
                    cut = np.searchsorted(v, t, side="left")  # rows below each threshold
                    lo, hi = cut[0] / v.size, 1 - np.searchsorted(v, t[-1], side="right") / v.size
                    occ = (np.count_nonzero(np.diff(np.concatenate([[0], cut, [v.size]]))) ) / (t.size + 1)
                    gap = max((abs(np.searchsorted(v, q, side="right") / v.size - p) for q, p in roots.get(x, ())), default=0.0)
                    per.append((lo, hi, occ, gap))
                lo, hi, occ, gap = (max(p[i] for p in per) if i != 2 else min(p[i] for p in per) for i in range(4))
                rows[x].append((lo + hi + (1 - occ) + 2 * gap, b, tr, lo, hi, occ, gap))
    for x in tg:
        t = th[ix[x]]
        rp = sorted(roots.get(x, []))
        rows[x].sort()
        print(f"-- {x}: thresholds {t.size} in [{t[0]:.5g}, {t[-1]:.5g}], root points {rp[:3]}; best of {len(rows[x])}:")
        for sc, b, tr, lo, hi, occ, gap in rows[x][: a.top]:
            print(f"   {sc:6.3f}  {tr:>6} {b:<48} below {lo:.3f} above {hi:.3f} intervals {occ:.2f} root gap {gap:.3f}")


if __name__ == "__main__":
    main()
