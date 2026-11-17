r"""Fingerprint model columns against PM's OWN day-end state keys (.jz), for families whose trees sit next to a state family.

famlinks (R147) links sourceless families to engine families we cannot rebuild intraday yet (zzeu / zzef / zzel next to
zzcb). PM's .jz holds those states at the day's end for every stock, exactly. Candidates: the state-formula grammar of
the settled state columns over the keys matching --key-re (K, K / px, (px - K) / K, (px - K) / px, log(K / px); per pair
K1 / K2, (K1 - K2) / K2, (K1 - K2) / px; px = cdc.lp), each day one cross-section, x transforms x scales. Scored like
seg_fit: share inside the column's training range + threshold / root fingerprint. Day-end cross-sections only: a pass is
a lead on the formula's shape, not a pin (the intraday distribution can differ). Read-only.

    python research/featgen/jz_fit.py --jz-dir /work/crucible_data/seed_probe/HS300 --days 20261110,20261111,20261112,20261113 \
        --target fam:zzeu,fam:zzef,fam:zzel --key-re '^zzcb' --models /work/crucible_data/models_gen/C
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
import pm_models_probe as PP  # noqa: E402
from implied_scan import raw_xs  # noqa: E402
from invert_cols import thresholds  # noqa: E402
from model_atlas import infos  # noqa: E402


def load(path: Path, key_re: re.Pattern, grids: bool) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One .jz -> (key -> value per stock; averages as sum / wgt), px per stock."""
    with gzip.open(path) as f:
        d = json.load(f)
    sids = sorted(d)
    keys = sorted({k for s in sids for k in d[s] if key_re.search(k)})
    col = lambda k: np.array([np.nan if d[s].get(k) is None else float(d[s][k]) for s in sids])  # noqa: E731
    out: dict[str, np.ndarray] = {}
    for k in keys:
        if not grids and re.search(r"\.\d+$", k):
            continue
        if k.endswith(".sum") and f"{k[:-4]}.wgt" in keys:
            w = col(f"{k[:-4]}.wgt")
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k[:-4]] = np.where(w != 0, col(k) / w, np.nan)
        elif not k.endswith(".t"):
            out[k] = col(k)
    return out, col("cdc.lp")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz-dir", type=Path, required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--target", required=True, help="comma list of columns and / or fam:<stem>")
    ap.add_argument("--key-re", required=True)
    ap.add_argument("--grids", action="store_true", help="also the numbered grid keys")
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/C"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1,s2,s3,s4")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--transforms", default="id,-id,1/b,dm,-dm,srank,-srank,abs4,log,sqrt")
    ap.add_argument("--scales", default="1,1e4,100,0.01")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    targets = []
    for t in [x for x in a.target.split(",") if x]:
        targets += [n for n in names if re.match(rf"^{re.escape(t[4:])}\d*(\.|$)", n)] if t.startswith("fam:") else [t]
    ds = [x for x in a.days.split(",") if x and (a.jz_dir / f"{x}.jz").exists()]
    kre = re.compile(a.key_re)
    per_day = [load(a.jz_dir / f"{d}.jz", kre, a.grids) for d in ds]
    keys = sorted(set.intersection(*[set(k) for k, _ in per_day])) if per_day else []
    print(f"days {ds}; {len(keys)} keys matching {a.key_re}: {keys[:40]}")
    if not keys:
        return
    marks = np.concatenate([np.full(px.size, i) for i, (_, px) in enumerate(per_day)])
    px = np.concatenate([p for _, p in per_day])
    K = {k: np.concatenate([kv[k] for kv, _ in per_day]) for k in keys}
    cands: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for k, v in K.items():
            cands[k] = v
            cands[f"{k}/px"] = v / px
            cands[f"(px-{k})/{k}"] = (px - v) / v
            cands[f"(px-{k})/px"] = (px - v) / px
            cands[f"log({k}/px)"] = np.log(np.where(v / px > 0, v / px, np.nan))
        for k1, k2 in itertools.permutations(keys, 2):
            cands[f"{k1}/{k2}"] = K[k1] / K[k2]
            cands[f"({k1}-{k2})/{k2}"] = (K[k1] - K[k2]) / K[k2]
            cands[f"({k1}-{k2})/px"] = (K[k1] - K[k2]) / px
    files = P.model_files(a.models, ds[-1], a.fits.split(","), a.subs.split(","))
    th = thresholds(files, len(names))
    seg = [infos(f) for f in files]
    roots: dict[str, set] = {}
    for f in files:
        try:
            r = PP.parse_model(f)[3]
        except KeyError:
            continue
        for feat, t, p, nl in r.iter_rows():
            j = int(feat[7:]) if feat.startswith("Column_") else ix.get(feat, -1)
            if j >= 0 and not nl:
                roots.setdefault(names[j], set()).add((round(t, 9), round(p, 4)))
    trs = [t for t in a.transforms.split(",") if t]
    scs = [float(s) for s in a.scales.split(",")]
    tg = [x for x in targets if x in ix and th[ix[x]].size]
    print(f"{len(cands)} candidates x {len(trs)} transforms x {len(scs)} scales, {len(tg)} targets, {px.size} stock-days, {len(files)} model files")
    rows: dict[str, list] = {x: [] for x in tg}
    for b, v0 in cands.items():
        v0 = np.where(np.isfinite(v0), v0, np.nan)
        for tr in trs:
            with np.errstate(divide="ignore", invalid="ignore"):
                w = v0 if tr == "id" else -v0 if tr == "-id" else 1.0 / v0 if tr == "1/b" else raw_xs(v0, marks, tr)
            w = np.sort(w[np.isfinite(w)])
            if w.size < 200:
                continue
            for sc in scs:
                v = w * sc if sc > 0 else w
                for x in tg:
                    j = ix[x]
                    rg = [s[j] for s in seg if s[j]]
                    if not rg:
                        continue
                    lo_u, hi_u = min(r[0] for r in rg), max(r[1] for r in rg)
                    t = th[j]
                    tol = 1e-9 * max(abs(lo_u), abs(hi_u), 1e-12)
                    inr = (np.searchsorted(v, hi_u + tol, side="right") - np.searchsorted(v, lo_u - tol, side="left")) / v.size
                    cut = np.searchsorted(v, t, side="left")
                    lo, hi = cut[0] / v.size, 1 - np.searchsorted(v, t[-1], side="right") / v.size
                    occ = np.count_nonzero(np.diff(np.concatenate([[0], cut, [v.size]]))) / (t.size + 1)
                    gap = max((abs(np.searchsorted(v, q, side="right") / v.size - p) for q, p in roots.get(x, ())), default=0.0)
                    cover = (v[-1] - v[0]) / (hi_u - lo_u) if hi_u > lo_u else 0.0
                    fp = lo + hi + (1 - occ) + 2 * gap
                    rows[x].append((2 * (1 - inr) + fp, b, tr, sc, inr, fp, lo, hi, occ, gap, cover))
    for x in tg:
        j = ix[x]
        rg = [s[j] for s in seg if s[j]]
        rows[x].sort()
        print(f"-- {x}: range [{min(r[0] for r in rg):.5g}, {max(r[1] for r in rg):.5g}]; thresholds {th[j].size}; root points {len(roots.get(x, ()))}; best of {len(rows[x])}:")
        for sc_, b, tr, s, inr, fp, lo, hi, occ, gap, cover in rows[x][: a.top]:
            print(f"   {sc_:6.3f} {tr:>6} x{s:<6g} {b[:64]:<64} in {inr:.3f} fp {fp:.3f} (below {lo:.3f} above {hi:.3f} int {occ:.2f} root {gap:.3f}) cover {cover:.2f}")


if __name__ == "__main__":
    main()
