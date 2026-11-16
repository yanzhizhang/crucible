r"""One trunk (family stem, e.g. zzwa) on one page: everything PM's files say about it, before guessing its meaning.

(a) every .jz engine key of the stem: value / weight / .t quantiles over stocks and the half-life the weight implies;
(b) the hstats profile of the stem, if any: its files and variables;
(c) its model columns: gain share, per-segment feature_infos ranges, threshold count and quantiles, root points, and
    the columns PM's trees split next to them most (lift);
(d) the settled formulas of the family and any earlier round report lines that mention its columns (--reports glob).
Read-only; prints one compact summary.

    python research/featgen/family_dossier.py zzwa --jz /work/crucible_data/seed_probe/HS300/20261113.jz --models /work/crucible_data/models_gen/B
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import pred_day as P  # noqa: E402
import pm_models_probe as PP  # noqa: E402
from col_probe import trees, walk  # noqa: E402
from gain_share import gains  # noqa: E402
from invert_cols import BASELINE  # noqa: E402
from model_atlas import SUBS, infos, settled_formulas  # noqa: E402


def q3(v: np.ndarray) -> str:
    """q10 / q50 / q90 as text."""
    v = v[np.isfinite(v)]
    return " ".join(f"{x:.5g}" for x in np.quantile(v, [0.1, 0.5, 0.9])) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stem", help="family stem without digits, e.g. zzwa (matches zzwa, zzwa2 ...)")
    ap.add_argument("--jz", type=Path, default=Path("/work/crucible_data/seed_probe/HS300/20261113.jz"))
    ap.add_argument("--hstats", type=Path, default=Path("/work/crucible_data/hstats_view/HS300"))
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--date", default="20261113")
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--reports", default="data/reports/r13[0-9]*.txt")
    ap.add_argument("--dt", type=float, default=3.0)
    a = ap.parse_args()
    stem = a.stem.lower()
    fam_re = re.compile(rf"^{re.escape(stem)}\d*(\.|$)")
    names = P.feature_names(a.feature_csv)
    formulas = settled_formulas(a.formulas)
    cols = [(i, n) for i, n in enumerate(names) if fam_re.match(n)]
    print(f"######## {stem}: {len(cols)} model columns")
    # (a) engine keys
    if a.jz.exists():
        st = json.loads(gzip.open(a.jz).read())
        keys = sorted({re.sub(r"\.(sum|wgt|t)$", "", k) for kv in st.values() for k in kv if fam_re.match(k)})
        print(f"(a) .jz keys ({a.jz.name}): {len(keys)}")
        for k in keys:
            def get(suf: str) -> np.ndarray:
                return np.array([kv.get(f"{k}{suf}", np.nan) for kv in st.values()], float)
            if np.isfinite(get(".wgt")).any():
                s, w, t = get(".sum"), get(".wgt"), get(".t")
                with np.errstate(divide="ignore", invalid="ignore"):
                    val = s / w
                wm = np.nanmedian(w)
                h = -a.dt / np.log2(1 - 1 / wm) if wm > 1 else 0.0
                print(f"   {k:<22} EMA  value {q3(val):<34} wgt {q3(w):<26} h~{h:.4g}s  t {q3(t)}")
            else:
                v = get("")
                print(f"   {k:<22} raw  value {q3(v):<34} finite {np.isfinite(v).mean():.2f}")
    # (b) hstats profile
    hs = [p for p in a.hstats.iterdir() if p.is_dir() and re.fullmatch(rf"{re.escape(stem)}\d*", p.name.lower())] if a.hstats.exists() else []
    print(f"(b) hstats profiles: {[p.name for p in hs] or 'none'}")
    for p in hs:
        fs = sorted(p.rglob("*"))
        print(f"   {p.name}: {len(fs)} entries, e.g. {[f.name for f in fs[:4]]}")
        pq = [f for f in fs if f.suffix == ".parquet"]
        if pq:
            import polars as pl
            sch = pl.read_parquet_schema(pq[-1])
            print(f"   columns of {pq[-1].name}: {list(sch)[:30]}")
    # (c) model columns
    files = P.model_files(a.models, a.date, [a.fit], SUBS)
    seg = [infos(f) for f in files]
    g = np.zeros(len(names))
    for f in files:
        g += gains(f, len(names))[0]
    g = g / g.sum() if g.sum() else g
    th: list[list[float]] = [[] for _ in names]
    pair: Counter = Counter()
    deg: Counter = Counter()
    for f in files:
        for kv in trees(f):
            sf = [int(x) for x in kv["split_feature"]]
            for j, t, dt in zip(sf, kv["threshold"], kv["decision_type"], strict=True):
                if not int(dt) & 1:
                    th[j].append(float(t))
            _, parent = walk(kv)
            for nd, p in enumerate(parent):
                if p >= 0 and sf[p] != sf[nd]:
                    pair[tuple(sorted((sf[p], sf[nd])))] += 1
                    deg[sf[p]] += 1
                    deg[sf[nd]] += 1
    N = sum(pair.values())
    roots: dict[int, list] = {}
    for f in files:
        try:
            r = PP.parse_model(f)[3]
        except KeyError:
            continue
        for feat, t, pp_, nl in r.iter_rows():
            j = int(feat[7:]) if feat.startswith("Column_") else -1
            if j >= 0 and not nl:
                roots.setdefault(j, []).append((round(t, 7), round(pp_, 3)))
    print(f"(c) columns (gain share of all 209; ranges s1..s4; thresholds; root points; top neighbours by lift)")
    tot = 0.0
    for i, n in cols:
        tot += g[i]
        t = np.unique(th[i])
        rg = " ".join(f"[{s[i][0]:.4g},{s[i][1]:.4g}]" if s[i] else "none" for s in seg)
        nb = sorted(((pair[tuple(sorted((i, j)))] * 2 * N / (deg[i] * deg[j]) if deg[i] and deg[j] else 0, j) for j in range(len(names))
                     if j != i and pair[tuple(sorted((i, j)))] >= 5), reverse=True)[:4]
        rp = sorted(set(roots.get(i, [])))
        print(f"   {n:<14} gain {g[i]:.4f} {'SETTLED ' + formulas[n] if n in formulas else 'open'}")
        print(f"      ranges {rg}")
        print(f"      thr {t.size} q10/50/90 {q3(t)}  roots {len(rp)} {rp[:4]}  nb {' '.join(f'{names[j]}:{l:.1f}' for l, j in nb)}")
    print(f"   family gain {tot:.4f}")
    # (d) earlier reports
    hits = []
    for rep in sorted(glob.glob(a.reports)):
        for ln in Path(rep).read_text(errors="replace").splitlines():
            if any(re.search(rf"(^|[^a-z0-9.]){re.escape(n)}([^a-z0-9.]|$)", ln) for _, n in cols) and ("--" in ln or "PASS" in ln or "pass" in ln):
                hits.append(f"   {Path(rep).name}: {ln.strip()[:170]}")
    print(f"(d) report lines mentioning the family ({len(hits)}):")
    print("\n".join(hits[:40]))
    print(f"######## {stem} done")


if __name__ == "__main__":
    main()
