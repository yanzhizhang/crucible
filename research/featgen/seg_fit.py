r"""Target-free pinning test per open column: in-range share against PM's training ranges + the threshold fingerprint.

The acceptance rule for "pinned" (1) >= 99% of the candidate's values inside the column's feature_infos range (union
over the sub-models s1..s4, plus the worst single segment), (2) the threshold / root-CDF fingerprint (as fingerprint.py,
thresholds pooled over s1..s4), (3) the source is a bit-exact PM state or profile (--exact-csv restricts engine-key
candidates to keys verified bit-exact). Candidates: every base / --extra column (optionally filtered by --cand-re),
ratios num / den built from --ratio "DENRE=NUMRE" pairs (E7: live quantity / profile cell), and session clocks (E5).
Transforms: the RAW set x scales. Ranked by 2 * (1 - in_range) + fingerprint score; scored against PM only afterwards.

    python research/featgen/seg_fit.py --days 20260430,20261112 --target zzop2.b,zzag2.e --x-dir "data/reports/stage6_r99_{mmdd}" \
        --base "data/featgen_r99/base_{date}.parquet" --models /work/crucible_data/models_gen/B --clocks
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
import pm_models_probe as PP  # noqa: E402
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from invert_cols import thresholds  # noqa: E402
from model_atlas import infos  # noqa: E402

KEY_RE = re.compile(r"zz[a-z0-9]+(?:\.[A-Za-z0-9#]+)+")


def clocks(marks: np.ndarray) -> dict[str, np.ndarray]:
    """Session clocks from the trading-clock mark (s since 09:30, lunch removed): per-session, since 09:15, capped."""
    m = marks.astype(float)
    am = m <= 7200
    out = {"clk.since0930": m, "clk.session": np.where(am, m, m - 7200), "clk.since0915_session": np.where(am, m + 900, m - 7200),
           "clk.to_close": 14400 - m, "clk.to_session_end": np.where(am, 7200 - m, 14400 - m), "clk.min_since0930": m / 60,
           "clk.since0915_wall": np.where(am, m + 900, m + 900 + 5400)}
    for cap in (600, 900, 1800, 3600):
        out[f"clk.session_cap{cap}"] = np.minimum(out["clk.session"], cap)
    return out


PEER = {"dmi", "-dmi", "neu", "-neu"}  # need the day's CED (Barra industry / exposures): --ced-dc


def ced_rows(D, path_tmpl: str, d: str) -> None:
    """Per scoring row: its Barra industry (argmax of the 0/1 dummies, as ic_score) and its exposure vector."""
    import glob
    hits = sorted(glob.glob(path_tmpl.format(date=d, mmdd=d[4:])))
    D.ind = D.E = None
    if not hits:
        print(f"  {d}: no CED dc.parquet for {path_tmpl}: dmi / neu unavailable")
        return
    dc = pl.read_parquet(hits[-1]).with_columns(pl.col("sid").cast(pl.Int64))
    ex = [c for c in dc.columns if c.startswith("exp.")]
    for c in ex:
        fv = dc[c].drop_nulls().drop_nans() if dc[c].dtype in (pl.Float32, pl.Float64) else dc[c].drop_nulls()
        if fv.len() and fv.is_in([0.0, 1.0]).all():
            dc = dc.with_columns(pl.col(c).fill_null(0.0).fill_nan(0.0))
    j = D.key.select("sid").join(dc, on="sid", how="left")
    dums = [c for c in ex if j[c].drop_nulls().is_in([0.0, 1.0]).all() and 0 < (j[c].mean() or 0) < 0.5]
    if len(dums) >= 5:
        st = np.nan_to_num(np.stack([j[c].to_numpy().astype(float) for c in dums]))
        D.ind = np.where(st.max(axis=0) == 1, st.argmax(axis=0), -1)
    ek = [c for c in ex if (j[c].std() or 0) > 0]
    D.E = np.column_stack([j[c].to_numpy().astype(float) for c in ek]) if ek else None
    print(f"  {d}: CED {Path(hits[-1]).parent.name}: {len(dums)} industry dummies, {len(ek)} exposures")


def peer(v: np.ndarray, D, tr: str) -> np.ndarray:
    """dmi: minus the industry mean per mark; neu: residual on [1, exposures] per mark (ic_score's demean / neutralize)."""
    from ic_score import demean, neutralize
    out = np.full(v.size, np.nan)
    if (tr.endswith("dmi") and D.ind is None) or (tr.endswith("neu") and D.E is None):
        return out
    for mk in np.unique(D.marks):
        m = D.marks == mk
        out[m] = demean(v[m], D.ind[m]) if tr.endswith("dmi") else neutralize(v[m], D.E[m])
    return -out if tr.startswith("-") else out


def apply(v: np.ndarray, D, tr: str, sc: float) -> np.ndarray:
    """RAW (or peer-relative) transform then scale."""
    if tr in PEER:
        return peer(v, D, tr) * sc
    with np.errstate(divide="ignore", invalid="ignore"):
        w = v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
    return w * sc


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--target", required=True, help="comma list, or audit.csv:<source> (e.g. data/reports/audit/audit.csv:engine)")
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--no-base", action="store_true", help="candidates only from --extra / ratios / clocks")
    ap.add_argument("--cand-re", default="", help="keep only candidates matching this regex")
    ap.add_argument("--ratio", action="append", default=[], help="DENRE=NUMRE: add num/den for every matching pair")
    ap.add_argument("--max-ratios", type=int, default=6000)
    ap.add_argument("--clocks", action="store_true")
    ap.add_argument("--exact-csv", type=Path, default=None, help="verify.py csv: engine-key candidates must use bit-exact keys only")
    ap.add_argument("--exact-min", type=float, default=0.987)
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1,s2,s3,s4")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--transforms", default="id,-id,1/b,dm,-dm,srank,log")
    ap.add_argument("--scales", default="1,1e4,1e-4,100,0.01")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--ced-dc", default="/work/crucible_data/cache/ic_build/*/date={date}.*/dc.parquet",
                    help="glob of the day's CED daily table (ic_score's build cache), for dmi / neu transforms")
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    if a.target.startswith(("audit.csv", "data/")) and ":" in a.target:
        path, src = a.target.rsplit(":", 1)
        au = pl.read_csv(path)
        targets = au.filter((pl.col("source").str.contains(src)) & (pl.col("status").is_in(["open", "stand-in"])))["col"].to_list()
    elif a.target.startswith("fam:"):  # every column of one trunk: fam:zzwa -> zzwa*, zzwa2* ...
        stem = a.target[4:]
        targets = [n for n in names if re.match(rf"^{re.escape(stem)}\d*(\.|$)", n)]
    else:
        targets = [t for t in a.target.split(",") if t]
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    ext = [x for x in a.extra.split(",") if x]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
                   [Path(e.format(date=d, mmdd=d[4:])) for e in ext]) for d in ds}
    files = P.model_files(a.models, ds[0], a.fits.split(","), a.subs.split(","))
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
    exact = None
    if a.exact_csv and a.exact_csv.exists():
        exact = {re.sub(r"\|.*$", "", r["key"]) for r in pl.read_csv(a.exact_csv).iter_rows(named=True) if float(r.get("all_1e4") or r.get("all") or 0) >= a.exact_min}
    trs = [t for t in a.transforms.split(",") if t in RAW or t in PEER]
    for d, D in days.items():  # derived candidates
        D.ind = D.E = None
        if PEER & set(trs):
            ced_rows(D, a.ced_dc, d)
        if a.no_base:
            D.bases = {k: v for k, v in D.bases.items() if k.startswith("x.")}
        if a.clocks:
            D.bases.update(clocks(D.marks))
        for spec in a.ratio:
            den_re, num_re = spec.split("=", 1)
            dens = [k for k in D.bases if re.search(den_re, k)]
            nums = [k for k in D.bases if re.search(num_re, k) and k not in dens]
            n = 0
            for dk in dens:
                for nk in nums:
                    if n >= a.max_ratios:
                        break
                    with np.errstate(divide="ignore", invalid="ignore"):
                        D.bases[f"r.({nk})/({dk})"] = D.bases[nk] / D.bases[dk]
                    n += 1
    bases = sorted(set.intersection(*[set(D.bases) for D in days.values()]))
    if a.cand_re:
        bases = [b for b in bases if re.search(a.cand_re, b) or b.startswith(("clk.", "r."))]
    if exact is not None:
        bases = [b for b in bases if all(k in exact or k.rsplit(".", 1)[0] in exact for k in KEY_RE.findall(b))]
    scs = [float(s) for s in a.scales.split(",")]
    tg = [x for x in targets if x in ix and th[ix[x]].size]
    print(f"{len(bases)} candidates x {len(trs)} transforms x {len(scs)} scales, {len(tg)} targets, days {ds}, {len(files)} model files")
    rows: dict[str, list] = {x: [] for x in tg}
    for b in bases:
        for tr in trs:
            base_vs = []
            for D in days.values():
                v = apply(D.bases[b], D, tr, 1.0)
                v = np.sort(v[np.isfinite(v)])
                if v.size < 1000:
                    break
                base_vs.append(v)
            if len(base_vs) != len(days):
                continue
            for sc in scs:
                vs = [v * sc for v in base_vs]
                for x in tg:
                    j = ix[x]
                    rg = [s[j] for s in seg if s[j]]
                    if not rg:
                        continue
                    lo_u, hi_u = min(r[0] for r in rg), max(r[1] for r in rg)
                    t = th[j]
                    ins, ins_seg, per = [], [], []
                    for v in vs:
                        tol = 1e-9 * max(abs(lo_u), abs(hi_u), 1e-12)
                        inside = (np.searchsorted(v, hi_u + tol, side="right") - np.searchsorted(v, lo_u - tol, side="left")) / v.size
                        ins.append(inside)
                        ins_seg.append(min((np.searchsorted(v, r[1] + tol, side="right") - np.searchsorted(v, r[0] - tol, side="left")) / v.size for r in rg))
                        cut = np.searchsorted(v, t, side="left")
                        lo, hi = cut[0] / v.size, 1 - np.searchsorted(v, t[-1], side="right") / v.size
                        occ = np.count_nonzero(np.diff(np.concatenate([[0], cut, [v.size]]))) / (t.size + 1)
                        gap = max((abs(np.searchsorted(v, q, side="right") / v.size - p) for q, p in roots.get(x, ())), default=0.0)
                        per.append((lo, hi, occ, gap))
                    lo, hi, occ, gap = (max(p[i] for p in per) if i != 2 else min(p[i] for p in per) for i in range(4))
                    fp = lo + hi + (1 - occ) + 2 * gap
                    inr = min(ins)
                    rows[x].append((2 * (1 - inr) + fp, b, tr, sc, inr, min(ins_seg), fp, lo, hi, occ, gap))
    for x in tg:
        j = ix[x]
        rg = [s[j] for s in seg if s[j]]
        rows[x].sort()
        nr = len(roots.get(x, ()))
        print(f"-- {x}: range union [{min(r[0] for r in rg):.5g}, {max(r[1] for r in rg):.5g}] over {len(rg)} segs; thresholds {th[j].size}; root points {nr}; best of {len(rows[x])}:")
        for sc_, b, tr, s, inr, inseg, fp, lo, hi, occ, gap in rows[x][: a.top]:
            ok = "PASS(1,2)" if inr >= 0.99 and fp < 0.05 and (nr > 0) else "pass(1,2) no-root" if inr >= 0.99 and fp < 0.05 else ""
            print(f"   {sc_:6.3f} {tr:>5} x{s:<6g} {b[:70]:<70} in {inr:.3f} worst-seg {inseg:.3f} fp {fp:.3f} (below {lo:.3f} above {hi:.3f} int {occ:.2f} root {gap:.3f}) {ok}")


if __name__ == "__main__":
    main()
