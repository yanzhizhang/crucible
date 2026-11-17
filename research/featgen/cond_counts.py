r"""Conditional-count test: a candidate for column k must reproduce the shares PM's training rows took at every node.

LightGBM stores internal_count / leaf_count per node, so each internal node that splits on k gives
P(x_k <= t | the path's conditions) on PM's training rows. The root gives the marginal CDF point (what fingerprint uses);
deeper nodes give conditional points of the joint distribution with the path's columns. A correlated proxy can match
the marginal but not the conditionals under many different parents. Usable nodes: every ancestor on the path splits
on a TRUSTED column (one we hold exactly: --trusted, default the four PM-state columns) or on k itself. Per candidate:
the count-weighted mean |our share - PM's share| over those nodes (missing values follow the node's default side as
PM's 'none' missing type does: NaN -> 0). --calibrate hides each trusted column in turn and ranks its true values
(the matrix column) among the candidates, to show the test's power. Read-only.

--trusted all (R155) trusts EVERY column of our matrix on the paths: many more nodes, and the rows reaching a node are
then ours, wrong columns included -- so the absolute error has a floor, but candidates for ONE column are compared on the
same rows, and the right one should still come first (calibrate with the PM-state columns as targets and --pool).
--pool adds every base x --pool-tr x --pool-scales that lands inside the column's training range (>= --pool-in of its
values) and covers >= 20% of it, so the search is not limited to hand-picked candidates.

    python research/featgen/cond_counts.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" \
        --target zzro.a --cands "dmi:vfrac10@0.01,dm:vfrac10@0.01,neu:vfrac10@0.01" --models /work/crucible_data/models_gen/B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from col_probe import trees  # noqa: E402
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from model_atlas import infos  # noqa: E402
from seg_fit import PEER, ced_rows, peer  # noqa: E402

TRUSTED = "zzup,zzvx.a,zzxu.a,zzxu.b"


def nodes(files: list[Path], trusted: set[int], k: int) -> list[tuple[list[tuple[int, float, bool]], float, float, int]]:
    """Every internal node splitting on k whose path uses only trusted columns or k: (path, threshold, PM left share, count)."""
    out = []
    for f in files:
        for kv in trees(f):
            sf = [int(x) for x in kv["split_feature"]]
            th = [float(x) for x in kv["threshold"]]
            dt = [int(x) for x in kv["decision_type"]]
            lc, rc = [int(x) for x in kv["left_child"]], [int(x) for x in kv["right_child"]]
            ic = [int(float(x)) for x in kv.get("internal_count", [])]
            lfc = [int(float(x)) for x in kv.get("leaf_count", [])]
            if not ic or not lfc:
                continue

            def cnt(c: int) -> int:
                return ic[c] if c >= 0 else lfc[~c]

            stack = [(0, [])]
            while stack:
                n, path = stack.pop()
                if dt[n] & 1:  # categorical: not handled
                    continue
                if sf[n] == k and ic[n] > 0:
                    out.append((path, th[n], cnt(lc[n]) / ic[n], ic[n]))
                if sf[n] not in trusted and sf[n] != k:
                    continue  # below an untrusted split the rows are not reproducible
                for c, left in ((lc[n], True), (rc[n], False)):
                    if c >= 0:
                        stack.append((c, path + [(sf[n], th[n], left, bool(dt[n] & 2))]))
    return out


class Prep:
    """Per target: each usable node's rows satisfying its trusted-column conditions, computed once on a row sample."""

    def __init__(self, nd, X: np.ndarray, k: int, rows: np.ndarray, max_nodes: int) -> None:
        nd = sorted(nd, key=lambda x: -x[3])[:max_nodes]
        Z = np.nan_to_num(X[rows], nan=0.0)
        self.rows, self.k, self.items = rows, k, []
        for path, t, share, c in nd:
            m = np.ones(rows.size, bool)
            kc = []
            for j, tj, left, _ in path:
                if j == k:
                    kc.append((tj, left))
                else:
                    m &= (Z[:, j] <= tj) if left else (Z[:, j] > tj)
            idx = np.flatnonzero(m).astype(np.int32)
            if idx.size >= 200:
                self.items.append((idx, kc, t, share, c, not path))

    def score(self, v: np.ndarray, which: str = "all") -> tuple[float, int]:
        """Count-weighted mean |share(v <= t | path) - PM share|; which = all / root / cond. NaN -> 0 (missing type none)."""
        vk = np.nan_to_num(v[self.rows], nan=0.0)
        err = wsum = 0.0
        used = 0
        for idx, kc, t, share, c, is_root in self.items:
            if (which == "root" and not is_root) or (which == "cond" and is_root):
                continue
            x = vk[idx]
            for tj, left in kc:
                x = x[x <= tj] if left else x[x > tj]
            if x.size < 200:
                continue
            err += c * abs(float((x <= t).mean()) - share)
            wsum += c
            used += 1
        return (err / wsum if wsum else np.nan), used


_PREP: Prep | None = None


def _one(item: tuple[str, np.ndarray]) -> tuple:
    spec, v = item
    e_all, n_all = _PREP.score(v, "all")
    e_root, _ = _PREP.score(v, "root")
    e_cond, n_cond = _PREP.score(v, "cond")
    return (e_cond if np.isfinite(e_cond) else e_all, spec, e_all, e_root, e_cond, n_all, n_cond)


def cand_values(D, spec: str) -> tuple[str, np.ndarray | None]:
    """'tr:base@scale' -> values; 'X:<col>' = the matrix's current column."""
    if spec.startswith("X:"):
        return spec, None
    tr, cand = spec.split(":", 1)
    cand, _, sc = cand.partition("@")
    if cand not in D.bases or (tr not in RAW and tr not in PEER):
        return spec, None
    v = D.bases[cand]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = peer(v, D, tr) if tr in PEER else v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
    return spec, v * float(sc) if sc else v


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--target", default="")
    ap.add_argument("--cands", default="", help="comma list of tr:base@scale, or X:<col> for the matrix column")
    ap.add_argument("--trusted", default=TRUSTED)
    ap.add_argument("--calibrate", action="store_true", help="hide each trusted column; rank its true values among --cands + all bases (id)")
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1,s2,s3,s4")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--ced-dc", default="/work/crucible_data/cache/ic_build/*/date={date}.*/dc.parquet")
    ap.add_argument("--max-bases", type=int, default=400)
    ap.add_argument("--rows", type=int, default=200000, help="row sample (the node shares are row-count based, a sample is enough)")
    ap.add_argument("--max-nodes", type=int, default=4000, help="keep the nodes with the most PM training rows")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pool", action="store_true", help="add every base x --pool-tr x --pool-scales that fits the column's training range")
    ap.add_argument("--pool-tr", default="id,-id,dm,-dm,dmi,-dmi,neu,-neu,abs4,srank")
    ap.add_argument("--pool-scales", default="1,1e4,100,0.01,1e-4")
    ap.add_argument("--pool-in", type=float, default=0.98)
    ap.add_argument("--pool-re", default="", help="only bases matching this regex go into the pool")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    d = a.day
    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    D = Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
            [Path(e.format(date=d, mmdd=d[4:])) for e in a.extra.split(",") if e])
    D.ind = D.E = None
    if any(c.split(":")[0] in PEER for c in a.cands.split(",") if c) or (a.pool and PEER & set(a.pool_tr.split(","))):
        ced_rows(D, a.ced_dc, d)
    files = P.model_files(a.models, d, a.fits.split(","), a.subs.split(","))
    trusted = set(range(len(names))) if a.trusted == "all" else {ix[t] for t in a.trusted.split(",") if t in ix}
    seg = [infos(f) for f in files]
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(D.X.shape[0], min(a.rows, D.X.shape[0]), replace=False))
    print(f"{d}: {D.X.shape[0]:,} rows, {len(files)} model files, trusted {'ALL columns' if a.trusted == 'all' else [names[j] for j in sorted(trusted)]}")
    runs = []
    if a.calibrate:
        for j in sorted(trusted):
            pool = [f"X:{names[j]}"] + [f"id:{b}" for b in sorted(D.bases)[: a.max_bases]] + [c for c in a.cands.split(",") if c]
            runs.append((j, trusted - {j}, pool))
    for tg in [t for t in a.target.split(",") if t in ix]:
        runs.append((ix[tg], trusted - {ix[tg]}, [f"X:{tg}"] + [c for c in a.cands.split(",") if c]))
    for k, tr_set, pool in runs:
        nd = nodes(files, tr_set, k)
        roots = sum(1 for p, *_ in nd if not p)
        print(f"-- {names[k]}: {len(nd)} usable nodes ({roots} roots, {len(nd) - roots} conditional), PM rows {sum(c for *_, c in nd):,}")
        global _PREP
        _PREP = Prep(nd, D.X, k, rows, a.max_nodes)
        print(f"   {len(_PREP.items)} nodes kept on {rows.size:,} sampled rows")
        work = []
        for spec in pool:
            if spec.startswith("X:"):
                v = D.X[:, k].copy()
            else:
                spec, v = cand_values(D, spec)
                if v is None:
                    continue
            if np.isfinite(v).mean() >= 0.5:
                work.append((spec, v))
        if a.pool:
            import re
            rg = [sg[k] for sg in seg if sg[k]]
            if rg:
                lo, hi = min(r[0] for r in rg), max(r[1] for r in rg)
                tol = 1e-9 * max(abs(lo), abs(hi), 1e-12)
                n0 = len(work)
                for b in sorted(D.bases):
                    if a.pool_re and not re.search(a.pool_re, b):
                        continue
                    for tr in a.pool_tr.split(","):
                        _, v0 = cand_values(D, f"{tr}:{b}")
                        if v0 is None:
                            continue
                        f0 = v0[rows]
                        f0 = f0[np.isfinite(f0)]
                        if f0.size < 0.5 * rows.size:
                            continue
                        q0, q1 = np.quantile(f0, [0.001, 0.999])
                        for sc in a.pool_scales.split(","):
                            c_ = float(sc)
                            a0, a1 = sorted((q0 * c_, q1 * c_))
                            inside = float(((f0 * c_ >= lo - tol) & (f0 * c_ <= hi + tol)).mean())
                            if inside >= a.pool_in and hi > lo and (min(a1, hi) - max(a0, lo)) / (hi - lo) >= 0.2:
                                work.append((f"{tr}:{b}@{sc}", v0 * c_))
                print(f"   pool: {len(work) - n0} candidates inside [{lo:.5g}, {hi:.5g}]")
        if a.workers > 1 and sys.platform != "win32":
            import multiprocessing as mp
            with mp.get_context("fork").Pool(a.workers) as pool_:
                res = pool_.map(_one, work, chunksize=4)
        else:
            res = [_one(w) for w in work]
        res.sort(key=lambda r: (np.inf if not np.isfinite(r[0]) else r[0]))
        true_rank = next((i for i, r in enumerate(res) if r[1].startswith("X:")), -1)
        print(f"   rank of the matrix column among {len(res)}: {true_rank + 1}")
        for r in res[: a.top]:
            print(f"   cond {r[4]:.4f}  root {r[3]:.4f}  all {r[2]:.4f}  nodes {r[5]} ({r[6]} cond)  {r[1]}")


if __name__ == "__main__":
    main()
