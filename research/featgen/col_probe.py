r"""Everything PM's trees say about one model column (read-only): training range, thresholds, depth, training CDF, neighbours.

For a column without a formula (zzeg: the open column with the most split gain), per model file of the day's generation:
``feature_infos`` range, split count and gain share, the distinct thresholds (how many, range, how many are integers or
halves -- a dummy / rank / count has a grid), missing handling (decision_type bits), the depth of each split (near the
root = a regime variable), the gain by depth, the root-split training CDF points P(x <= t), and the columns split on
just above and below it. With ``--implied``, the R96 stage-1 implied values of the column: how much of their variance is
between stocks rather than within a stock's day, and whether a stock's mean agrees between two days (a per-stock-day
constant shows both).

    python research/featgen/col_probe.py --col zzeg --date 20261112 --models $DATA/models_gen/B \
        --implied "data/invert_r96_1112/implied_{date}.parquet" --days 20261112,20260430
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402


def trees(path: Path) -> list[dict[str, list[str]]]:
    """The tree blocks of one LightGBM text model, key -> whitespace-split values."""
    text = path.read_text(errors="replace")
    out = []
    for block in re.split(r"\nTree=\d+\n", text)[1:]:
        kv = {k: v.split() for k, v in (ln.split("=", 1) for ln in block.splitlines() if "=" in ln)}
        if kv.get("split_feature"):
            out.append(kv)
    return out


def header(path: Path) -> dict[str, str]:
    """The model's header lines (feature_names, feature_infos ...)."""
    head = path.read_text(errors="replace").split("\nTree=0", 1)[0]
    return dict(ln.split("=", 1) for ln in head.splitlines() if "=" in ln)


def walk(kv: dict[str, list[str]]) -> tuple[list[int], list[int]]:
    """Depth and parent node of every internal node (root depth 0, parent -1)."""
    lc, rc = [int(x) for x in kv["left_child"]], [int(x) for x in kv["right_child"]]
    depth, parent = [0] * len(lc), [-1] * len(lc)
    stack = [0]
    while stack:
        n = stack.pop()
        for c in (lc[n], rc[n]):
            if c >= 0:
                depth[c], parent[c] = depth[n] + 1, n
                stack.append(c)
    return depth, parent


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--col", default="zzeg")
    ap.add_argument("--date", required=True, help="picks the model generation (latest month <= date)")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--implied", default="", help="invert_cols stage-1 output, {date} filled per --days")
    ap.add_argument("--days", default="")
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    j = names.index(a.col)
    files = P.model_files(a.models, a.date, a.fits.split(","), a.subs.split(","))
    print(f"{a.col} = column {j} of {len(names)}; model files {[str(f) for f in files]}")
    for f in files:
        h = header(f)
        infos = h.get("feature_infos", "").split()
        fn = h.get("feature_names", "").split()
        print(f"\n== {f.name}: feature_names[{j}]={fn[j] if j < len(fn) else '?'}  training range feature_infos={infos[j] if j < len(infos) else '?'}")
        gain_all, gain_col = 0.0, 0.0
        th, dts, depths, gd = [], Counter(), Counter(), Counter()
        up, down, roots, cdf = Counter(), Counter(), 0, []
        n_trees = 0
        for kv in trees(f):
            n_trees += 1
            sf = [int(x) for x in kv["split_feature"]]
            sg = [float(x) for x in kv.get("split_gain", ["0"] * len(sf))]
            gain_all += sum(sg)
            depth, parent = walk(kv)
            lc, rc = [int(x) for x in kv["left_child"]], [int(x) for x in kv["right_child"]]
            ic = [float(x) for x in kv.get("internal_count", [])]
            lfc = [float(x) for x in kv.get("leaf_count", [])]
            for n, (feat, g) in enumerate(zip(sf, sg, strict=True)):
                if feat != j:
                    continue
                gain_col += g
                dt = int(kv["decision_type"][n])
                dts[(dt & 1, (dt >> 1) & 1, (dt >> 2) & 3)] += 1
                if not dt & 1:
                    th.append(float(kv["threshold"][n]))
                depths[depth[n]] += 1
                gd[min(depth[n], 6)] += g
                if parent[n] >= 0:
                    up[names[sf[parent[n]]]] += 1
                for c in (lc[n], rc[n]):
                    if c >= 0:
                        down[names[sf[c]]] += 1
                if n == 0 and ic and not dt & 1:
                    roots += 1
                    nl = ic[lc[0]] if lc[0] >= 0 else lfc[~lc[0]]
                    nr = ic[rc[0]] if rc[0] >= 0 else lfc[~rc[0]]
                    cdf.append((float(kv["threshold"][0]), nl / (nl + nr), (dt >> 1) & 1, (dt >> 2) & 3))
        t = np.unique(np.array(th))
        print(f"  trees {n_trees}; {sum(depths.values())} splits, gain share {gain_col / gain_all if gain_all else float('nan'):.4f}; root splits {roots}")
        print("  decision_type (categorical, default_left, missing 0 none/1 zero/2 nan): " + ", ".join(f"{k}x{v}" for k, v in dts.most_common()))
        if t.size:
            frac = t - np.floor(t)
            print(f"  {t.size} distinct thresholds in [{t.min():.6g}, {t.max():.6g}]; integer-ish {np.mean(np.isclose(frac, 0) | np.isclose(frac, 1)):.2f}, "
                  f"half-ish {np.mean(np.isclose(frac, 0.5)):.2f}; quantiles " + " ".join(f"{q:.4g}" for q in np.quantile(t, [0, .05, .25, .5, .75, .95, 1])))
            gaps = np.diff(t)
            print(f"  gaps: min {gaps.min() if gaps.size else 0:.3g} median {np.median(gaps) if gaps.size else 0:.3g}; first 12 {np.round(t[:12], 6).tolist()}; last 6 {np.round(t[-6:], 6).tolist()}")
        print("  splits by depth: " + " ".join(f"d{k}:{v}" for k, v in sorted(depths.items())))
        print("  gain by depth (6 = 6+): " + " ".join(f"d{k}:{v / gain_col:.2f}" for k, v in sorted(gd.items())) if gain_col else "")
        print("  parent columns: " + ", ".join(f"{k} {v}" for k, v in up.most_common(10)))
        print("  child columns:  " + ", ".join(f"{k} {v}" for k, v in down.most_common(10)))
        if cdf:
            cdf.sort()
            print("  root CDF P(x<=t) [default_left, missing]: " + "  ".join(f"{t_:.5g}:{p:.4f}[{dl},{mt}]" for t_, p, dl, mt in cdf[:40]))
    if a.implied and a.days:
        means = {}
        for d in a.days.split(","):
            p = Path(a.implied.format(date=d))
            if not p.exists():
                print(f"\n{p}: missing")
                continue
            df = pl.read_parquet(p, columns=["sid", "mark", a.col]).filter(pl.col(a.col).is_finite())
            tot = df[a.col].var()
            g = df.group_by("sid").agg(pl.col(a.col).mean().alias("m"), pl.col(a.col).var().alias("v"), pl.len().alias("n"))
            within = float((g["v"].fill_null(0) * g["n"]).sum() / g["n"].sum())
            means[d] = g.select("sid", "m")
            print(f"\nimplied {a.col} {d}: {df.height:,} rows, {g.height} stocks; total var {tot:.4g}, within-stock {within:.4g} "
                  f"(between-stock share {1 - within / tot if tot else float('nan'):.3f}); quantiles "
                  + " ".join(f"{df[a.col].quantile(x):.4g}" for x in (0, .1, .5, .9, 1)))
        if len(means) == 2:
            (d1, m1), (d2, m2) = means.items()
            jm = m1.join(m2, on="sid", suffix="_2")
            r = jm.select(pl.corr("m", "m_2", method="spearman")).item()
            print(f"per-stock mean of implied {a.col}, {d1} vs {d2}: spearman {r:.3f} over {jm.height} stocks")


if __name__ == "__main__":
    main()
