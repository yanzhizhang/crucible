r"""Model-file-only atlas of PM's 209 columns: four target-free studies that need no market data.

  ranges     E1  every column's feature_infos [min, max] per sub-model s1..s4 per generation; flags columns whose range is
                 the same in every segment (a per-day constant or a slow state) and columns whose max grows with the
                 segment (a clock or a cumulative quantity) -- which segment owns which hours shows up here.
  ladders    E2  families with numbered siblings (zzqf2..10, zzie2..5, zzyj..yj5, zzhi/hi2 ...): per leaf, range width and
                 threshold quantiles against the sibling index; a monotone ladder says what the index is (window, level).
  neighbours E3  over all trees, parent/child split pairs per open column against chance (lift); high lift to a settled
                 family hints at a shared source or a conditional pair.
  grammar    E4  over the settled formulas: leaf letter x transform prefix x base stem; does the letter predict the
                 transform (priors for every open leaf)?
Read-only; prints tables, writes <out>/<study>.csv.

    python research/featgen/model_atlas.py all --gens /work/crucible_data/models_gen --date 20261113 --out data/reports/atlas
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402
from col_probe import header, trees, walk  # noqa: E402
from invert_cols import BASELINE  # noqa: E402

SUBS = ["s1", "s2", "s3", "s4"]


def infos(f: Path) -> list[tuple[float, float] | None]:
    """feature_infos of one model file as (min, max) per column; None for 'none' / categorical."""
    out = []
    for s in header(f).get("feature_infos", "").split():
        m = re.fullmatch(r"\[(.+):(.+)\]", s)
        out.append((float(m.group(1)), float(m.group(2))) if m else None)
    return out


def settled_formulas(path: Path) -> dict[str, str]:
    """Column -> formula of the settled baseline."""
    out = {}
    for ln in path.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, _, rest = ln.partition(" ")
            out[k] = rest.strip()
    return out


def study_ranges(a, names: list[str], out: Path) -> None:
    """E1: per column x generation x segment, [min, max]; constant-across-segment and growing-max flags."""
    rows = []
    for g in a.gens_list:
        for s in SUBS:
            fs = P.model_files(a.gens / g, a.date, [a.fit], [s])
            if not fs:
                continue
            for i, r in enumerate(infos(fs[0])):
                if r:
                    rows.append({"gen": g, "seg": s, "i": i, "col": names[i], "lo": r[0], "hi": r[1]})
    t = pl.DataFrame(rows)
    t.write_csv(out / "ranges.csv")
    print(f"== E1 ranges: {t.height} rows ({t['gen'].n_unique()} gens x {t['seg'].n_unique()} segs)")
    for g in a.gens_list:
        sub = t.filter(pl.col("gen") == g)
        if sub.is_empty():
            continue
        piv = sub.group_by("col").agg(pl.col("hi").min().alias("hi_min"), pl.col("hi").max().alias("hi_max"), pl.col("lo").min().alias("lo_min"),
                                      pl.col("lo").max().alias("lo_max"), pl.col("hi").sort_by("seg").alias("his"), pl.len().alias("n"))
        piv = piv.with_columns(((pl.col("hi_max") - pl.col("hi_min")) / (pl.col("hi_max").abs() + 1e-12)).alias("hi_spread"))
        const = piv.filter((pl.col("n") == len(SUBS)) & (pl.col("hi_spread") < 1e-9) & ((pl.col("lo_max") - pl.col("lo_min")).abs() < 1e-12))
        mono = piv.filter(pl.col("n") == len(SUBS)).filter(pl.col("his").list.eval(pl.element().diff().drop_nulls() > 0).list.all() & (pl.col("hi_spread") > 0.2))
        print(f"-- gen {g}: identical range in all {len(SUBS)} segments: {const.height} columns: {' '.join(sorted(const['col'].to_list()))[:900]}")
        print(f"-- gen {g}: max strictly growing s1->s4 by > 20%: {mono.height}")
        for r in mono.sort("hi_spread", descending=True).head(25).iter_rows(named=True):
            print(f"   {r['col']:<14} his {' '.join(f'{x:.5g}' for x in r['his'])}")
        wide = piv.sort("hi_spread", descending=True).head(15)
        print(f"-- gen {g}: largest segment-to-segment max spread:")
        for r in wide.iter_rows(named=True):
            print(f"   {r['col']:<14} his {' '.join(f'{x:.5g}' for x in r['his'])}  lo [{r['lo_min']:.5g}, {r['lo_max']:.5g}]")
    for c in a.watch.split(","):
        sub = t.filter(pl.col("col") == c).sort("gen", "seg")
        if not sub.is_empty():
            print(f"-- watch {c}: " + "  ".join(f"{r['gen']}/{r['seg']} [{r['lo']:.5g},{r['hi']:.5g}]" for r in sub.iter_rows(named=True)))


def all_thresholds(files: list[Path], n: int) -> list[np.ndarray]:
    """Numeric split thresholds per column over the files."""
    th: list[list[float]] = [[] for _ in range(n)]
    for f in files:
        for kv in trees(f):
            for j, t, dt in zip(kv["split_feature"], kv["threshold"], kv["decision_type"], strict=True):
                if not int(dt) & 1:
                    th[int(j)].append(float(t))
    return [np.unique(np.array(x)) for x in th]


def study_ladders(a, names: list[str], out: Path) -> None:
    """E2: numbered sibling families, per leaf: threshold count / quantiles / range against the index."""
    fs = P.model_files(a.gens / a.gen, a.date, [a.fit], SUBS)
    th = all_thresholds(fs, len(names))
    rng = infos(fs[0])
    groups: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for i, n in enumerate(names):
        m = re.fullmatch(r"(zz[a-z]+?)(\d*)((?:\.[^.]+)*)", n)
        if not m:
            continue
        base, num, leaf = m.group(1), m.group(2), m.group(3)
        groups[(base, leaf)].append((int(num) if num else 1, i))
    rows = []
    print("== E2 ladders: families with >= 3 numbered siblings sharing a leaf")
    for (base, leaf), mem in sorted(groups.items()):
        if len(mem) < 3:
            continue
        mem.sort()
        print(f"-- {base}#{leaf}")
        for k, i in mem:
            t = th[i]
            q = np.quantile(t, [0.1, 0.5, 0.9]) if t.size else [np.nan] * 3
            r = rng[i] or (np.nan, np.nan)
            rows.append({"family": base, "leaf": leaf, "k": k, "col": names[i], "n_thr": t.size, "q10": q[0], "q50": q[1], "q90": q[2], "lo": r[0], "hi": r[1]})
            print(f"   {k:>2} {names[i]:<14} thr {t.size:>4}  q10/50/90 {q[0]:>10.4g} {q[1]:>10.4g} {q[2]:>10.4g}  range [{r[0]:.4g}, {r[1]:.4g}]  width90 {q[2] - q[0]:.4g}")
    pl.DataFrame(rows).write_csv(out / "ladders.csv")


def study_neighbours(a, names: list[str], formulas: dict[str, str], out: Path) -> None:
    """E3: parent/child split pairs per open column, lift = n_ab * N / (n_a * n_b)."""
    fs = P.model_files(a.gens / a.gen, a.date, [a.fit], SUBS)
    pair: Counter = Counter()
    deg: Counter = Counter()
    for f in fs:
        for kv in trees(f):
            sf = [int(x) for x in kv["split_feature"]]
            _, parent = walk(kv)
            for nd, p in enumerate(parent):
                if p >= 0 and sf[p] != sf[nd]:
                    k = tuple(sorted((sf[p], sf[nd])))
                    pair[k] += 1
                    deg[sf[p]] += 1
                    deg[sf[nd]] += 1
    N = sum(pair.values())
    rows = []
    print(f"== E3 neighbours: {N} parent-child edges over {len(fs)} files; per open column, top lift (n >= {a.min_pairs})")
    for i, n in enumerate(names):
        nb = []
        for (x, y), c in pair.items():
            if i in (x, y) and c >= a.min_pairs:
                j = y if x == i else x
                lift = c * N / (deg[i] * deg[j]) * 2 if deg[i] and deg[j] else 0.0
                nb.append((lift, c, j))
                rows.append({"col": n, "nb": names[j], "n": c, "lift": lift, "col_settled": n in formulas, "nb_settled": names[j] in formulas})
        if n in formulas:
            continue
        nb.sort(reverse=True)
        print(f"   {n:<14} " + "  ".join(f"{names[j]}{'*' if names[j] in formulas else ''}:{l:.1f}({c})" for l, c, j in nb[:6]))
    pl.DataFrame(rows).write_csv(out / "neighbours.csv")
    print("   (* = settled; lift > 1 = more often adjacent than chance)")


def study_grammar(names: list[str], formulas: dict[str, str], out: Path) -> None:
    """E4: leaf letter x transform prefix x base stem over the settled formulas."""
    rows = []
    for n, f in formulas.items():
        leaf = n.split(".")[-1] if "." in n else "-"
        pref = re.findall(r"(cal|neu|dm|rank|srank|crank|z):", f)
        g = f.split(":")[-1]
        abs_ = g.lstrip("-").startswith("|")
        scale = "*1e4" if "*1e4" in g else ""
        neg = g.startswith("-") or g.startswith("|-")
        stem = re.sub(r"\d+(s|_?\d*)?$", "", re.sub(r"[|*\-]|1e4", "", g).replace("gen.", "").replace("x.", ""))
        rows.append({"col": n, "family": n.split(".")[0], "leaf": leaf, "prefix": "+".join(p for p in pref if p != "cal") or "none",
                     "cal": "cal" in pref, "abs": abs_, "scale": scale, "neg": neg, "stem": stem, "formula": f})
    t = pl.DataFrame(rows)
    t.write_csv(out / "grammar.csv")
    print(f"== E4 grammar: {t.height} settled formulas")
    for by in ("prefix", "abs", "neg"):
        ct = t.group_by("leaf", by).len().sort("leaf", by)
        print(f"-- leaf x {by}:")
        for (leaf,), sub in sorted(ct.group_by("leaf"), key=lambda kv: kv[0][0]):
            print(f"   {leaf:<4} " + "  ".join(f"{r[by]}:{r['len']}" for r in sub.iter_rows(named=True)))
    print("-- stems by leaf (most common):")
    for (leaf,), sub in sorted(t.group_by("leaf"), key=lambda kv: kv[0][0]):
        print(f"   {leaf:<4} " + " ".join(f"{s}:{c}" for s, c in Counter(sub['stem'].to_list()).most_common(6)))
    open_ = [n for n in names if n not in formulas]
    print(f"-- open leaves ({len(open_)} columns): " + " ".join(f"{k}:{v}" for k, v in Counter(n.split('.')[-1] if '.' in n else '-' for n in open_).most_common()))


def unit_class(lo: float, hi: float, t: np.ndarray) -> str:
    """Scale class of a column from its training range and thresholds alone."""
    span = hi - lo
    frac = np.abs(t - np.round(t * 2) / 2) if t.size else np.array([1.0])
    intlike = t.size >= 5 and float((frac < 1e-9).mean()) > 0.9
    inner = (float(np.quantile(t, 0.99)) - float(np.quantile(t, 0.01))) if t.size else span
    if intlike:
        return "count/code (integer steps)"
    if lo >= -1.0001 and hi <= 1.0001 and lo < -0.9 and hi > 0.9:
        return "rank/imbalance [-1,1]"
    if lo >= 0 and hi <= 1.0001 and hi > 0.9:
        return "share/rank [0,1]"
    if lo > 0.3 and hi < 3 and lo < 1 < hi:
        return "ratio around 1"
    if lo >= 0 and hi > 1000:
        return "large non-negative (seconds / counts / amounts)"
    if lo >= 0 and hi > 20:
        return "bp-scale or count, non-negative"
    if abs(lo + hi) < 0.35 * max(abs(lo), abs(hi)) and hi > 2:
        return "z-score-like, symmetric"
    if abs(lo) < 0.6 and abs(hi) < 0.6 and lo < 0 < hi:
        return "small signed (return / z/100 / imbalance)" + (", tight core" if inner < 0.05 * span else "")
    if lo >= 0 and hi <= 20:
        return "small non-negative (ratio / fraction)"
    return "other signed"


def study_units(a, names: list[str], formulas: dict[str, str], out: Path) -> None:
    """Per column: scale class from feature_infos + thresholds; per family, whether its columns share a class."""
    fs = P.model_files(a.gens / a.gen, a.date, [a.fit], SUBS)
    th = all_thresholds(fs, len(names))
    seg = [infos(f) for f in fs]
    rows = []
    for i, n in enumerate(names):
        rg = [s[i] for s in seg if s[i]]
        if not rg:
            continue
        lo, hi = min(r[0] for r in rg), max(r[1] for r in rg)
        t = th[i]
        q = np.quantile(t, [0.01, 0.5, 0.99]) if t.size else [np.nan] * 3
        rows.append({"col": n, "family": n.split(".")[0], "lo": lo, "hi": hi, "n_thr": t.size, "t01": q[0], "t50": q[1], "t99": q[2],
                     "core_share": (q[2] - q[0]) / (hi - lo) if hi > lo else np.nan, "cls": unit_class(lo, hi, t),
                     "status": "settled" if n in formulas else "open", "formula": formulas.get(n, "")})
    t = pl.DataFrame(rows)
    t.write_csv(out / "units.csv")
    print(f"== units: {t.height} columns; scale class from training range + thresholds (gen {a.gen})")
    for (c,), sub in sorted(t.group_by("cls"), key=lambda kv: -kv[1].height):
        print(f"-- {c}: {sub.height} columns ({sub.filter(pl.col('status') == 'open').height} open)")
        for r in sub.sort("col").iter_rows(named=True):
            print(f"   {r['col']:<14} {r['status']:<7} [{r['lo']:.5g}, {r['hi']:.5g}] thr {r['n_thr']:>4} q01/50/99 {r['t01']:.4g} {r['t50']:.4g} {r['t99']:.4g}"
                  f" core {r['core_share']:.3f}  {r['formula'][:48]}")
    print("-- families whose columns are not all one class (a family mixing scales = several different quantities):")
    for (fm,), sub in sorted(t.group_by("family"), key=lambda kv: kv[0][0]):
        if sub["cls"].n_unique() > 1:
            print(f"   {fm:<8} " + "  ".join(f"{r['col'].split('.', 1)[-1] if '.' in r['col'] else '-'}:{r['cls'].split(' ')[0]}" for r in sub.iter_rows(named=True)))


def study_famlinks(a, names: list[str], out: Path) -> None:
    """Family x family adjacency lift in the trees: which sourced families does each sourceless family split next to?"""
    fs = P.model_files(a.gens / a.gen, a.date, [a.fit], SUBS)
    fam = [re.sub(r"\d+$", "", n.split(".")[0]) for n in names]
    src: dict[str, str] = {}
    if a.audit.exists():
        for r in pl.read_csv(a.audit).iter_rows(named=True):
            f = re.sub(r"\d+$", "", r["family"])
            if r["source"] != "none":
                src[f] = r["source"]
    pair: Counter = Counter()
    deg: Counter = Counter()
    for f in fs:
        for kv in trees(f):
            sf = [int(x) for x in kv["split_feature"]]
            _, parent = walk(kv)
            for nd, p in enumerate(parent):
                if p >= 0 and sf[p] != sf[nd]:
                    x, y = sorted((fam[sf[p]], fam[sf[nd]]))
                    pair[(x, y)] += 1
                    deg[fam[sf[p]]] += 1
                    deg[fam[sf[nd]]] += 1
    N = sum(pair.values())
    rows = []
    fams = sorted(set(fam))
    print(f"== famlinks: {len(fams)} family stems, {N} parent-child edges; sourced families: {len(src)} ({'audit ' + str(a.audit) if src else 'no audit.csv: all shown as unsourced'})")
    for f in fams:
        nb = []
        for (x, y), c in pair.items():
            if f in (x, y) and c >= a.min_pairs:
                g = y if x == f else x
                if x == y:
                    g = f
                lift = c * (4 if x == y else 2) * N / (deg[f] * deg[g])  # a self edge adds 2 to the family's degree
                nb.append((lift, c, g))
                rows.append({"family": f, "partner": g, "n": c, "lift": lift, "family_source": src.get(f, "none"), "partner_source": src.get(g, "none")})
        nb.sort(reverse=True)
        if f in src:
            continue
        sourced = [(l, c, g) for l, c, g in nb if g in src and g != f][:5]
        own = next((l for l, c, g in nb if g == f), 0.0)
        print(f"   {f:<8} (no source) self {own:.1f} | sourced partners: " + "  ".join(f"{g}[{src[g][:3]}]:{l:.1f}({c})" for l, c, g in sourced)
              + " | top any: " + " ".join(f"{g}:{l:.1f}" for l, c, g in nb[:4] if g != f))
    pl.DataFrame(rows).write_csv(out / "famlinks.csv")
    print("   (lift > 1 = split next to each other more often than chance; [eng] / [hst] = the partner has .jz keys / an hstats profile)")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=["ranges", "ladders", "neighbours", "grammar", "units", "famlinks", "all"])
    ap.add_argument("--gens", type=Path, default=Path("/work/crucible_data/models_gen"))
    ap.add_argument("--gen-list", default="A,B,C")
    ap.add_argument("--gen", default="B", help="generation for ladders / neighbours")
    ap.add_argument("--date", default="20261113")
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/work/crucible_data/prod_models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--watch", default="zzop2.b,zzag2.e,zzag2.a,zzro2.a,zzrg,zzwa.g,zzqf.q.c3,zzqf7.q.c2,zzwa.i")
    ap.add_argument("--min-pairs", type=int, default=5)
    ap.add_argument("--audit", type=Path, default=Path("data/reports/audit/audit.csv"), help="audit_sources csv: which families have a PM source")
    ap.add_argument("--out", type=Path, default=Path("data/reports/atlas"))
    a = ap.parse_args()
    a.gens_list = [g for g in a.gen_list.split(",") if (a.gens / g).exists()]
    a.out.mkdir(parents=True, exist_ok=True)
    names = P.feature_names(a.feature_csv)
    formulas = settled_formulas(a.formulas)
    if a.study in ("ranges", "all"):
        study_ranges(a, names, a.out)
    if a.study in ("ladders", "all"):
        study_ladders(a, names, a.out)
    if a.study in ("neighbours", "all"):
        study_neighbours(a, names, formulas, a.out)
    if a.study in ("grammar", "all"):
        study_grammar(names, formulas, a.out)
    if a.study in ("units", "all"):
        study_units(a, names, formulas, a.out)
    if a.study in ("famlinks", "all"):
        study_famlinks(a, names, a.out)
    print(f"[done] {a.study}; csv in {a.out}")


if __name__ == "__main__":
    main()
