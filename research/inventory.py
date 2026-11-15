r"""The full inventory for a bit-exact PM copy: every state key in PM's .jz and every one of the 209 model features.

Table 1, PM's day-end state: every key of the .jz, numbered keys collapsed (``zzfb.c.17`` -> ``zzfb.c.#``), with how many
there are per stock, whether the value is the same for every stock (a market-level key), its range, and -- from an engine
``verify.py --csv`` table of the same day -- whether our engine builds it and on what share of stocks (all / SZSE) it equals
PM's. Table 2, the 209 features in model order: family, split-gain share, the settled formula (``--formulas``) or OPEN,
the threshold count / range / root splits / missing handling of PM's trees, and the .jz key patterns of the same family.
Table 3, per family: features settled / open, gain open, state keys built / missing. Read-only; writes <out>/inventory.md.

    python research/inventory.py --jz /work/crucible_data/seed_probe/HS300/20261113.jz --verify-csv data/reports/r109/verify_20261113.csv \
        --models /work/crucible_data/models_gen/B --date 20261113 --out data/reports/r109
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "featgen"))
from gain_share import gains  # noqa: E402
from pred_day import feature_names, model_files  # noqa: E402

BASELINE = HERE / "engine" / "baseline_r41c.txt"


def pattern(k: str) -> str:
    """A key with its trailing bin / grid number collapsed, and an EMA's .sum / .wgt / .t folded into the average."""
    k = re.sub(r"\.(sum|wgt)$", "", k)
    return re.sub(r"\.(\d+)$", ".#", k)


def family(name: str) -> str:
    """zzro2.a -> zzro2; zzgh.c.# -> zzgh; cdc.lp -> cdc."""
    return name.split(".")[0]


def tree_stats(files: list[Path], n: int) -> list[dict]:
    """Per feature: split count, distinct thresholds, their range, root splits, missing types seen."""
    th = [set() for _ in range(n)]
    roots, miss = np.zeros(n, int), [set() for _ in range(n)]
    for f in files:
        tree: dict[str, list[str]] = {}
        for ln in f.read_text().splitlines():
            if ln.startswith("Tree="):
                tree = {}
            for key in ("split_feature", "threshold", "decision_type"):
                if ln.startswith(key + "="):
                    tree[key] = ln[len(key) + 1:].split()
            if len(tree) == 3:
                for i, (j, t, dt) in enumerate(zip(tree["split_feature"], tree["threshold"], tree["decision_type"], strict=True)):
                    j, dt = int(j), int(dt)
                    miss[j].add(("none", "zero", "nan")[(dt >> 2) & 3])
                    if not dt & 1:
                        th[j].add(float(t))
                    if i == 0:
                        roots[j] += 1
                tree = {}
    out = []
    for j in range(n):
        t = np.array(sorted(th[j]))
        out.append({"thr": t.size, "lo": t[0] if t.size else None, "hi": t[-1] if t.size else None, "roots": int(roots[j]), "missing": "/".join(sorted(miss[j])) or "-"})
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--verify-csv", type=Path, default=None, help="engine/verify.py --csv of the same day")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--out", type=Path, default=Path("data/reports/inventory"))
    a = ap.parse_args()

    d = json.loads(gzip.open(a.jz).read())
    per: dict[str, list[float]] = defaultdict(list)
    nkeys: dict[str, set] = defaultdict(set)
    for kv in d.values():
        for k, v in kv.items():
            p = pattern(k)
            nkeys[p].add(k)
            if v is not None and not k.endswith(".t") and isinstance(v, (int, float)):
                per[p].append(float(v))
    ver = {}
    if a.verify_csv and a.verify_csv.exists():
        for r in pl.read_csv(a.verify_csv).iter_rows(named=True):
            ver[r["key"]] = (r.get("all"), r.get("SZSE"))
            ver.setdefault(pattern(r["key"]), (r.get("all"), r.get("SZSE")))
    rows1 = []
    for p in sorted(nkeys):
        v = np.array(per.get(p, []))
        f = v[np.isfinite(v)] if v.size else v
        same = "yes" if f.size and np.unique(f).size == 1 else ""
        base = re.sub(r"\.t$", "", p)
        m = ver.get(p) or ver.get(base) or ver.get(re.sub(r"\.#$", "", p))
        rows1.append((p, len(nkeys[p]) / max(len(d), 1), same, f.min() if f.size else None, float(np.median(f)) if f.size else None,
                      f.max() if f.size else None, m))

    names = feature_names(a.feature_csv)
    formulas = {}
    for ln in a.formulas.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, _, rest = ln.partition(" ")
            formulas[k] = rest.strip()
    files = model_files(a.models, a.date, a.fits.split(","), a.subs.split(","))
    g = np.zeros(len(names))
    for fl in files:
        g += gains(fl, len(names))[0]
    g = g / g.sum() if g.sum() else g
    ts = tree_stats(files, len(names))
    fam_keys = defaultdict(list)
    for p, *_ in rows1:
        fam_keys[family(p)].append(p)

    fmt = lambda x: "" if x is None else f"{x:.6g}"  # noqa: E731
    L = [f"# PM inventory: {a.jz.name}, model {', '.join(str(f) for f in files)}", "",
         f"{len(d)} stocks in the state file, {len(nkeys)} key patterns; {len(names)} model features, "
         f"{sum(n in formulas for n in names)} settled, {sum(n not in formulas for n in names)} open "
         f"(open gain {sum(g[i] for i, n in enumerate(names) if n not in formulas):.3f}).", "",
         "## 1. State keys (.jz)", "",
         "| key | per stock | same for all stocks | min | median | max | our engine: match all / SZSE |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for p, n, same, lo, med, hi, m in rows1:
        mm = "NOT BUILT" if m is None else f"{fmt(m[0])} / {fmt(m[1])}"
        L.append(f"| `{p}` | {n:.3g} | {same} | {fmt(lo)} | {fmt(med)} | {fmt(hi)} | {mm} |")
    L += ["", "## 2. Model features (model order)", "",
          "| # | feature | gain | formula | thresholds | range | root splits | missing | state keys of the family |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, n in enumerate(names):
        t = ts[i]
        fk = ", ".join(f"`{x}`" for x in fam_keys.get(family(n), [])[:8]) + (" ..." if len(fam_keys.get(family(n), [])) > 8 else "")
        f = f"`{formulas[n]}`" if n in formulas else "**OPEN**"
        L.append(f"| {i} | `{n}` | {g[i]:.4f} | {f} | {t['thr']} | {fmt(t['lo'])} .. {fmt(t['hi'])} | {t['roots']} | {t['missing']} | {fk} |")
    L += ["", "## 3. Families", "", "| family | features settled / open | open gain | state keys built / not built |", "| --- | --- | --- | --- |"]
    fams = sorted({family(n) for n in names} | set(fam_keys), key=lambda x: -sum(g[i] for i, n in enumerate(names) if family(n) == x))
    for fm in fams:
        idx = [i for i, n in enumerate(names) if family(n) == fm]
        st = sum(names[i] in formulas for i in idx)
        og = sum(g[i] for i in idx if names[i] not in formulas)
        ks = [r for r in rows1 if family(r[0]) == fm and not r[0].endswith(".t")]
        built = sum(r[6] is not None for r in ks)
        L.append(f"| `{fm}` | {st} / {len(idx) - st} | {og:.4f} | {built} / {len(ks) - built} |")
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "inventory.md").write_text("\n".join(L) + "\n")
    print(f"{len(rows1)} key patterns ({sum(r[6] is None for r in rows1 if not r[0].endswith('.t'))} not built), {len(names)} features "
          f"({sum(n not in formulas for n in names)} open) -> {a.out / 'inventory.md'}")


if __name__ == "__main__":
    main()
