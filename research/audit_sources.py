r"""209-row audit: for every model column, which PM source its family has and whether our formula actually uses it.

The user's point (2026-11-16): each PM factor (an engine state key in the .jz, a samplerS / hstats profile) should map to
model columns, and a column should be searched from its family's source -- yet of the 144 settled formulas only a handful
read PM state; the rest are gen.* stand-ins, even for families whose source we have. Columns of the table: family, PM
source kind (engine .jz key / hstats profile / both / none), our engine builds it and how exactly (verify csv), current
formula, does the formula read the source, status (matched / stand-in / open), gain. Summary: gain by status x source.
Read-only, writes <out>/audit.md.

    python research/audit_sources.py --jz /work/crucible_data/seed_probe/HS300/20261113.jz --verify data/reports/r125/verify_20261113.csv \
        --hstats /work/crucible_data/hstats_view/HS300 --models /work/crucible_data/models_gen/B --date 20261113 --out data/reports/audit
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


def fam(name: str) -> str:
    """zzwa.c -> zzwa; zzro2.a -> zzro2; zzgh.c.# -> zzgh."""
    return name.split(".")[0]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--verify", type=Path, default=None, help="engine/verify.py --csv of the same day")
    ap.add_argument("--hstats", type=Path, default=Path("/work/crucible_data/hstats_view/HS300"))
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=HERE / "engine" / "baseline_r41c.txt")
    ap.add_argument("--out", type=Path, default=Path("data/reports/audit"))
    a = ap.parse_args()
    names = feature_names(a.feature_csv)
    formulas = {}
    for ln in a.formulas.read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            k, _, rest = ln.partition(" ")
            formulas[k] = rest.strip()
    d = json.loads(gzip.open(a.jz).read())
    keys = sorted({re.sub(r"\.(sum|wgt|t)$", "", re.sub(r"\.\d+$", ".#", k)) for kv in d.values() for k in kv})
    jz_by_fam = defaultdict(list)
    for k in keys:
        jz_by_fam[fam(k)].append(k)
    hst = {p.name.lower(): p.name for p in a.hstats.iterdir() if p.is_dir()} if a.hstats.exists() else {}
    ver = {}
    if a.verify and a.verify.exists():
        for r in pl.read_csv(a.verify).iter_rows(named=True):
            ver[re.sub(r"\|.*$", "", r["key"])] = max(ver.get(re.sub(r"\|.*$", "", r["key"]), 0.0), float(r.get("all_1e4") or r.get("all") or 0))
    g = np.zeros(len(names))
    for f in model_files(a.models, a.date, ["livev1_fit1"], ["s1"]):
        g += gains(f, len(names))[0]
    g = g / g.sum() if g.sum() else g
    rows = []
    for i, n in enumerate(names):
        fm = fam(n)
        base = re.sub(r"\d+$", "", fm)  # zzgh2 -> zzgh: variants share a source
        jk = jz_by_fam.get(fm) or jz_by_fam.get(base) or []
        hs = hst.get(fm) or hst.get(base)
        kind = "+".join(x for x, ok in (("engine", bool(jk)), ("hstats", bool(hs))) if ok) or "none"
        built = [k for k in jk if k in ver]
        exact = min((ver[k] for k in built), default=None)
        f = formulas.get(n)
        uses = bool(f) and (any(k in f for k in jk) or (hs is not None and hs in f) or "ced." in f and bool(hs))
        status = "open" if f is None else ("matched" if uses else ("stand-in" if kind != "none" else "no-source"))
        rows.append({"i": i, "col": n, "family": fm, "source": kind, "jz_keys": len(jk), "built": f"{len(built)}/{len(jk)}" if jk else "-",
                     "built_min_1e4": exact, "hstats": hs or "", "formula": f or "", "uses_source": uses, "status": status, "gain": float(g[i])})
    t = pl.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    t.write_csv(a.out / "audit.csv")
    L = ["# Source audit: 209 model columns vs PM's factors", "", "## Gain by status x source", "", "| status | source | columns | gain |", "| --- | --- | --- | --- |"]
    for (st, src), sub in sorted(t.group_by("status", "source"), key=lambda kv: (kv[0][0], kv[0][1])):
        L.append(f"| {st} | {src} | {sub.height} | {sub['gain'].sum():.4f} |")
    L += ["", "## Families with a PM source, by gain", "", "| family | source | hstats | engine keys built | columns (settled / open / stand-in) | gain |", "| --- | --- | --- | --- | --- | --- |"]
    for (fm,), sub in sorted(t.filter(pl.col("source") != "none").group_by("family"), key=lambda kv: -kv[1]["gain"].sum()):
        r0 = sub.row(0, named=True)
        L.append(f"| {fm} | {r0['source']} | {r0['hstats']} | {r0['built']} | {sub.filter(pl.col('status') != 'open').height} / {sub.filter(pl.col('status') == 'open').height} / "
                 f"{sub.filter(pl.col('status') == 'stand-in').height} | {sub['gain'].sum():.4f} |")
    L += ["", "## All 209 columns", "", "| # | column | family | source | built | status | formula | gain |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in t.iter_rows(named=True):
        L.append(f"| {r['i']} | {r['col']} | {r['family']} | {r['source']} | {r['built']} | {r['status']} | `{r['formula']}` | {r['gain']:.4f} |")
    (a.out / "audit.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
