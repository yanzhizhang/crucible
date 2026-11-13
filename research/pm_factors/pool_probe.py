r"""Are PM's sub-models s1..s4 trained on different stock pools (HS300 / ZZ500 / ZZ1000 ...) instead of day segments? Read-only.

1 rows each sub-model was trained on: the root ``internal_count`` of its trees (LightGBM text model). Pools of 300 / 500 /
  1000 stocks over the same days give row counts near 3 : 5 : 10; equal parts of one pool give equal counts.
2 ``--meta`` (pm_lgb_meta.csv): per pair of sub-models, how many columns have the same training range, and how many
  ranges of one contain the other's. One pool cut into parts gives mostly equal or mutually overlapping ranges; a wider
  pool contains a narrower one's ranges on most columns.

    python research/pm_factors/pool_probe.py --models /data/prod/models/tk5m --meta data/reports/pm_lgb_meta.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

PATH = re.compile(r"(?P<month>[^/\\]+)[/\\]lgb[/\\](?P<fit>[^/\\]+)[/\\]model_(?P<model>s\d+)\.txt$")
PARAM = re.compile(r"\[(\w+): (.*)\]\s*$")
KEYS = ("objective", "max_feature_idx", "version")
PARAMS = ("bagging_fraction", "bagging_freq", "feature_fraction", "num_leaves", "min_data_in_leaf", "learning_rate", "num_iterations", "max_bin", "seed", "data")


def read(f: Path) -> dict:
    """Tree count, root row counts and a few header / parameter lines of one LightGBM text model."""
    roots: list[int] = []
    leaves: list[int] = []
    out: dict = {k: "" for k in KEYS}
    out.update({"p_" + k: "" for k in PARAMS})
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("internal_count="):
                tok = line.split("=", 1)[1].split(None, 1)
                if tok:
                    roots.append(int(float(tok[0])))
            elif line.startswith("num_leaves="):
                leaves.append(int(line.split("=", 1)[1]))
            elif line.startswith("["):
                m = PARAM.match(line)
                if m and m[1] in PARAMS:
                    out["p_" + m[1]] = m[2][:60]
            else:
                for k in KEYS:
                    if line.startswith(k + "="):
                        out[k] = line.split("=", 1)[1].strip()[:40]
    r = np.array(roots, float)
    has = len(r) > 0
    out.update(trees=len(leaves), rows_tree0=int(r[0]) if has else 0, rows_min=int(r.min()) if has else 0,
               rows_med=int(np.median(r)) if has else 0, rows_max=int(r.max()) if has else 0,
               leaves_med=int(np.median(leaves)) if leaves else 0, mb=round(f.stat().st_size / 1e6, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--meta", type=Path, default=Path("data/reports/pm_lgb_meta.csv"))
    ap.add_argument("--months", default="", help="comma list of model month folders; default all")
    a = ap.parse_args()
    want = {x for x in a.months.split(",") if x}
    rows = []
    for f in sorted(a.models.glob("*/lgb/*/model_s*.txt")):
        m = PATH.search(str(f))
        if not m or (want and m["month"] not in want):
            continue
        rows.append({"month": m["month"], "fit": m["fit"], "model": m["model"], **read(f)})
    if not rows:
        print(f"no model files under {a.models}")
        return
    t = pl.DataFrame(rows, infer_schema_length=None)
    base = t.filter(pl.col("model") == "s1").select("month", "fit", pl.col("rows_max").alias("s1_rows"))
    t = t.join(base, on=["month", "fit"], how="left").with_columns((pl.col("rows_max") / pl.col("s1_rows")).round(3).alias("rows_vs_s1"))
    print(f"== 1 rows each sub-model was trained on (root internal_count; rows_max = all rows when bagging is on), {t.height} files")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=260, fmt_str_lengths=60):
        print(t.select("month", "fit", "model", "trees", "leaves_med", "rows_tree0", "rows_min", "rows_med", "rows_max", "rows_vs_s1", "mb"))
        pc = [c for c in t.columns if c.startswith("p_") or c in KEYS]
        print("  header / parameters, distinct value sets:")
        print(t.group_by(["model"] + pc).agg(pl.len().alias("files")).sort("model"))
        print("  rows_vs_s1 per sub-model over all months and fits:")
        print(t.group_by("model").agg(pl.col("rows_vs_s1").min().alias("min"), pl.col("rows_vs_s1").median().alias("median"),
                                      pl.col("rows_vs_s1").max().alias("max"), pl.col("rows_max").median().alias("rows_median")).sort("model"))
    if not a.meta.exists():
        print(f"no {a.meta}")
        return
    m = pl.read_csv(a.meta, schema_overrides={"month": pl.String})
    print(f"\n== 2 training ranges per pair of sub-models ({a.meta}); same = lo and hi equal to 1e-9 relative")
    out = []
    for (month, fit), g in m.group_by(["month", "fit"], maintain_order=True):
        if want and month not in want:
            continue
        subs = sorted(g["model"].unique().to_list())
        d = {s: g.filter(pl.col("model") == s).sort("col") for s in subs}
        for i, x in enumerate(subs):
            for y in subs[i + 1:]:
                j = d[x].join(d[y], on="col", suffix="_y")
                lo, hi, lo2, hi2 = (j[c].cast(pl.Float64).fill_null(float("nan")).to_numpy() for c in ("lo", "hi", "lo_y", "hi_y"))
                ok = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(lo2) & np.isfinite(hi2)
                lo, hi, lo2, hi2 = lo[ok], hi[ok], lo2[ok], hi2[ok]
                tol = 1e-9 * np.maximum(np.abs(hi - lo), 1e-300)
                same = (np.abs(lo - lo2) <= tol) & (np.abs(hi - hi2) <= tol)
                xin = (lo >= lo2 - tol) & (hi <= hi2 + tol) & ~same
                yin = (lo2 >= lo - tol) & (hi2 <= hi + tol) & ~same
                w1, w2 = hi - lo, hi2 - lo2
                good = (w1 > 0) & (w2 > 0)
                out.append({"month": month, "fit": fit, "pair": f"{x}-{y}", "cols": int(ok.sum()), "same": int(same.sum()),
                            "first_inside": int(xin.sum()), "second_inside": int(yin.sum()), "neither": int((~same & ~xin & ~yin).sum()),
                            "width_ratio_med": float(np.median(w1[good] / w2[good])) if good.any() else float("nan")})
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        o = pl.DataFrame(out)
        print(o)
        print("  per pair over all months and fits (medians):")
        print(o.group_by("pair").agg(pl.col("same", "first_inside", "second_inside", "neither", "width_ratio_med").median()).sort("pair"))


if __name__ == "__main__":
    main()
