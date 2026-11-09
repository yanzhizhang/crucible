r"""Every PM tk5m model file's per-column metadata into one csv: training range (``feature_infos`` [min:max]) and split
count (``feature_importances``), so the 209 x 36 values live in a file instead of a transcription.

Rows: month, fit, model (s1..s4), col (0..208), lo, hi, splits (0 when the column is never split on). ``none`` in
feature_infos (a column with no usable values) gives NaN. Printed: the row count and, per model, how many columns have a
range, the 5 most-split columns, and how far the ranges of fit1 / fit2 / fit3 differ.

    python research/pm_factors/pm_lgb_meta.py --models /data/prod/models/tk5m --out data/reports/pm_lgb_meta.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl

PATH = re.compile(r"(?P<month>[^/\\]+)[/\\]lgb[/\\](?P<fit>[^/\\]+)[/\\]model_(?P<model>s\d+)\.txt$")


def read(f: Path) -> tuple[list[tuple[float, float]], dict[int, int]]:
    """(per-column [lo, hi], split count per column index) from one LightGBM text model."""
    ranges: list[tuple[float, float]] = []
    imp: dict[int, int] = {}
    in_imp = False
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("feature_infos="):
                for tok in line.split("=", 1)[1].split():
                    m = re.fullmatch(r"\[(.+):(.+)\]", tok)
                    ranges.append((float(m[1]), float(m[2])) if m else (np.nan, np.nan))
            elif line.startswith("feature_importances:"):
                in_imp = True
            elif in_imp:
                m = re.fullmatch(r"Column_(\d+)=(\d+)\s*", line)
                if not m:
                    in_imp = False
                    continue
                imp[int(m[1])] = int(m[2])
    return ranges, imp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/pm_lgb_meta.csv"))
    a = ap.parse_args()
    rows = []
    for f in sorted(a.models.glob("*/lgb/*/model_s*.txt")):
        m = PATH.search(str(f))
        if not m:
            continue
        ranges, imp = read(f)
        for c, (lo, hi) in enumerate(ranges):
            rows.append((m["month"], m["fit"], m["model"], c, lo, hi, imp.get(c, 0)))
    if not rows:
        print(f"no model files under {a.models}")
        return
    df = pl.DataFrame(rows, schema={"month": pl.String, "fit": pl.String, "model": pl.String, "col": pl.Int64, "lo": pl.Float64,
                                    "hi": pl.Float64, "splits": pl.Int64}, orient="row")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out)
    print(f"{df.height:,} rows, {df.select('month', 'fit', 'model').n_unique()} files -> {a.out}")
    for (mo, s), g in df.group_by("month", "model", maintain_order=True):
        fits = sorted(g["fit"].unique().to_list())
        top = g.group_by("col").agg(pl.col("splits").sum()).sort("splits", descending=True).head(5)
        w = g.pivot(on="fit", index="col", values="hi")
        span = (g.group_by("col").agg((pl.col("hi") - pl.col("lo")).max().alias("s"))).join(w, on="col")
        dif = {f2: float(np.nanmax(np.abs(span[fits[0]] - span[f2]) / span["s"].clip(1e-12))) for f2 in fits[1:] if f2 in span.columns}
        print(f"  {mo} {s}: {g.filter(pl.col('fit') == fits[0])['lo'].is_finite().sum()} columns with a range; most split "
              + ", ".join(f"{c}:{n}" for c, n in top.iter_rows()) + "; max |hi - hi(" + fits[0] + ")| / span: "
              + ", ".join(f"{k} {v:.3g}" for k, v in dif.items()))


if __name__ == "__main__":
    main()
