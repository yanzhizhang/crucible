r"""Do PM's model generations mean the same thing by each column? Names, training ranges and split thresholds per month.

Every tk5m month folder has ``lgb/<fit>/model_s1..s4.txt``. Per month (one fit, all four sub-models pooled) and column:
the name (``feature_names``), the training range (``feature_infos``), how often the trees split on it and the 10 / 50 / 90 %
points of its split thresholds. Each month is compared with a reference month (default the latest): a column whose name,
range or thresholds moved a lot is likely a different quantity (or a different scale) in that generation, so a formula
matched on November days need not hold on an April day. Written to ``--out`` (one row per month x column), printed: per
month the columns that moved, and the reference's names next to them.

    python research/pm_factors/gen_compare.py --models /data/prod/models/tk5m --fit livev1_fit1 --out data/reports/gen_compare.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import polars as pl


def read(f: Path) -> tuple[list[str], list[tuple[float, float]], dict[int, list[float]]]:
    """(names, per-column [lo, hi], thresholds per column index) from one LightGBM text model (numerical splits only)."""
    names: list[str] = []
    ranges: list[tuple[float, float]] = []
    thr: dict[int, list[float]] = {}
    feats: list[int] = []
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("feature_names="):
                names = line.split("=", 1)[1].split()
            elif line.startswith("feature_infos="):
                for tok in line.split("=", 1)[1].split():
                    m = re.fullmatch(r"\[(.+):(.+)\]", tok)
                    ranges.append((float(m[1]), float(m[2])) if m else (np.nan, np.nan))
            elif line.startswith("split_feature="):
                feats = [int(x) for x in line.split("=", 1)[1].split()]
            elif line.startswith("threshold=") and feats:
                for j, t in zip(feats, line.split("=", 1)[1].split(), strict=False):
                    thr.setdefault(j, []).append(float(t))
                feats = []
    return names, ranges, thr


def month_table(folder: Path, fit: str) -> pl.DataFrame | None:
    """One row per column of one month: name, range, split count, threshold quantiles (sub-models pooled)."""
    files = sorted((folder / "lgb" / fit).glob("model_s*.txt"))
    if not files:
        return None
    names, ranges, pooled = None, None, {}
    for f in files:
        n, r, t = read(f)
        names, ranges = names or n, ranges or r
        for j, v in t.items():
            pooled.setdefault(j, []).extend(v)
    rows = []
    for j, nm in enumerate(names or []):
        t = np.array(pooled.get(j, []))
        lo, hi = ranges[j] if ranges and j < len(ranges) else (np.nan, np.nan)
        q = np.quantile(t, [0.1, 0.5, 0.9]) if t.size else np.full(3, np.nan)
        rows.append({"month": folder.name, "col": j, "name": nm, "lo": lo, "hi": hi, "splits": int(t.size),
                     "t10": q[0], "t50": q[1], "t90": q[2]})
    return pl.DataFrame(rows)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--ref", default="", help="reference month folder (default: the latest)")
    ap.add_argument("--out", type=Path, default=Path("data/reports/gen_compare.csv"))
    ap.add_argument("--range-ratio", type=float, default=2.0, help="flag a column whose training span changed by more than this factor")
    ap.add_argument("--shift", type=float, default=0.25, help="flag a column whose median threshold moved more than this share of the span")
    a = ap.parse_args()
    folders = sorted(p for p in a.models.iterdir() if p.is_dir() and re.fullmatch(r"\d{6}(\.\w+)?", p.name))
    tabs = [t for p in folders if (t := month_table(p, a.fit)) is not None]
    if not tabs:
        raise SystemExit(f"no {a.fit} models under {a.models}")
    df = pl.concat(tabs)
    ref = a.ref or max(df["month"].unique().to_list())
    r = df.filter(pl.col("month") == ref).select("col", pl.col("name").alias("ref_name"), pl.col("lo").alias("ref_lo"),
                                                  pl.col("hi").alias("ref_hi"), pl.col("splits").alias("ref_splits"),
                                                  pl.col("t50").alias("ref_t50"))
    df = df.join(r, on="col", how="left").with_columns(
        ((pl.col("hi") - pl.col("lo")) / (pl.col("ref_hi") - pl.col("ref_lo"))).alias("span_ratio"),
        ((pl.col("t50") - pl.col("ref_t50")).abs() / (pl.col("ref_hi") - pl.col("ref_lo"))).alias("t50_shift"),
        (pl.col("name") != pl.col("ref_name")).alias("renamed")).with_columns(pl.col("span_ratio", "t50_shift").fill_nan(None))  # polars ranks NaN above all
    df = df.with_columns(((pl.col("span_ratio") > a.range_ratio) | (pl.col("span_ratio") < 1 / a.range_ratio)
                          | (pl.col("t50_shift") > a.shift) | pl.col("renamed")).fill_null(False).alias("moved"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out)
    print(f"{len(tabs)} months x {df['col'].n_unique()} columns, fit {a.fit}, reference {ref} -> {a.out}")
    for mo in sorted(df["month"].unique().to_list()):
        g = df.filter(pl.col("month") == mo)
        sr = g.filter(pl.col("splits") > 0)
        rho = sr.select(pl.corr(pl.col("splits").rank(), pl.col("ref_splits").rank())).item() if sr.height > 2 else np.nan
        mv = g.filter(pl.col("moved"))
        print(f"  {mo}: renamed {int(g['renamed'].sum())}, moved {mv.height}; split-count rank corr with {ref} {rho:+.3f}")
        if mo != ref and mv.height:
            for row in mv.sort("t50_shift", descending=True, nulls_last=True).head(25).iter_rows(named=True):
                print(f"      col {row['col']:3d} {row['ref_name']:<14} {('name ' + row['name'] + ' ') if row['renamed'] else ''}"
                      f"span x{row['span_ratio']:.2f} median split moved {row['t50_shift']:.2f} span; splits {row['splits']} vs {row['ref_splits']}")


if __name__ == "__main__":
    main()
