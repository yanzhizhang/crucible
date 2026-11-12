r"""What did PM's LightGBM models learn to predict? Read-only: the model text files and PM's recorded alpha only.

Per model file (``<models>/<month>/lgb/<fit>/model_s<k>.txt``): objective and its parameters, trees, shrinkage, the first
tree's count-weighted mean leaf (with boost_from_average the training-label mean is folded into tree 0), per-tree
|leaf| quantiles, and the prediction scale implied by the trees: mean = sum of each tree's count-weighted leaf mean,
std ~ sqrt(sum of each tree's count-weighted leaf variance) (trees taken as independent -- a rough scale, not exact).
Then the same per month x fit across s1..s4 (does the scale grow with s, i.e. horizons, or stay flat, i.e. time-of-day
segments?) and per generation (month groups by identical file content).

``--alpha``: PM's recorded predictions for one day -- overall quantiles, the cross-sectional std per half hour, so the
model scale can be compared with what PM actually emitted (unit check).

    python research/pm_factors/lgb_label_probe.py --models /data/prod/models/tk5m \
        --alpha /work/crucible_data/pm_livev1/rec10_p/20260430/alpha.parquet --out data/reports/lgb_label
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import polars as pl

KEEP = ("objective", "boosting", "learning_rate", "num_iterations", "boost_from_average", "metric", "label_column", "weight_column",
        "huber_delta", "fair_c", "alpha", "tweedie_variance_power", "num_class", "max_bin", "linear_tree")


def parse(path: Path) -> dict:
    text = path.read_text(errors="replace")
    head, _, rest = text.partition("\nTree=0")
    info = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
    params = dict(re.findall(r"\[(\w+): ([^\]]*)\]", text.partition("parameters:")[2]))
    means, vars_, absq, shr = [], [], [], []
    t0 = None
    for i, block in enumerate(re.split(r"\nTree=\d+\n", "\nTree=0" + rest)[1:]):
        kv = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "leaf_value" not in kv:
            continue
        lv = np.array([float(x) for x in kv["leaf_value"].split()])
        lc = np.array([float(x) for x in kv.get("leaf_count", " ".join(["1"] * lv.size)).split()])
        w = lc / lc.sum() if lc.sum() > 0 else np.full(lv.size, 1 / lv.size)
        m = float((w * lv).sum())
        means.append(m)
        vars_.append(float((w * (lv - m) ** 2).sum()))
        absq.append(np.abs(lv))
        shr.append(float(kv.get("shrinkage", "nan")))
        if i == 0:
            t0 = m
    a = np.concatenate(absq) if absq else np.array([np.nan])
    return {"objective": info.get("objective", params.get("objective", "")), **{f"p_{k}": params.get(k) for k in KEEP},
            "trees": len(means), "shrinkage": shr[1] if len(shr) > 1 else (shr[0] if shr else None), "tree0_mean": t0,
            "rest_mean": float(np.sum(means[1:])) if len(means) > 1 else 0.0, "pred_mean": float(np.sum(means)),
            "pred_std_indep": float(np.sqrt(np.sum(vars_))), "leaf_abs_p1": float(np.quantile(a, 0.01)),
            "leaf_abs_p50": float(np.quantile(a, 0.5)), "leaf_abs_p99": float(np.quantile(a, 0.99)),
            "md5": hashlib.md5(path.read_bytes()).hexdigest()[:8]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--alpha", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/reports/lgb_label"))
    a = ap.parse_args()
    rows = []
    for f in sorted(a.models.glob("*/lgb/*/model_s*.txt")):
        month, fit, s = f.parts[-4], f.parts[-2], f.stem.replace("model_", "")
        try:
            rows.append({"month": month, "fit": fit, "s": s, **parse(f)})
        except Exception as e:  # noqa: BLE001
            print(f"  {f}: {e}")
    df = pl.DataFrame(rows, infer_schema_length=None)
    a.out.mkdir(parents=True, exist_ok=True)
    df.write_csv(a.out / "models.csv")
    gen = df.group_by("month").agg(pl.col("md5").sort().str.join(",").alias("files")).with_columns(
        pl.col("files").rank("dense").alias("gen_id")).select("month", "gen_id")
    df = df.join(gen, on="month")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=230, float_precision=6, fmt_str_lengths=40):
        p = [c for c in df.columns if c.startswith("p_") and df[c].drop_nulls().n_unique() > 0]
        print("== objective / parameters (distinct values over all files)")
        print(df.select(["objective", *p]).unique())
        print("\n== per file (fit1 only; all fits in models.csv)")
        print(df.filter(pl.col("fit").str.ends_with("fit1")).select("month", "gen_id", "s", "trees", "shrinkage", "tree0_mean", "rest_mean",
                                                                     "pred_mean", "pred_std_indep", "leaf_abs_p50", "leaf_abs_p99", "md5").sort("month", "s"))
        print("\n== scale by s within each generation (median over its months and fits)")
        print(df.group_by("gen_id", "s").agg(pl.col("month").min().alias("first"), pl.col("month").max().alias("last"),
                                             pl.col("tree0_mean", "pred_mean", "pred_std_indep", "leaf_abs_p99").median()).sort("gen_id", "s"))
        print("\n== fits within a month: same scale? (std of pred_std_indep across fits, per month x s)")
        print(df.group_by("month", "s").agg(pl.col("pred_std_indep").std().alias("std_across_fits"), pl.col("pred_std_indep").mean().alias("mean")).sort("month", "s").head(40))
    if a.alpha and a.alpha.exists():
        al = pl.read_parquet(a.alpha)
        x = al["alpha"].to_numpy()
        x = x[np.isfinite(x)]
        print(f"\n== PM alpha {a.alpha}: {al.height:,} rows, columns {al.columns}")
        print("  quantiles p1/p10/p50/p90/p99: " + " / ".join(f"{q:.6g}" for q in np.quantile(x, [0.01, 0.1, 0.5, 0.9, 0.99])) + f"; mean {x.mean():.6g} std {x.std():.6g}")
        t = "rec_epoch" if "rec_epoch" in al.columns else "src_epoch"
        al = al.with_columns((((pl.col(t) + 28800) % 86400) // 1800 * 30).alias("half_hour_from_0000_min"))
        print("  cross-sectional std of alpha per half hour (local minute of day), median over the snapshots in it:")
        cs = al.with_columns(pl.col(t).round(0).alias("sec")).group_by("half_hour_from_0000_min", "sec").agg(pl.col("alpha").std().alias("xs"), pl.len().alias("n")).filter(
            pl.col("n") >= 50).group_by("half_hour_from_0000_min").agg(pl.col("xs").median(), pl.col("n").median().alias("stocks"), pl.len().alias("seconds")).sort("half_hour_from_0000_min")
        with pl.Config(tbl_rows=-1, float_precision=6):
            print(cs)
    print(f"\n-> {a.out}/models.csv")


if __name__ == "__main__":
    main()
