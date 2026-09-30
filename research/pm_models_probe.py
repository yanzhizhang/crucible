"""Stage 5: read the PM's LightGBM models without their data -- what they predict and from what.

Parses LightGBM's text model format directly (no lightgbm import, so any version, any
machine), for every ``models/<set>/<YYYYMM>/lgb/<fit>/model_s<k>.txt`` under ``--models``:

* per model: objective, trees, leaves, learning rate, bagging / feature fraction, lambdas,
  min_data_in_leaf, number of features, the feature-set hash;
* per feature: total split **gain** and split count, summed over the trees; the family is the
  name's prefix (``zzug.ab`` -> ``ZZUG``);
* against ``--features`` (``lgb_fn_v1_209_cte.csv``) and ``--hstats`` (the family directories):
  families the model uses that the sample does not ship, and the reverse.

What the tables answer: regression or classification (the target's shape); whether ``s1..s4``
differ in **features** (time-of-day segments / horizons) or only in **trees** (an ensemble of
seeds); whether ``fit1..3`` differ at all; how importance moves month to month.

Outputs (``--out``): ``models.parquet``, ``importance.parquet``, ``families.parquet``, and a
printed summary. Everything is derived statistics of the model files -- safe to bring back.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import polars as pl

_PARAMS = ("objective", "boosting", "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
           "min_sum_hessian_in_leaf", "bagging_fraction", "bagging_freq", "feature_fraction",
           "lambda_l1", "lambda_l2", "min_gain_to_split", "num_iterations", "early_stopping_round",
           "metric", "seed", "max_bin", "linear_tree", "monotone_constraints")


def parse_model(path: Path) -> tuple[dict, pl.DataFrame]:
    """One LightGBM text model -> (header/parameter dict, per-feature gain and split counts)."""
    text = path.read_text(errors="replace")
    head, _, rest = text.partition("\nTree=0")
    info: dict = {}
    for line in head.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    names = info.get("feature_names", "").split()
    params: dict[str, str] = {}
    m = re.search(r"\nparameters:\n(.*?)\nend of parameters", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            pm = re.match(r"\[(\w+): (.*)\]", line.strip())
            if pm:
                params[pm.group(1)] = pm.group(2)
    gain = [0.0] * len(names)
    splits = [0] * len(names)
    leaves = []
    shrink = []
    for block in re.split(r"\nTree=\d+\n", "\nTree=0" + rest)[1:]:
        kv = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "split_feature" in kv and kv["split_feature"].strip():
            for f, g in zip(kv["split_feature"].split(), kv["split_gain"].split(), strict=True):
                gain[int(f)] += float(g)
                splits[int(f)] += 1
        if "num_leaves" in kv:
            leaves.append(int(kv["num_leaves"]))
        if "shrinkage" in kv:
            shrink.append(float(kv["shrinkage"]))
    summary = {
        "objective": info.get("objective", params.get("objective", "")),
        "num_class": int(info.get("num_class", "1")),
        "trees": len(leaves),
        "leaves_mean": sum(leaves) / len(leaves) if leaves else None,
        "leaves_max": max(leaves) if leaves else None,
        "shrinkage": shrink[1] if len(shrink) > 1 else (shrink[0] if shrink else None),
        "n_features": len(names),
        "feature_hash": hashlib.sha1(" ".join(names).encode()).hexdigest()[:10],
        "features_used": sum(1 for s in splits if s),
        **{f"p_{k}": params.get(k) for k in _PARAMS},
    }
    imp = pl.DataFrame({"feature": names, "gain": gain, "splits": splits})
    return summary, imp


def family_of(feature: str) -> str:
    """``zzug.ab`` -> ``ZZUG``; names without a dot keep their leading letters."""
    stem = feature.split(".", 1)[0] if "." in feature else re.match(r"[A-Za-z]+", feature).group(0)
    return stem.upper()


def locate(path: Path, root: Path) -> dict[str, str]:
    """Model set / month / fit / sub-model from the path (whatever parts exist)."""
    rel = path.relative_to(root).parts
    month = next((p for p in rel if re.fullmatch(r"\d{6}", p)), "")
    fit = next((p for p in rel if p.startswith(("livev", "fit")) or "fit" in p), "")
    s = re.search(r"model_(s\d+)", path.name)
    return {"set": rel[0] if len(rel) > 1 else "", "month": month, "fit": fit,
            "sub": s.group(1) if s else path.stem, "file": "/".join(rel)}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=Path, default=Path("/work/prod/models"))
    ap.add_argument("--features", type=Path, help="lgb_fn_v1_209_cte.csv (default: <models>/lgb_fn_*.csv)")
    ap.add_argument("--hstats", type=Path, default=Path("/work/prod/hstats/sod/HS300"),
                    help="directory of the shipped family folders")
    ap.add_argument("--glob", default="**/model_*.txt")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    a = ap.parse_args()

    files = sorted(a.models.glob(a.glob))
    if not files:
        raise SystemExit(f"no models matching {a.glob} under {a.models}")
    rows, imps = [], []
    for f in files:
        summ, imp = parse_model(f)
        where = locate(f, a.models)
        rows.append({**where, **summ})
        imps.append(imp.with_columns(*[pl.lit(v).alias(k) for k, v in where.items()]))
    models = pl.DataFrame(rows)
    imp = pl.concat(imps).with_columns(
        pl.col("feature").map_elements(family_of, return_dtype=pl.String).alias("family"),
        (pl.col("gain") / pl.col("gain").sum().over("file")).alias("gain_share"))

    fams = imp.group_by("family").agg(
        pl.col("feature").n_unique().alias("features"), pl.col("gain_share").mean().alias("mean_gain_share"))
    csv = a.features or next(iter(sorted(a.models.glob("lgb_fn_*.csv"))), None)
    listed: set[str] = set()
    if csv and csv.exists():
        names = [ln.strip().split(",")[0] for ln in csv.read_text().splitlines() if ln.strip()]
        listed = {family_of(n) for n in names if "." in n}
        print(f"{csv.name}: {len(names)} names, {len(listed)} families")
    shipped = {p.name.upper() for p in a.hstats.iterdir() if p.is_dir()} if a.hstats.is_dir() else set()
    fams = fams.with_columns(pl.col("family").is_in(list(shipped)).alias("shipped"),
                             pl.col("family").is_in(list(listed)).alias("in_feature_csv"))

    a.out.mkdir(parents=True, exist_ok=True)
    models.write_parquet(a.out / "models.parquet")
    imp.write_parquet(a.out / "importance.parquet")
    fams.write_parquet(a.out / "families.parquet")

    with pl.Config(tbl_rows=60, tbl_width_chars=220, tbl_cols=20, float_precision=4):
        print(f"\n{len(files)} models, months {sorted(models['month'].unique().to_list())}")
        print("\nobjective / shape (distinct combinations):")
        print(models.group_by("objective", "num_class", "p_learning_rate", "p_num_leaves",
                              "p_feature_fraction", "p_bagging_fraction").agg(pl.len().alias("models"),
                              pl.col("trees").mean().alias("trees_mean")))
        print("\ndo s1..s4 differ in features (hash) or only in trees? -- per month / fit:")
        print(models.group_by("month", "fit").agg(
            pl.col("sub").n_unique().alias("subs"), pl.col("feature_hash").n_unique().alias("feature_sets"),
            pl.col("trees").min().alias("trees_min"), pl.col("trees").max().alias("trees_max"))
            .sort("month", "fit").head(24))
        print("\ndo fit1..3 share feature sets?")
        print(models.group_by("month", "sub").agg(pl.col("feature_hash").n_unique().alias("feature_sets"),
                                                   pl.col("p_seed").n_unique().alias("seeds"))
              .sort("month", "sub").head(24))
        print("\nfamilies by mean gain share (shipped = folder in the sample):")
        print(fams.sort("mean_gain_share", descending=True))
        print(f"\nused but NOT shipped: {sorted(set(fams['family']) - shipped) or 'none'}")
        print(f"shipped but NOT used: {sorted(shipped - set(fams['family'])) or 'none'}")
        top = (imp.group_by("feature", "family").agg(pl.col("gain_share").mean())
               .sort("gain_share", descending=True).head(25))
        print("\ntop 25 features by mean gain share:")
        print(top)
    print(f"\ntables: {a.out}/{{models,importance,families}}.parquet")


if __name__ == "__main__":
    main()
