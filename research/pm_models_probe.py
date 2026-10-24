"""Stage 5: read the PM's LightGBM models without their data -- what they predict and from what.

Parses LightGBM's text model format directly (no lightgbm import, so any version, any
machine), for every ``models/<set>/<YYYYMM>/lgb/<fit>/model_s<k>.txt`` under ``--models``:

* per model: objective, trees, leaves, learning rate, bagging / feature fraction, lambdas,
  min_data_in_leaf, number of features, the feature-set hash;
* per feature: total split **gain** and split count, summed over the trees; the family is the
  name's prefix (``zzug.ab`` -> ``ZZUG``);
* per feature, a **fingerprint** of what kind of quantity it is, usable even for families with
  no shipped output: its training range (``feature_infos`` ``[min:max]``) and the distribution of
  the split thresholds the trees chose on it (quantiles, how many distinct, share on half-integers =
  a count, share at ~0 = a zero / non-zero split), with a guess (const / flag / count / unit /
  signed / positive);
* against ``--features`` (``lgb_fn_v1_209_cte.csv``) and ``--hstats`` (the family directories):
  families the model uses that the sample does not ship, and the reverse.

What the tables answer: regression or classification (the target's shape); whether ``s1..s4``
differ in **features** (time-of-day segments / horizons) or only in **trees** (an ensemble of
seeds); whether ``fit1..3`` differ at all; how importance moves month to month.

Outputs (``--out``): ``models.parquet``, ``importance.parquet``, ``families.parquet``,
``fingerprint.parquet``, and a
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


def parse_model(path: Path) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    """One LightGBM text model -> (header/parameter dict, per-feature gain and splits, every split threshold)."""
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
    thr_f: list[int] = []
    thr_v: list[float] = []
    leaves = []
    shrink = []
    for block in re.split(r"\nTree=\d+\n", "\nTree=0" + rest)[1:]:
        kv = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "split_feature" in kv and kv["split_feature"].strip():
            for f, g in zip(kv["split_feature"].split(), kv["split_gain"].split(), strict=True):
                gain[int(f)] += float(g)
                splits[int(f)] += 1
            if "threshold" in kv:
                fs = [int(f) for f in kv["split_feature"].split()]
                thr_f.extend(fs)
                thr_v.extend(float(t) for t in kv["threshold"].split()[: len(fs)])
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
    ranges = info.get("feature_infos", "").split()
    bounds = [_range(r) for r in ranges] if len(ranges) == len(names) else [(None, None)] * len(names)
    imp = pl.DataFrame({"feature": names, "gain": gain, "splits": splits,
                        "lo": [b[0] for b in bounds], "hi": [b[1] for b in bounds]},
                       schema_overrides={"lo": pl.Float64, "hi": pl.Float64})
    thr = pl.DataFrame({"feature": [names[i] for i in thr_f], "t": thr_v}, schema={"feature": pl.String, "t": pl.Float64})
    return summary, imp, thr


def _range(r: str) -> tuple[float | None, float | None]:
    """``[a:b]`` -> (a, b); ``none`` (feature never usable) or a categorical list -> (None, None)."""
    m = re.fullmatch(r"\[(.*):(.*)\]", r)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def fingerprint(imp: pl.DataFrame, thr: pl.DataFrame) -> pl.DataFrame:
    """Per feature: training range, gain share, threshold quantiles and a guess at the kind of quantity."""
    rng = imp.group_by("feature", "family").agg(pl.col("lo").min(), pl.col("hi").max(), pl.col("gain_share").mean())
    two = pl.col("t") * 2
    half = ((two - two.round()).abs() < 1e-6) & (two.round() % 2 != 0)  # x.5 thresholds sit between integers
    t = thr.group_by("feature").agg(
        pl.len().alias("n_thr"), pl.col("t").n_unique().alias("distinct"),
        pl.col("t").quantile(0.1).alias("t_p10"), pl.col("t").median().alias("t_med"), pl.col("t").quantile(0.9).alias("t_p90"),
        half.mean().alias("half_int"), (pl.col("t").abs() < 1e-20).mean().alias("at_zero"))
    f = rng.join(t, on="feature", how="left")
    kind = (pl.when(pl.col("lo") == pl.col("hi")).then(pl.lit("const"))
            .when((pl.col("distinct") <= 2) & (pl.col("lo") >= 0) & (pl.col("hi") <= 1)).then(pl.lit("flag"))
            .when(pl.col("half_int") >= 0.9).then(pl.lit("count"))
            .when((pl.col("lo") >= 0) & (pl.col("hi") <= 1)).then(pl.lit("unit"))
            .when((pl.col("lo") < 0) & (pl.col("hi") > 0)).then(pl.lit("signed"))
            .otherwise(pl.lit("positive")))
    return f.with_columns(kind.alias("kind")).sort("gain_share", descending=True)


def family_of(feature: str) -> str:
    """Family = the name's leading letters, version digits dropped: ``zzhi2.c`` / ``zzhi.b`` / ``zzab2`` -> ZZHI / ZZHI / ZZAB."""
    m = re.match(r"[A-Za-z]+", feature)
    return m.group(0).upper() if m else feature.upper()


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
    rows, imps, thrs = [], [], []
    for f in files:
        summ, imp, thr = parse_model(f)
        where = locate(f, a.models)
        rows.append({**where, **summ})
        imps.append(imp.with_columns(*[pl.lit(v).alias(k) for k, v in where.items()]))
        thrs.append(thr)
    models = pl.DataFrame(rows)
    imp = pl.concat(imps)
    thr = pl.concat(thrs)
    csv = a.features or next(iter(sorted(a.models.glob("lgb_fn_*.csv"))), None)
    csv_names: list[str] = []
    if csv and csv.exists():
        lines = [ln.strip() for ln in csv.read_text(errors="replace").splitlines() if ln.strip()]
        print(f"{csv.name}: first lines {lines[:3]}, last {lines[-1:]}")
        csv_names = [ln.split(",")[0].strip() for ln in lines]
    # models saved without feature names carry Column_<i>; map i to the i-th name of the feature csv
    n_feat = int(models["n_features"].max())
    if imp["feature"].str.starts_with("Column_").all() and csv_names:
        if len(csv_names) == n_feat + 1:
            print(f"feature csv has {len(csv_names)} lines for {n_feat} features: first line {csv_names[0]!r} taken as header")
            csv_names = csv_names[1:]
        if len(csv_names) == n_feat:
            mapping = {f"Column_{i}": n for i, n in enumerate(csv_names)}
            imp = imp.with_columns(pl.col("feature").replace_strict(mapping).alias("feature"))
            thr = thr.with_columns(pl.col("feature").replace_strict(mapping).alias("feature"))
            print(f"Column_i -> csv name i for {n_feat} features (e.g. Column_0 = {csv_names[0]!r}, Column_97 = {mapping.get('Column_97')!r})")
        else:
            print(f"WARNING: csv has {len(csv_names)} names, models have {n_feat} features -- names left as Column_i")
    imp = imp.with_columns(
        pl.col("feature").map_elements(family_of, return_dtype=pl.String).alias("family"),
        (pl.col("gain") / pl.col("gain").sum().over("file")).alias("gain_share"))

    # family share = its features' gain summed within each model, then averaged over models
    per_model = imp.group_by("file", "family").agg(pl.col("gain_share").sum().alias("share"))
    fams = imp.group_by("family").agg(pl.col("feature").n_unique().alias("features")).join(
        per_model.group_by("family").agg(pl.col("share").mean().alias("mean_gain_share"),
                                         pl.col("share").min().alias("min_share"),
                                         pl.col("share").max().alias("max_share")), on="family")
    fams = fams.sort("mean_gain_share", descending=True).with_columns(
        pl.col("mean_gain_share").cum_sum().alias("cum_share"))
    listed = {family_of(n) for n in csv_names}
    if csv_names:
        print(f"feature csv: {len(csv_names)} names, {len(listed)} families")
    shipped = {p.name.upper() for p in a.hstats.iterdir() if p.is_dir()} if a.hstats.is_dir() else set()
    fams = fams.with_columns(pl.col("family").is_in(list(shipped)).alias("shipped"),
                             pl.col("family").is_in(list(listed)).alias("in_feature_csv"))

    a.out.mkdir(parents=True, exist_ok=True)
    models.write_parquet(a.out / "models.parquet")
    imp.write_parquet(a.out / "importance.parquet")
    fams.write_parquet(a.out / "families.parquet")
    fp = fingerprint(imp, thr)
    fp.write_parquet(a.out / "fingerprint.parquet")

    with pl.Config(tbl_rows=90, tbl_width_chars=220, tbl_cols=20, float_precision=4):
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
        print("\nfamilies by total gain share per model (shipped = folder in the sample):")
        print(fams.sort("mean_gain_share", descending=True))
        print(f"\nused but NOT shipped: {sorted(set(fams['family']) - shipped) or 'none'}")
        print(f"shipped but NOT used: {sorted(shipped - set(fams['family'])) or 'none'}")
        top = (imp.group_by("feature", "family").agg(pl.col("gain_share").mean())
               .sort("gain_share", descending=True).head(40))
        print("\ntop 40 features by mean gain share:")
        print(top)
        print("\nfingerprint of the top 60 features: training range, split thresholds (p10 / median / p90, distinct,"
              " share on half-integers, share at ~0) and the kind it suggests")
        with pl.Config(tbl_rows=70, tbl_cols=14, tbl_width_chars=230, float_precision=6, fmt_str_lengths=12):
            print(fp.head(60).select("feature", "gain_share", "lo", "hi", "t_p10", "t_med", "t_p90", "distinct",
                                     "half_int", "at_zero", "kind"))
            print("\nkinds per family (features of each kind):")
            print(fp.group_by("family").agg(pl.len().alias("features"), pl.col("gain_share").sum().alias("gain"),
                                            *[(pl.col("kind") == k).sum().alias(k) for k in
                                              ("const", "flag", "count", "unit", "signed", "positive")])
                  .sort("gain", descending=True))
    print(f"\ntables: {a.out}/{{models,importance,families}}.parquet")


if __name__ == "__main__":
    main()
