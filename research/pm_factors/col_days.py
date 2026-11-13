r"""One assignment scored on two days: which columns carry IC on both, which only on the first, which hurt on the second.
Reads the ic_evidence.parquet of two ic_score runs of the same assignment (the ablation: IC lost without the column, t over
marks). A column that helps on A and hurts on B is either fitted to day A or fed by something day B's target has wrong; the
split by source (state / seed / xs / plain) says which.

    python research/pm_factors/col_days.py --a data/reports/stage6_r41_audit2 --b data/reports/stage6_r42_r41c --names 0430,1113
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import polars as pl


def kind(formula: str, source: str) -> str:
    if "gen.sd_" in formula:
        return "seed"
    if source.startswith("state"):
        return "state"
    m = re.match(r"(?:cal:)?(neu:|dmi:|dm:|z:|s?rank:|crank:)", formula)
    return m.group(1).rstrip(":") if m else "plain"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    ap.add_argument("--names", default="A,B")
    ap.add_argument("--t", type=float, default=2.0)
    a = ap.parse_args()
    na, nb = a.names.split(",")
    ea = pl.read_parquet(a.a / "ic_evidence.parquet").select("feature", "formula", "source", pl.col("ic_loss").alias("loss_a"), pl.col("t_loss").alias("t_a"))
    eb = pl.read_parquet(a.b / "ic_evidence.parquet").select("feature", pl.col("formula").alias("formula_b"), pl.col("ic_loss").alias("loss_b"), pl.col("t_loss").alias("t_b"))
    j = ea.join(eb, on="feature", how="full", coalesce=True)
    print(f"{na}: {ea.height} columns, {nb}: {eb.height}; in both {j.filter(pl.col('t_a').is_not_null() & pl.col('t_b').is_not_null()).height}; "
          f"only {na}: {j.filter(pl.col('t_b').is_null())['feature'].to_list()}; formula differs: {j.filter(pl.col('formula') != pl.col('formula_b'))['feature'].to_list()}")
    j = j.filter(pl.col("t_a").is_not_null() & pl.col("t_b").is_not_null()).with_columns(
        pl.struct("formula", "source").map_elements(lambda r: kind(r["formula"], r["source"]), return_dtype=pl.String).alias("kind"),
        pl.col("feature").str.extract(r"^([a-z]+)", 1).alias("family"))
    T = a.t
    j = j.with_columns(pl.when((pl.col("t_a") >= T) & (pl.col("t_b") >= T)).then(pl.lit("both"))
                       .when((pl.col("t_a") >= T) & (pl.col("t_b") <= -T)).then(pl.lit(f"helps_{na}_hurts_{nb}"))
                       .when((pl.col("t_a") <= -T) & (pl.col("t_b") >= T)).then(pl.lit(f"hurts_{na}_helps_{nb}"))
                       .when(pl.col("t_a") >= T).then(pl.lit(f"only_{na}"))
                       .when(pl.col("t_b") >= T).then(pl.lit(f"only_{nb}"))
                       .when(pl.col("t_b") <= -T).then(pl.lit(f"hurts_{nb}"))
                       .otherwise(pl.lit("neither")).alias("cls"))
    agg = [pl.len().alias("columns"), pl.col("loss_a").sum().alias(f"ic_{na}"), pl.col("loss_b").sum().alias(f"ic_{nb}")]
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=60, float_precision=4):
        print(f"\n== columns by class (|t| >= {T}; ic = summed IC lost without each column)")
        print(j.group_by("cls").agg(agg).sort("columns", descending=True))
        print("\n== by kind of formula")
        print(j.group_by("kind").agg(*agg, (pl.col("t_b") <= -T).sum().alias(f"hurt_{nb}"), ((pl.col("t_a") >= T) & (pl.col("t_b") >= T)).sum().alias("both")).sort(f"ic_{nb}"))
        print("\n== by kind x class")
        print(j.group_by("kind", "cls").agg(pl.len().alias("n")).pivot("cls", index="kind", values="n").fill_null(0).sort("kind"))
        print(f"\n== columns that hurt on {nb} (t <= -{T}), worst first")
        print(j.filter(pl.col("t_b") <= -T).sort("loss_b").select("feature", "formula", "kind", "loss_a", "t_a", "loss_b", "t_b"))
        print("\n== columns that help on both, by the smaller t")
        print(j.filter(pl.col("cls") == "both").with_columns(pl.min_horizontal("t_a", "t_b").alias("t_min")).sort("t_min", descending=True)
              .select("feature", "formula", "kind", "loss_a", "t_a", "loss_b", "t_b").head(60))
        print(f"\n== rank correlation of the per-column IC loss between the two days: {j.select(pl.corr('loss_a', 'loss_b', method='spearman')).item():.3f}")
    drop = j.filter(pl.col("t_b") <= -T)["feature"].to_list()
    print("hurt list: " + ",".join(drop))


if __name__ == "__main__":
    main()
