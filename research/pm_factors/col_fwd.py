r"""Column by column against the realised forward return, over several days. Read-only on ic_score outputs.

Each ``--runs`` directory is an ic_score run WITHOUT --target (its IC is against the forward 5-minute return), so its
ic_evidence.parquet holds, per assigned column, the IC lost when that column goes back to missing. A column whose formula
is PM's should help predict the return on most days (PM's model was trained for that); one that hurts on every day is a
suspect formula, whatever it does for the similarity with PM's prediction on the one production day (``--pm`` run).

Printed: per column the loss on every day, mean, t over days, days helping; the classes; how the columns picked as hurting
on the ``--fit`` days behave on the other days (out of sample); the same against the similarity loss of ``--pm``.
``--assign`` + ``--out``: the assignment without the columns that hurt on every fit day.

    python research/pm_factors/col_fwd.py --runs 0430=data/reports/stage6_r43_0430_fwd,1106=data/reports/stage6_r44_1106_fwd \
        --pm data/reports/stage6_r41_audit2 --fit 0430,1106 --assign A.json --out A_pruned.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="name=dir,name=dir: no-target ic_score runs of the same assignment")
    ap.add_argument("--pm", type=Path, help="a --target run of the same assignment (similarity with PM's prediction)")
    ap.add_argument("--fit", default="", help="comma list of run names used to pick the hurting columns (default: all)")
    ap.add_argument("--mean-t", type=float, default=1.0, help="hurting = negative on every fit day and mean per-day t <= -this")
    ap.add_argument("--assign", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = [x.split("=", 1) for x in a.runs.split(",") if "=" in x]
    j = None
    names = []
    for name, d in runs:
        f = Path(d) / "ic_evidence.parquet"
        if not f.exists():
            print(f"  {name}: no {f}, skipped")
            continue
        e = pl.read_parquet(f).select("feature", "formula", pl.col("ic_loss").alias(f"l_{name}"), pl.col("t_loss").alias(f"t_{name}"))
        j = e if j is None else j.join(e.drop("formula"), on="feature", how="inner")
        names.append(name)
    if j is None or not names:
        print("no runs")
        return
    fit = [x for x in a.fit.split(",") if x in names] or names
    test = [x for x in names if x not in fit]
    L = np.nan_to_num(j.select([f"l_{n}" for n in names]).to_numpy().astype(float))
    T = np.nan_to_num(j.select([f"t_{n}" for n in names]).to_numpy().astype(float))
    n = len(names)
    sd = L.std(axis=1, ddof=1) if n > 1 else np.full(L.shape[0], np.nan)
    fi = [names.index(x) for x in fit]
    ti = [names.index(x) for x in test]
    j = j.with_columns(pl.Series("mean", L.mean(axis=1)), pl.Series("t_days", L.mean(axis=1) / np.where(sd > 0, sd / np.sqrt(n), np.nan)),
                       pl.Series("days_pos", (L > 0).sum(axis=1)), pl.Series("fit_mean", L[:, fi].mean(axis=1)),
                       pl.Series("fit_t", T[:, fi].mean(axis=1)), pl.Series("fit_neg", (L[:, fi] < 0).all(axis=1)),
                       pl.Series("fit_pos", (L[:, fi] > 0).all(axis=1)),
                       pl.Series("test_mean", L[:, ti].mean(axis=1) if ti else np.full(L.shape[0], np.nan)))
    hurt = j.filter(pl.col("fit_neg") & (pl.col("fit_t") <= -a.mean_t))
    help_ = j.filter(pl.col("fit_pos") & (pl.col("fit_t") >= a.mean_t))
    print(f"{j.height} columns over {n} days {names}; fit days {fit}, test days {test}")
    print(f"summed loss per day (IC against the forward return lost without each column, summed over columns): "
          + ", ".join(f"{nm} {L[:, i].sum():+.4f}" for i, nm in enumerate(names)))
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=50, float_precision=4):
        print(f"\n== classes over all {n} days")
        cls = j.with_columns(pl.when(pl.col("t_days") >= 2).then(pl.lit("helps (t>=2)")).when(pl.col("t_days") <= -2).then(pl.lit("hurts (t<=-2)"))
                             .otherwise(pl.lit("unclear")).alias("cls"))
        print(cls.group_by("cls").agg(pl.len().alias("columns"), pl.col("mean").sum().alias("sum_mean_loss")).sort("sum_mean_loss"))
        print(f"\n== picked on the fit days: hurting {hurt.height} (negative on every fit day, mean t <= -{a.mean_t}), helping {help_.height}")
        if ti:
            for nm, g in (("hurting", hurt), ("helping", help_), ("all", j)):
                if g.height:
                    tm = g["test_mean"].to_numpy()
                    print(f"  {nm:8s}: {g.height:3d} columns, fit mean loss {g['fit_mean'].mean():+.5f}, test mean loss {tm.mean():+.5f}, "
                          f"share with the same sign on the test days {np.mean(np.sign(tm) == np.sign(g['fit_mean'].to_numpy())):.2f}")
        cols = ["feature", "formula"] + [f"l_{x}" for x in names] + ["mean", "t_days", "days_pos"]
        print("\n== hurting columns (picked on the fit days), worst first")
        print(hurt.sort("fit_mean").select(cols))
        print("\n== the 25 columns that help the forward return most")
        print(j.sort("mean", descending=True).select(cols).head(25))
        print("\n== the 25 that hurt it most over all days")
        print(j.sort("mean").select(cols).head(25))
        if n > 1:
            c = np.corrcoef(np.argsort(np.argsort(L, axis=0), axis=0).T.astype(float))
            print("\n== rank correlation of the per-column loss between days")
            print(pl.DataFrame({"day": names, **{nm: np.round(c[:, i], 3) for i, nm in enumerate(names)}}))
        if a.pm and (a.pm / "ic_evidence.parquet").exists():
            p = pl.read_parquet(a.pm / "ic_evidence.parquet").select("feature", pl.col("ic_loss").alias("pm_loss"), pl.col("t_loss").alias("pm_t"))
            k = j.join(p, on="feature", how="inner")
            print(f"\n== against the similarity with PM's prediction ({a.pm}): {k.height} columns; rank correlation of pm_loss with the mean "
                  f"forward loss {k.select(pl.corr('pm_loss', 'mean', method='spearman')).item():.3f}")
            k = k.with_columns(pl.when(pl.col("pm_t") >= 2).then(pl.lit("pm_helps")).when(pl.col("pm_t") <= -2).then(pl.lit("pm_hurts")).otherwise(pl.lit("pm_flat")).alias("pm"),
                               pl.when(pl.col("t_days") >= 2).then(pl.lit("fwd_helps")).when(pl.col("t_days") <= -2).then(pl.lit("fwd_hurts")).otherwise(pl.lit("fwd_unclear")).alias("fwd"))
            print(k.group_by("pm", "fwd").agg(pl.len().alias("columns"), pl.col("pm_loss").sum(), pl.col("mean").sum().alias("fwd_mean_loss")).sort("pm", "fwd"))
            print("  hurting columns and what they do for the similarity:")
            print(k.filter(pl.col("feature").is_in(hurt["feature"].to_list())).sort("pm_loss", descending=True).select("feature", "formula", "pm_loss", "pm_t", "mean", "t_days"))
    drop = hurt["feature"].to_list()
    print("hurting list: " + ",".join(drop))
    if a.assign and a.out:
        b = json.loads(a.assign.read_text())
        kept = {f: e for f, e in b.items() if f not in set(drop)}
        a.out.write_text(json.dumps(kept, indent=1))
        print(f"wrote {a.out}: {len(kept)} of {len(b)} columns")


if __name__ == "__main__":
    main()
