r"""PM's feature names -> our formulas, as a family / stem / leaf tree (``zzqf7.q.c2``: family ZZQF, stem zzqf7, leaf q.c2).

Reads one ic_score round (``ic_assign.json`` + ``ic_evidence.parquet``) and the model fingerprint (all 209 columns, gain share,
kind); prints the tree with each column's formula and evidence and writes ``name_map.csv`` next to the assignment. Unassigned
columns are listed too, so the gaps show per family. No model, no data: runs in a second.

    python research/pm_factors/name_map.py --round data/reports/stage6_r15
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=Path, required=True, help="an ic_score --out directory")
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    a = ap.parse_args()
    assign = json.loads((a.round / "ic_assign.json").read_text())
    fp = pl.read_parquet(a.fingerprint)
    fp = fp.select("feature", *[c for c in ("gain_share", "kind") if c in fp.columns]).unique("feature", keep="first")
    ev_f = a.round / "ic_evidence.parquet"
    ev = pl.read_parquet(ev_f).select("feature", "cdf_gap", "t_loss", "grade") if ev_f.exists() else None
    df = fp.with_columns(
        pl.col("feature").str.extract(r"^([A-Za-z]+)", 1).str.to_uppercase().alias("family"),
        pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"),
        pl.col("feature").str.extract(r"^[^.]+\.(.+)$", 1).fill_null("").alias("leaf"),
        pl.col("feature").replace_strict(assign, default=None, return_dtype=pl.String).alias("formula"))
    if ev is not None:
        df = df.join(ev, on="feature", how="left")
    fam = df.group_by("family").agg(pl.col("gain_share").sum().alias("fam_gain"), pl.col("formula").is_not_null().sum().alias("n_ok"),
                                    pl.len().alias("n"))
    df = df.join(fam, on="family").sort(["fam_gain", "stem", "leaf"], descending=[True, False, False])
    out = a.round / "name_map.csv"
    df.select("family", "stem", "leaf", "feature", *[c for c in ("gain_share", "kind") if c in df.columns], "formula",
              *[c for c in ("grade", "cdf_gap", "t_loss") if c in df.columns]).write_csv(out)
    n_ok = df["formula"].is_not_null().sum()
    print(f"{n_ok} of {df.height} columns mapped ({df.filter(pl.col('formula').is_not_null())['gain_share'].sum():.1%} of model gain)")
    for (f,), g in df.group_by("family", maintain_order=True):
        r = g.row(0, named=True)
        print(f"\n{f}  {r['n_ok']}/{r['n']} mapped, gain {r['fam_gain']:.1%}")
        for (s,), gs in g.group_by("stem", maintain_order=True):
            print(f"  {s}")
            for x in gs.iter_rows(named=True):
                ev_s = (f"  [{x['grade']} gap {x['cdf_gap']:.3f} t {x['t_loss']:.1f}]"
                        if x.get("grade") is not None and x.get("t_loss") is not None else "")
                leaf = x["leaf"] or "(stem)"
                print(f"    {leaf:<8} {x['gain_share']:.2%}  {x['formula'] or '-'}{ev_s}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
