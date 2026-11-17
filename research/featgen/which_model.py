r"""Which model generation did 224 score with? Our filled matrix through every generation / fit, against PM's logged prediction.

Our features are not PM's, so no generation reproduces PM's numbers; but the generation PM really ran should agree with
PM's prediction clearly better than the others on the SAME matrix. Per day, per model root, per fit (and per month folder
when a root has several): the mean per-mark rank correlation with PM's prediction (missing_damage.sim), the Pearson
correlation, and our score's scale against PM's (std ratio). Read-only on inputs; work files under --work are removed.

    python research/featgen/which_model.py --days 20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" \
        --roots /work/crucible_data/models_gen/A,/work/crucible_data/models_gen/B,/work/crucible_data/models_gen/C,/data/prod/models/tk5m
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402
from missing_damage import sim  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--subs", default="s1,s2,s3,s4")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--work", type=Path, default=Path("data/which_model_work"))
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    exe = P.ensure_binary()
    a.work.mkdir(parents=True, exist_ok=True)
    for d in [x for x in a.days.split(",") if x]:
        xd = Path(a.x_dir.format(mmdd=d[4:], date=d))
        x = pl.read_parquet(xd / "x.parquet")
        pr = pl.read_parquet(xd / "pred.parquet")
        if "date" in x.columns:
            x, pr = x.filter(pl.col("date") == int(d)), pr.filter(pl.col("date") == int(d))
        x = x.with_columns(pl.col("sid").cast(pl.Int64)).sort("sid", "mark")
        pm = x.select("sid", "mark").join(pr.with_columns(pl.col("sid").cast(pl.Int64)).select("sid", "mark", "pm"), on=["sid", "mark"], how="left")["pm"].to_numpy().astype(float)
        marks = x["mark"].to_numpy()
        X = np.full((x.height, len(names)), np.nan)
        for i, n in enumerate(names):
            if n in x.columns:
                X[:, i] = x[n].cast(pl.Float64).to_numpy()
        P.write_x(a.work / "x.bin", X)
        print(f"== {d}: {x.height:,} rows, {int(np.isfinite(pm).sum()):,} with PM's prediction, {int(np.isfinite(X).any(axis=0).sum())} columns filled")
        res = []
        for root in [Path(r) for r in a.roots.split(",") if r]:
            if not root.exists():
                print(f"   {root}: missing")
                continue
            for mdir in sorted(p for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name)):
                for fdir in sorted(p for p in (mdir / "lgb").iterdir() if p.is_dir()) if (mdir / "lgb").exists() else []:
                    files = [fdir / f"model_{s}.txt" for s in a.subs.split(",") if (fdir / f"model_{s}.txt").exists()]
                    if not files:
                        continue
                    subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files), "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
                    s = P.read_scores(a.work / "p.bin").copy()
                    m = np.isfinite(s) & np.isfinite(pm)
                    res.append((sim(s, pm, marks), float(np.corrcoef(s[m], pm[m])[0, 1]), float(s[m].std() / pm[m].std()), f"{root.name}/{mdir.name}/{fdir.name}", len(files)))
        for sm, pe, sd, tag, nf in sorted(res, reverse=True):
            print(f"   rank-corr {sm:.4f}  pearson {pe:.4f}  std ours/PM {sd:.3f}  {tag}  ({nf} files)")
        for f in ("x.bin", "p.bin"):
            (a.work / f).unlink(missing_ok=True)
    print("[done]")


if __name__ == "__main__":
    main()
