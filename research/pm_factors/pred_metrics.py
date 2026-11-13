r"""IC, rank IC and IR of our prediction and of PM's against the realised forward return, and of ours against PM's.
Per mark: Pearson and Spearman across stocks; then mean, std, IR = mean / std over marks, t = IR * sqrt(marks), share > 0.
Also the decile spread: mean forward return of the top decile minus the bottom decile by each prediction, in bp.

--pred is an ic_score --dump-pred run WITH --target (ours, pm = PM's prediction); --fwd is one WITHOUT --target on the same
day and marks (its "pm" column is then the forward 5-minute log mid return).

    python research/pm_factors/pred_metrics.py --pred data/reports/stage6_r43_0430_r41c --fwd data/reports/stage6_r43_0430_fwd --tag "0430 r41c"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def rk(x: np.ndarray) -> np.ndarray:
    return pl.Series(x).rank("average").to_numpy()


def stats(v: list[float]) -> str:
    a = np.array(v)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return "n/a"
    sd = a.std(ddof=1)
    return f"mean {a.mean():+.4f}  std {sd:.4f}  IR {a.mean() / sd:+.3f}  t {a.mean() / sd * np.sqrt(a.size):+.1f}  >0 {np.mean(a > 0):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--fwd", type=Path, required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--q", type=float, default=0.1)
    a = ap.parse_args()
    p = pl.read_parquet(a.pred / "pred.parquet").select("date", "sid", "mark", "ours", "pm")
    f = pl.read_parquet(a.fwd / "pred.parquet").select("date", "sid", "mark", pl.col("pm").alias("fwd"))
    d = p.join(f, on=["date", "sid", "mark"], how="inner").filter(pl.col("ours").is_finite() & pl.col("pm").is_finite() & pl.col("fwd").is_finite())
    pairs = (("ours", "fwd"), ("pm", "fwd"), ("ours", "pm"))
    ic = {k: [] for k in pairs}
    ric = {k: [] for k in pairs}
    spread = {"ours": [], "pm": []}
    n_st = []
    for _, g in d.group_by("date", "mark"):
        if g.height < 30:
            continue
        n_st.append(g.height)
        col = {c: g[c].to_numpy() for c in ("ours", "pm", "fwd")}
        r = {c: rk(v) for c, v in col.items()}
        for x, y in pairs:
            if col[x].std() > 0 and col[y].std() > 0:
                ic[(x, y)].append(float(np.corrcoef(col[x], col[y])[0, 1]))
                ric[(x, y)].append(float(np.corrcoef(r[x], r[y])[0, 1]))
        k = max(int(g.height * a.q), 1)
        for c in spread:
            o = np.argsort(col[c])
            spread[c].append(float(col["fwd"][o[-k:]].mean() - col["fwd"][o[:k]].mean()) * 1e4)
    print(f"== {a.tag}: {len(n_st)} marks, {int(np.mean(n_st)) if n_st else 0} stocks a mark, {d.height:,} rows (forward return = 5-minute log mid return)")
    for x, y in pairs:
        name = {"fwd": "forward return", "pm": "PM's prediction"}[y]
        who = {"ours": "ours", "pm": "PM's"}[x]
        print(f"  {who:5s} vs {name:16s} IC      {stats(ic[(x, y)])}")
        print(f"  {who:5s} vs {name:16s} rank IC {stats(ric[(x, y)])}")
    for c, w in (("ours", "ours"), ("pm", "PM's")):
        print(f"  {w:5s} top - bottom decile forward return (bp) {stats(spread[c])}")


if __name__ == "__main__":
    main()
