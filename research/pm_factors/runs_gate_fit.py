r"""runS on 20261113: what decides "made an order" among the decisions that pass the visible gate -- read off a small
LightGBM classifier (lightgbm is in 97's env) on everything the decision row and its neighbours carry.

R74-R78: |trade| >= 100 and the side threshold leave 2.6% made; bursts after an order, a rate that rises with how far fa
is past the threshold. Features per gate-passing decision: fa, |trade|, over / hs, hs, lthr, vpen, vr, mp, asq, cpos, pp,
executp and their changes since the stock's previous decision (fa, mp, hs, executp, trade), seconds since the previous
decision, since the stock's last order sent / ended, a live order, minute of day, the side. Prints the held-out AUC
(time split: first 70% of the day trains), split-gain importance, and for the top features the made rate by quantile bin.

    python research/pm_factors/runs_gate_fit.py --orders /work/crucible_data/pm_orders --date 20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    import lightgbm as lgb

    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).sort("sid", "src_epoch")
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"))
    d = d.with_columns(pl.when(c("trade") < 0).then(c("lthr") - c("fa")).when(c("trade") > 0).then(c("fa") + c("lthr")).otherwise(None).alias("over"))
    lag = ["fa", "ev_mp", "hs", "executp", "trade", "asq", "ev_cpos"]
    d = d.with_columns(*[(c(k) - c(k).shift(1).over("sid")).alias(f"d_{k}") for k in lag],
                       ((c("src_epoch") - c("src_epoch").shift(1).over("sid")) / 1e9).alias("dt_prev"),
                       (c("nullptr") == 0).shift(1).over("sid").cast(pl.Float64).alias("prev_made"))
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("f3").alias("exoid"))
    life = pl.read_parquet(root / "life.parquet")
    cx = life.filter(c("ev") == "OnECM").group_by("exoid").agg(c("ts").min().alias("t_cxl"))
    o = po.join(cx, on="exoid", how="left").with_columns(pl.coalesce(c("t_cxl"), c("po_ts") + 7_000_000_000).alias("t_end")).sort("po_ts")
    d = d.sort("src_epoch").join_asof(o.select("sid", (c("po_ts") + 1).alias("k"), c("po_ts").alias("last_po"), "t_end"), left_on="src_epoch", right_on="k",
                                      by="sid", strategy="backward")
    d = d.with_columns(((c("src_epoch") - c("last_po")) / 1e9).alias("since_po"), ((c("src_epoch") - c("t_end")) / 1e9).alias("since_end"),
                       (c("t_end") > c("src_epoch")).fill_null(False).cast(pl.Float64).alias("live"),
                       (((c("src_epoch") / 1e9 + 28800) % 86400) / 60).alias("minute"), c("trade").sign().alias("side"),
                       c("trade").abs().alias("abs_trade"), (c("over") / c("hs")).alias("over_hs"))
    g = d.filter((c("abs_trade") >= 100) & (c("over") > 0)).sort("src_epoch")
    feats = ["fa", "abs_trade", "over_hs", "hs", "lthr", "ev_vpen", "ev_vr", "ev_mp", "asq", "ev_cpos", "pp", "executp", "side", "minute",
             "dt_prev", "since_po", "since_end", "live", "prev_made", *[f"d_{k}" for k in lag]]
    feats = [f for f in feats if f in g.columns]
    X = g.select(feats).to_numpy().astype(float)
    y = (g["nullptr"] == 0).to_numpy().astype(int)
    n = len(y)
    cut = int(0.7 * n)
    print(f"==== {a.date}: {n:,} decisions pass |trade| >= 100 and the side threshold; made {y.mean():.4f}; train on the first {cut:,} (by time)")
    bst = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "verbose": -1},
                    lgb.Dataset(X[:cut], y[:cut]), num_boost_round=300)
    p = bst.predict(X[cut:])
    yt = y[cut:]
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    pos = yt == 1
    auc = (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(pos.sum() * (~pos).sum(), 1)
    top = np.argsort(-p)[: max(int(pos.sum()), 1)]
    print(f"  held-out AUC {auc:.4f}; precision at the base rate (top {len(top):,}) {yt[top].mean():.3f} (base {yt.mean():.4f})")
    imp = bst.feature_importance("gain")
    oi = np.argsort(-imp)
    print("  split-gain share: " + ", ".join(f"{feats[i]} {imp[i] / imp.sum():.3f}" for i in oi[:12]))
    for i in oi[:6]:
        v = X[:, i]
        ok = np.isfinite(v)
        qs = np.unique(np.nanpercentile(v[ok], [0, 10, 25, 50, 75, 90, 100]))
        b = np.digitize(v, qs[1:-1])
        print(f"  {feats[i]:>12}: made by bin " + " | ".join(f"[{qs[k]:.4g}, {qs[k + 1]:.4g}] {y[ok & (b == k)].mean():.4f} ({int((ok & (b == k)).sum()):,})"
                                                         for k in range(len(qs) - 1)) + f" | NaN {y[~ok].mean() if (~ok).any() else float('nan'):.4f} ({int((~ok).sum()):,})")


if __name__ == "__main__":
    main()
