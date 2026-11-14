r"""runS on 20261113: two layers? The decision (alg.cpp) makes 2,389 orders, but 5,948 are sent -- are the rest the
execution service (es.cpp) re-placing after its own cancel?

For every sent order (Po) of a stock: the time since the stock's latest decision that made an order (nullptr false),
since its latest cancel confirm (OnECM), since its latest fill; whether its qty equals the cancelled remainder
(exSoQ of the cancel request) or the previous order's qty minus its fills; its price against the previous order's
price. Read-only on pm_orders' parquet; prints shares and quantiles.

    python research/pm_factors/runs_es.py --orders /work/crucible_data/pm_orders --date 20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def q(x, qs=(10, 50, 90)) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "/".join(f"{v:.4g}" for v in np.percentile(x, qs)) if x.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    made = d.filter(c("nullptr") == 0).select("sid", c("src_epoch").alias("t_made")).sort("t_made")
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("f3").alias("exoid")).sort("po_ts")
    life = pl.read_parquet(root / "life.parquet")
    sid_of = po.select("exoid", "sid")
    ecm = life.filter(c("ev") == "OnECM").join(sid_of, on="exoid", how="inner").select("sid", c("ts").alias("t_ecm"), c("exoid").alias("ecm_oid")).sort("t_ecm")
    fil = life.filter(c("ev") == "OnEOM").join(sid_of, on="exoid", how="inner").select("sid", c("ts").alias("t_fill")).sort("t_fill")
    req = life.filter(c("ev") == "cxl_req").select(c("exoid").alias("ecm_oid"), "cxl_qty")
    prev = po.with_columns(c("qty").shift(1).over("sid").alias("prev_qty"), c("px").shift(1).over("sid").alias("prev_px"),
                           c("side").shift(1).over("sid").alias("prev_side"), c("exoid").shift(1).over("sid").alias("prev_oid"))
    fq = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("fill_qty").sum().alias("prev_filled"))
    j = (prev.join_asof(made, left_on="po_ts", right_on="t_made", by="sid", strategy="backward")
         .join_asof(ecm, left_on="po_ts", right_on="t_ecm", by="sid", strategy="backward")
         .join_asof(fil, left_on="po_ts", right_on="t_fill", by="sid", strategy="backward")
         .join(req, on="ecm_oid", how="left").join(fq, left_on="prev_oid", right_on="exoid", how="left")
         .with_columns(c("prev_filled").fill_null(0.0)))
    g = lambda k: j[k].to_numpy().astype(float)  # noqa: E731
    t = g("po_ts")
    s_made, s_ecm, s_fill = (t - g("t_made")) / 1e9, (t - g("t_ecm")) / 1e9, (t - g("t_fill")) / 1e9
    print(f"==== {a.date}: {j.height:,} orders sent, {made.height:,} decisions made an order")
    after_made = np.isfinite(s_made) & (s_made < 1.0) & ~(np.isfinite(s_ecm) & (s_ecm < s_made))
    after_ecm = np.isfinite(s_ecm) & (s_ecm < 1.0) & ~after_made
    print(f"  sent within 1 s of a made decision (and no cancel confirm in between): {after_made.mean():.3f}; within 1 s after a cancel confirm instead: "
          f"{after_ecm.mean():.3f}; neither: {np.mean(~after_made & ~after_ecm):.3f}")
    print(f"  seconds since the made decision {q(s_made[after_made])}; since the cancel confirm {q(s_ecm[after_ecm])}")
    for name, m in (("after a made decision", after_made), ("after a cancel confirm", after_ecm), ("neither", ~after_made & ~after_ecm)):
        if not m.any():
            continue
        same_side = (j["side"].to_numpy() == j["prev_side"].to_numpy())[m]
        rem = g("prev_qty")[m] - g("prev_filled")[m]
        print(f"  {name} ({int(m.sum()):,}): same side as the stock's previous order {same_side.mean():.3f}; qty == previous qty {np.mean(g('qty')[m] == g('prev_qty')[m]):.3f}, "
              f"== previous qty - its fills {np.mean(g('qty')[m] == rem):.3f}, == cancelled remainder {np.mean(g('qty')[m] == g('cxl_qty')[m]):.3f}; "
              f"px - previous px {q(g('px')[m] - g('prev_px')[m])}; seconds since the last fill {q(s_fill[m])}")
    ch = j.with_columns(pl.Series("kind", np.where(after_made, "M", np.where(after_ecm, "C", "N")))).group_by("sid", maintain_order=True).agg(c("kind").str.join(""))
    lens = [len(x) for x in ch["kind"].to_list()]
    print(f"  per stock the sequence of order kinds (M = after a decision, C = after a cancel, N = neither), longest stocks: "
          + " | ".join(f"{s}:{k[:60]}" for s, k, _ in sorted(zip(ch["sid"].to_list(), ch["kind"].to_list(), lens, strict=True), key=lambda r: -r[2])[:4]))


if __name__ == "__main__":
    main()
