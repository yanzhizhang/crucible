r"""runS on 20261113: the 61% of orders that follow neither a made decision nor a cancel confirm (R81) -- the re-quote
loop of the execution service. For each such order: seconds since the stock's previous send, since that order's cancel
request, and where that order's cancel confirm falls (before or after this send: replace before the confirm?); its
qty against the previous order's unfilled rest; how many re-quotes one made decision gets, and what ends the chain
(a fill, a new decision, the side flipping, nothing). Read-only on pm_orders' parquet.

    python research/pm_factors/runs_requote.py --orders /work/crucible_data/pm_orders --date 20261113
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
    ecm = life.filter(c("ev") == "OnECM").group_by("exoid").agg(c("ts").min().alias("t_ecm"))
    req = life.filter(c("ev") == "cxl_req").group_by("exoid").agg(c("ts").min().alias("t_req"))
    fq = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("fill_qty").sum().alias("filled"), c("ts").min().alias("t_fill"))
    o = (po.join(ecm, on="exoid", how="left").join(req, on="exoid", how="left").join(fq, on="exoid", how="left")
         .with_columns(c("filled").fill_null(0.0)).sort("sid", "po_ts"))
    o = o.with_columns(*[c(k).shift(1).over("sid").alias(f"p_{k}") for k in ("po_ts", "t_ecm", "t_req", "t_fill", "qty", "filled", "side", "px")])
    o = o.sort("po_ts").join_asof(made, left_on="po_ts", right_on="t_made", by="sid", strategy="backward")
    o = o.with_columns(((c("po_ts") - c("t_made")) / 1e9).alias("s_made"))
    # an order follows its decision when the decision came after the stock's previous send (and within 1 s)
    o = o.with_columns(((c("s_made") < 1.0) & (c("p_po_ts").is_null() | (c("t_made") > c("p_po_ts")))).fill_null(False).alias("from_decision"))
    rq = o.filter(~c("from_decision") & c("p_po_ts").is_not_null())
    g = lambda df, k: df[k].to_numpy().astype(float)  # noqa: E731
    t = g(rq, "po_ts")
    print(f"==== {a.date}: {o.height:,} orders; from a decision {int(o['from_decision'].sum()):,}, re-quotes (another order of the stock before it) {rq.height:,}")
    print(f"  re-quote: seconds since the previous send {q((t - g(rq, 'p_po_ts')) / 1e9)}; since the previous order's cancel request {q((t - g(rq, 'p_t_req')) / 1e9)}; "
          f"previous order's cancel confirm minus this send (s) {q((g(rq, 'p_t_ecm') - t) / 1e9)}")
    pe, pq_, pf = g(rq, "p_t_ecm"), g(rq, "p_qty"), g(rq, "p_filled")
    print(f"  previous order cancel-confirmed before this send {np.mean(pe < t):.3f}, after {np.mean(pe >= t):.3f}, never {np.mean(~np.isfinite(pe)):.3f}; "
          f"previous fully filled {np.mean(pf >= pq_):.3f}; qty == previous rest {np.mean(g(rq, 'qty') == pq_ - pf):.3f}, == previous qty {np.mean(g(rq, 'qty') == pq_):.3f}; "
          f"same side {np.mean(rq['side'].to_numpy() == rq['p_side'].to_numpy()):.3f}; px - previous px {q(g(rq, 'px') - g(rq, 'p_px'))}")
    o = o.with_columns(c("from_decision").cast(pl.Int32).cum_sum().over("sid").alias("chain"))
    ch = o.filter(c("chain") > 0).group_by("sid", "chain").agg(pl.len().alias("n"), c("filled").sum().alias("filled"), c("qty").first().alias("q0"),
                                                               c("side").n_unique().alias("sides"), ((c("po_ts").max() - c("po_ts").min()) / 1e9).alias("secs"))
    n, fl, q0 = (ch[k].to_numpy().astype(float) for k in ("n", "filled", "q0"))
    print(f"  chains (a decision's order and the re-quotes until the next decision): {ch.height:,}; orders per chain {q(n)} (max {int(n.max()) if n.size else 0}); "
          f"seconds {q(ch['secs'].to_numpy())}; filled share of the first qty {q(fl / q0)}; any fill {np.mean(fl > 0):.3f}; one side {np.mean(ch['sides'].to_numpy() == 1):.3f}")
    for k in (1, 2, 3, 4, 5, 6, 8, 10):
        m = n == k
        if m.any():
            print(f"    {k:>2} orders: {int(m.sum()):>5,} chains, any fill {np.mean(fl[m] > 0):.3f}, filled share {np.mean(fl[m] / q0[m]):.3f}")


if __name__ == "__main__":
    main()
