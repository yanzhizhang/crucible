r"""runS on 20261113: a one-step replay of the order decision -- from the logged decision rows, does a candidate rule say
"send" exactly where runS sent?

Each decision row carries runS's own executp, cpos, fa, lthr, hs, asq; R72-R75 found trade = executp - cpos, the side
thresholds (sell fa < lthr, buy fa > -lthr), |trade| >= 100, bursts that follow a live / just-ended order, and a send rate
that rises with how far fa is past the threshold in units of hs. Here every candidate rule is scored as a classifier of
``nullptr == 0`` (runS made an order at this decision), with the stock's live / last order taken from the real orders:

* rules: gate & over >= k (k in hs units) for a grid of k, with and without "no live order", with and without a recent
  order end (within T s) lowering k -- morning and afternoon apart (lthr == 0 all morning on 1113);
* precision, recall, F1 for each, and the best one per half day;
* qty: the sent qty against floor(|trade| / 100) x 100, that capped by asq (sells), by one lot, by a fraction of
  winsdQty -- the share of exact matches per side.

    python research/pm_factors/runs_replay.py --orders /work/crucible_data/pm_orders --date 20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def prf(pred: np.ndarray, true: np.ndarray) -> tuple[float, float, float]:
    tp = float((pred & true).sum())
    p = tp / max(pred.sum(), 1)
    r = tp / max(true.sum(), 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"), ((c("src_epoch") / 1e9 + 28800) % 86400 / 3600).alias("hour"))
    d = d.with_columns(pl.when(c("trade") < 0).then(c("lthr") - c("fa")).when(c("trade") > 0).then(c("fa") + c("lthr")).otherwise(None).alias("over"))
    d = d.with_columns((c("over") / c("hs")).alias("over_hs"), (c("nullptr") == 0).alias("made")).sort("src_epoch")

    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("f3").alias("exoid"))
    life = pl.read_parquet(root / "life.parquet")
    cx = life.filter(c("ev") == "OnECM").group_by("exoid").agg(c("ts").min().alias("t_cxl"))
    fl = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("fill_qty").sum().alias("fq"), c("ts").max().alias("t_fill"))
    o = po.join(cx, on="exoid", how="left").join(fl, on="exoid", how="left").with_columns(c("fq").fill_null(0.0))
    o = o.with_columns(pl.when(c("t_cxl").is_not_null()).then(c("t_cxl")).when(c("fq") >= c("qty")).then(c("t_fill"))
                       .otherwise(c("po_ts") + 7_000_000_000).alias("t_end")).sort("po_ts")
    # the stock's last order sent strictly before the decision (an order sent from this very decision is not "live" for it)
    j = d.join_asof(o.select("sid", (c("po_ts") + 1).alias("k_po"), c("po_ts").alias("last_po"), c("t_end").alias("last_end")),
                    left_on="src_epoch", right_on="k_po", by="sid", strategy="backward")
    j = j.with_columns((c("last_end") > c("src_epoch")).fill_null(False).alias("live"),
                       ((c("src_epoch") - c("last_end")) / 1e9).alias("since_end"))
    print(f"==== {a.date}: {j.height:,} decisions, {int(j['made'].sum()):,} made an order")
    for name, part in (("morning", j.filter(c("hour") < 12)), ("afternoon", j.filter(c("hour") >= 12))):
        true = part["made"].to_numpy()
        over = part["over_hs"].to_numpy().astype(float)
        lot = np.abs(part["trade"].to_numpy().astype(float)) >= 100
        live = part["live"].to_numpy()
        se = part["since_end"].to_numpy().astype(float)
        ok = np.isfinite(over)
        print(f"-- {name}: {part.height:,} rows, {int(true.sum()):,} made")
        best = (0.0, "")
        for k in (0, 0.5, 1, 2, 3, 4, 6, 8):
            for no_live in (False, True):
                for T, k2 in ((None, None), (10, 0), (10, 1), (30, 1)):
                    base = ok & lot & (over >= k)
                    if T is not None:
                        base = base | (ok & lot & (over >= k2) & (se >= 0) & (se <= T))
                    pred = base & ~live if no_live else base
                    p, r, f = prf(pred, true)
                    tag = f"over >= {k:g} hs{', no live order' if no_live else ''}{f', or >= {k2:g} hs within {T} s of an order end' if T else ''}"
                    if f > best[0]:
                        best = (f, tag)
                    if T is None or k in (2, 4):
                        print(f"   {tag:70s} precision {p:.3f} recall {r:.3f} F1 {f:.3f} (predicted {int(pred.sum()):,})")
        print(f"   best F1 {best[0]:.3f}: {best[1]}")
    print("-- qty: sent qty against candidate sizes (exact match share), per side")
    jj = o.sort("po_ts").join_asof(d.sort("src_epoch").select("sid", "src_epoch", "trade", "asq", "winsdQty", "asf", "executp"),
                                   left_on="po_ts", right_on="src_epoch", by="sid", strategy="backward")
    for sd in ("buy", "sell"):
        s = jj.filter(c("side") == sd)
        if s.is_empty():
            continue
        qty, tr, asq, w = (s[k].to_numpy().astype(float) for k in ("qty", "trade", "asq", "winsdQty"))
        lots = np.floor(np.abs(tr) / 100) * 100
        cand = {"floor(|trade|) lots": lots, "min(lots, asq)": np.minimum(lots, asq), "one lot": np.full_like(qty, 100),
                "min(lots, asq) if sell": np.minimum(lots, asq) if sd == "sell" else lots}
        for frac in (0.001, 0.002, 0.005, 0.01):
            cand[f"min(lots, {frac:g} x winsdQty in lots)"] = np.minimum(lots, np.floor(frac * w / 100) * 100)
        print(f"   {sd} ({s.height:,}): " + "; ".join(f"{k} {np.mean(qty == v):.3f}" for k, v in cand.items()))


if __name__ == "__main__":
    main()
