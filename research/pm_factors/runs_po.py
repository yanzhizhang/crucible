r"""PM's order trace against what chrono received: is price / quantity swapped on the way (20261118)?

PM's LiveEs trace line is "nowTs:<t>SO,<epoch>,<ac>,<OrderId>,<exOrderId>,<Sid>,<Price>,<Qty>..." (po table: px = the
7th field, qty = the 8th). chrono logs what the plugin asked for as trk ORD (symbol, qty, px, side) and what it sent as
trk SEND. For every SO line, the next ORD (and SEND) of the same stock within --window seconds:
  (a) qty == SO Qty and px == SO Price: nothing swapped reaches chrono;
  (b) qty == SO Price (as an integer) and px == SO Qty: swapped as compiled;
  (c) neither, or no ORD / SEND in the window.
Three examples per case; SO Qty on the lot grid and SO Price on the tick grid; SO lines with Qty 0 (a disabled stock).
Read-only.

    python research/pm_factors/runs_po.py --orders /work/crucible_data/pm_orders --date 20261118
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
    ap.add_argument("--window", type=float, default=2.0)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("po_ts").cast(pl.Int64), c("px").cast(pl.Float64), c("qty").cast(pl.Float64)).sort("po_ts")
    trk = pl.read_parquet(root / "trk.parquet")
    print(f"== {a.date}: {po.height:,} SO lines (columns {po.columns}); trk kinds {dict(sorted(trk.group_by('kind').len().iter_rows(), key=lambda kv: -kv[1])[:12])}")
    print(f"   SO Qty: == 0 on {int((po['qty'] == 0).sum())}, a multiple of 100 on {float((po['qty'] % 100 == 0).mean()):.4f}, min / median / max {po['qty'].min():.0f} / {po['qty'].median():.0f} / {po['qty'].max():.0f};"
          f" SO Price on the 0.01 grid on {float(((po['px'] * 100 - (po['px'] * 100).round()).abs() < 1e-6).mean()):.4f}, min / median / max {po['px'].min():.2f} / {po['px'].median():.2f} / {po['px'].max():.2f}")
    with pl.Config(tbl_cols=20, tbl_width_chars=200, tbl_rows=3, fmt_float="full"):
        print(po.head(3))
    for kind in ("ORD", "SEND"):
        t = trk.filter(c("kind") == kind)
        keep = [k for k in t.columns if t[k].null_count() < t.height]
        tk = next((k for k in ("now", "ts", "ts_ns", "send_ts") if k in keep), None)
        if t.is_empty() or tk is None or not {"symbol", "qty", "px"} <= set(keep):
            print(f"-- {kind}: {t.height} rows, filled columns {keep}: cannot match (need symbol, qty, px and a time)")
            continue
        t = t.select(c("symbol").str.slice(0, 6).cast(pl.Int64).alias("sid"), c(tk).cast(pl.Int64).alias("t"), c("qty").cast(pl.Float64).alias("c_qty"), c("px").cast(pl.Float64).alias("c_px"),
                     (c("side").cast(pl.String) if "side" in keep else pl.lit("")).alias("side")).sort("t")
        j = po.join_asof(t, left_on="po_ts", right_on="t", by="sid", strategy="forward", tolerance=int(a.window * 1e9))
        n = j.height
        has = j["t"].is_not_null().to_numpy()
        q, p, cq, cp = (j[k].to_numpy().astype(float) for k in ("qty", "px", "c_qty", "c_px"))
        same = has & (np.abs(cq - q) < 1e-6) & (np.abs(cp - p) < 1e-6)
        swap = has & (np.abs(cq - np.floor(p)) < 1e-6) & (np.abs(cp - q) < 1e-6) & ~same
        other = has & ~same & ~swap
        print(f"-- {kind} ({t.height:,} rows; time column {tk}): next {kind} of the stock within {a.window:g} s of the SO line: found {int(has.sum()):,} of {n:,};"
              f" (a) same qty and px {int(same.sum()):,}; (b) swapped {int(swap.sum()):,}; (c) neither {int(other.sum()):,}, none found {int((~has).sum()):,}")
        if has.any():
            lag = (j["t"] - j["po_ts"]).to_numpy()[has] / 1e6
            print(f"   {kind} time - SO time, ms, q1 / q50 / q99: {np.percentile(lag, 1):.3f} / {np.percentile(lag, 50):.3f} / {np.percentile(lag, 99):.3f}")
        for name, m in (("(a)", same), ("(b)", swap), ("(c) neither", other)):
            for r in j.filter(pl.Series(m)).head(3).iter_rows(named=True):
                print(f"   {name}  sid {r['sid']:06d}  SO px {r['px']} qty {r['qty']:.0f}  ->  {kind} px {r['c_px']} qty {r['c_qty']:.0f} side {r['side']}")
        for r in j.filter(pl.Series(~has)).head(3).iter_rows(named=True):
            print(f"   none found  sid {r['sid']:06d}  SO px {r['px']} qty {r['qty']:.0f}  f2 {r.get('f2')} f3 {r.get('f3')} f4 {r.get('f4')}")
    print("[done]")


if __name__ == "__main__":
    main()
