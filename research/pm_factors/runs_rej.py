r"""Rejected orders on a live day: does PM's library keep trying to cancel them, and what does chrono log?

Decoded LiveEs::OnFOM (order rejected) unwinds the pending quantity but does not erase the order, so its cancel sweep
should retry it (with a back-off). For each chrono trk REJ (strategy order idx): its ORD line, PM's SO trace line of that
order (matched by stock, time, price, quantity; f3 = PM's exOid), then PM's order-life lines for that exOid (cancel
requests "cxl_req", and whatever else the life table holds) with the gaps between attempts, and chrono's trk CXL /
CXL_RSP lines for that idx. Writes <out>/rej_idx.txt (the idx list) for a grep of the raw log. Read-only on inputs.

    python research/pm_factors/runs_rej.py --orders /work/crucible_data/pm_orders --date 20261118 --out data/reports/r186
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
    ap.add_argument("--out", type=Path, default=Path("data/reports/r186"))
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    trk = pl.read_parquet(root / "trk.parquet")
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("po_ts").cast(pl.Int64)).sort("po_ts")
    life = pl.read_parquet(root / "life.parquet") if (root / "life.parquet").exists() else pl.DataFrame()
    tk = next(k for k in ("now", "ts", "ts_ns") if k in trk.columns)
    rej = trk.filter(c("kind") == "REJ")
    keep = [k for k in rej.columns if rej[k].null_count() < rej.height]
    print(f"== {a.date}: {rej.height} trk REJ; filled columns {keep}")
    if "reason" in keep:
        print("   reasons: " + "; ".join(f"{n} x {str(r)[:90]}" for r, n in sorted(rej.group_by(c("reason").str.replace_all(r"\d+", "#")).len().iter_rows(), key=lambda kv: -kv[1])[:5]))
    ordl = trk.filter(c("kind") == "ORD").select(c("idx").cast(pl.Int64), c("symbol").str.slice(0, 6).cast(pl.Int64).alias("sid"), c(tk).cast(pl.Int64).alias("t"),
                                                 c("qty").cast(pl.Float64).alias("o_qty"), c("px").cast(pl.Float64).alias("o_px"), c("side")).sort("t")
    m = ordl.join_asof(po.select("sid", "po_ts", "f3", "f4", "px", "qty"), left_on="t", right_on="po_ts", by="sid", strategy="backward", tolerance=2_000_000_000)
    m = m.with_columns(((c("px") - c("o_px")).abs() < 1e-6).alias("same_px"), ((c("qty") - c("o_qty")).abs() < 1e-6).alias("same_qty"))
    print(f"   ORD -> SO match (the SO line just before, same stock, within 2 s): {int(m['po_ts'].is_not_null().sum()):,} of {m.height:,}; same px and qty on {int((m['same_px'] & m['same_qty']).sum()):,}")
    r = rej.select(c("idx").cast(pl.Int64)).unique().join(m, on="idx", how="left")
    print(f"   REJ idx with an ORD line: {int(r['t'].is_not_null().sum())}; with a matched SO line: {int(r['f3'].is_not_null().sum())}; side: {dict(r.group_by('side').len().iter_rows())};"
          f" by half hour: " + ", ".join(f"{int(b) // 2:02d}:{'30' if int(b) % 2 else '00'} {n}" for b, n in sorted(r.drop_nulls('t').group_by((((c('t') / 1e9 + 28800) % 86400) // 1800).cast(pl.Int64)).len().iter_rows())))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "rej_idx.txt").write_text("\n".join(str(x) for x in sorted(r["idx"].to_list())) + "\n")
    for kind in ("CXL", "CXL_RSP", "RPT", "SEND", "INS_RSP", "TRD"):
        t = trk.filter(c("kind") == kind)
        if t.is_empty() or "idx" not in t.columns:
            continue
        n = t.select(c("idx").cast(pl.Int64, strict=False)).join(r.select("idx"), on="idx", how="inner")
        print(f"   chrono trk {kind}: {t.height:,} rows in all; for the rejected idx: {n.height} rows on {n['idx'].n_unique()} orders")
    if life.is_empty():
        print("   no life table")
    else:
        print(f"-- PM order-life lines: {life.height:,}; events {dict(sorted(life.group_by('ev').len().iter_rows(), key=lambda kv: -kv[1]))}; columns {life.columns}")
        ex = r.drop_nulls("f3").select(c("f3").cast(pl.Int64).alias("exoid"), "idx", c("t").alias("ord_t"))
        for key in ("f3", "f4"):
            exk = r.drop_nulls(key).select(c(key).cast(pl.Int64).alias("exoid"), "idx", c("t").alias("ord_t"))
            lj = life.with_columns(c("exoid").cast(pl.Int64)).join(exk, on="exoid", how="inner")
            print(f"   keyed by the SO line's {key}: life lines for rejected orders {lj.height} on {lj['idx'].n_unique() if lj.height else 0} orders; events {dict(lj.group_by('ev').len().iter_rows()) if lj.height else {}}")
            cx = lj.filter(c("ev") == "cxl_req").sort("idx", "ts")
            if cx.height:
                per = cx.group_by("idx").agg(pl.len().alias("n"), ((c("ts") - c("ord_t").first()) / 1e9).min().alias("first_after"), (c("ts").diff() / 1e9).drop_nulls().alias("gaps"))
                gaps = np.array([g for gs in per["gaps"].to_list() for g in gs])
                print(f"      cancel requests per rejected order: orders {per.height}, attempts min / median / max {per['n'].min()} / {per['n'].median():.0f} / {per['n'].max()};"
                      f" first attempt seconds after the ORD q10/50/90 {' / '.join(f'{x:.2f}' for x in np.percentile(per['first_after'].drop_nulls().to_numpy(), [10, 50, 90]))}")
                if gaps.size:
                    print(f"      gaps between attempts, s: q10/50/90/max {' / '.join(f'{x:.2f}' for x in np.percentile(gaps, [10, 50, 90, 100]))}; rounded counts {dict(sorted(zip(*np.unique(np.round(gaps), return_counts=True), strict=True), key=lambda kv: -kv[1])[:8])}")
                one = cx.filter(c("idx") == per.sort("n", descending=True)["idx"][0])
                print(f"      the order with most attempts (idx {one['idx'][0]}): seconds after its ORD: {[round((t - one['ord_t'][0]) / 1e9, 1) for t in one['ts'].to_list()[:25]]}")
        allc = life.filter(c("ev") == "cxl_req")
        print(f"   all cancel requests in the day: {allc.height:,} on {allc['exoid'].n_unique():,} exOid; requests per exOid q50 / q90 / max {allc.group_by('exoid').len()['len'].median():.0f} / {allc.group_by('exoid').len()['len'].quantile(0.9):.0f} / {allc.group_by('exoid').len()['len'].max()}")
    print("[done]")


if __name__ == "__main__":
    main()
