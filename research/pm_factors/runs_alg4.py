r"""The order rule's false positives by board, stock, price and time (20261118): is it the STAR 200-share minimum?

Decoded Alg::onInnerMsg drops a STAR (688 / 689) order of fewer than 200 shares; the 700 false positives are all
"sell the last 100 shares". Split them and the true orders by board (code prefix), with the smallest order quantity per
board among the true orders (from the wanted trade and, where the po table has it, the sent quantity); distinct stocks
and how the false positives repeat on a stock (every snapshot?); their price and time of day against the orders'.
Then the rule with "STAR needs 200 shares" added: precision / recall. Read-only.

    python research/pm_factors/runs_alg4.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

CR, ASF = 0.0004, 0.8


def board(sid: int) -> str:
    s = f"{sid:06d}"
    return "STAR 688/689" if s.startswith(("688", "689")) else "ChiNext 30" if s.startswith("30") else "SH main 60" if s.startswith("60") else "SZ main 00" if s.startswith("00") else "other"


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"]).sort("src_epoch")
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "ev_mp", "ev_cpos") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num).with_columns(((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod"), c("sid").map_elements(board, return_dtype=pl.String).alias("board"))
    g = lambda k: d[k].to_numpy().astype(float)  # noqa: E731
    fa, hs, lt, ex, cp, wq, mp, sod = (g(k) for k in ("fa", "hs", "lthr", "executp", "ev_cpos", "winsdQty", "ev_mp", "sod"))
    order = g("nullptr") == 0
    tr = ex - cp
    lot = np.floor(np.minimum(np.abs(tr), wq) / 100) * 100
    rule = (np.abs(fa) * ASF >= CR + hs + lt) & (lot >= 100) & (np.sign(tr) == np.sign(fa))
    d = d.with_columns(pl.Series("order", order), pl.Series("rule", rule), pl.Series("lot", lot), pl.Series("tr", tr))
    tp, fp = d.filter(c("rule") & c("order")), d.filter(c("rule") & ~c("order"))
    print(f"== {a.date}: {tp.height:,} orders inside the rule, {fp.height:,} false positives; stocks per board in the day's decisions: {dict(sorted(d.group_by('board').agg(c('sid').n_unique()).iter_rows()))}")
    for name, t in (("orders", tp), ("false positives", fp)):
        print(f"-- {name}: by board (rows, stocks, smallest / median wanted lots): " + "; ".join(
            f"{b}: {n} rows, {k} stocks, lots {lo:.0f} / {md:.0f}" for b, n, k, lo, md in sorted(t.group_by("board").agg(pl.len(), c("sid").n_unique(), c("lot").min().alias("lo"), c("lot").median().alias("md")).iter_rows())))
        print(f"   price q1/10/50/90/99 {q(t['ev_mp'].to_numpy())}; 100 shares in yuan {q(100 * t['ev_mp'].to_numpy())}; by half hour: "
              + ", ".join(f"{int(b) // 2:02d}:{'30' if int(b) % 2 else '00'} {n}" for b, n in sorted(t.group_by((c('sod') // 1800).cast(pl.Int64)).len().iter_rows())))
    per = fp.group_by("sid").agg(pl.len().alias("n"), c("sod").min().alias("first"), c("sod").max().alias("last"), c("board").first(), c("ev_mp").median().alias("mp")).sort("n", descending=True)
    print(f"-- false positives per stock: {per.height} stocks; rows per stock q10/50/90/max {q(per['n'].to_numpy(), (10, 50, 90, 100))}; the 12 with most:")
    for r in per.head(12).iter_rows(named=True):
        o_n = tp.filter(c("sid") == r["sid"]).height
        print(f"   {r['sid']:06d} {r['board']:<12} {r['n']:>4} rows  {int(r['first']) // 3600:02d}:{int(r['first']) % 3600 // 60:02d} - {int(r['last']) // 3600:02d}:{int(r['last']) % 3600 // 60:02d}  price {r['mp']:.2f}  true orders on that stock {o_n}")
    gaps = fp.sort("sid", "src_epoch").with_columns(((c("src_epoch") - c("src_epoch").shift(1).over("sid")) / 1e9).alias("gap"))["gap"].drop_nulls().to_numpy()
    print(f"   seconds between consecutive false positives of one stock q10/50/90 {q(gaps, (10, 50, 90))}; exactly 3 s on {float((np.abs(gaps - 3) < 0.5).mean()) if gaps.size else float('nan'):.3f}")
    star = d["board"].to_numpy() == "STAR 688/689"
    if (root / "po.parquet").exists():
        po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64).map_elements(board, return_dtype=pl.String).alias("board"))
        print("-- sent orders (po table) by board: rows, smallest / median qty: " + "; ".join(f"{b}: {n}, {lo:.0f} / {md:.0f}" for b, n, lo, md in sorted(po.group_by("board").agg(pl.len(), c("qty").min().alias("lo"), c("qty").median().alias("md")).iter_rows())))
    for name, r2 in (("+ STAR needs lots >= 200", rule & ~(star & (lot < 200))), ("+ no STAR orders at all", rule & ~star)):
        tp_ = float((r2 & order).sum())
        print(f"   rule {name}: predicts {int(r2.sum()):,}; precision {tp_ / max(r2.sum(), 1):.4f}, recall {tp_ / max(order.sum(), 1):.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
