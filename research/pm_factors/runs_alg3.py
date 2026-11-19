r"""The order rule's false positives on a live day: which unlogged-looking term is in fact in the log.

Rule so far (R180b): |fa| asf >= cr + hs + lthr, lot-rounded min(|executp - cpos|, winsdQty) >= 100, trade sign == sign(fa):
recall 0.998, precision 0.685; the false positives are 100-lot, mostly sells, almost all with pp != pos. Decoded cOrd:
(ve8, vf0) = GPpp(sid), pos = ve8 + vf0; no order if the pending leg vf0 opposes the alpha; sells clamped by the sellable
quantity; the quantity rounded down to 100 after min(.., room, top-of-book size). Here, on orders vs false positives:
  - the three logged positions (pos, pp, cpos): their differences and the sign of each difference against sign(fa);
  - asq (quantiles, share < 100, share < the wanted lots), winsdQty < 100;
  - the wanted quantity cut from each position: lots of |executp - cpos|, |executp - pos|, |executp - pp|;
  - then the rule with each extra term added, and the best combination: precision / recall.
Read-only.

    python research/pm_factors/runs_alg3.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

CR, ASF = 0.0004, 0.8


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    c = pl.col
    d = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"]).sort("src_epoch")
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "liid", "ev_mp", "ev_cpos", "ev_asq") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num)
    g = lambda k: d[k].to_numpy().astype(float) if k in d.columns else np.full(d.height, np.nan)  # noqa: E731
    fa, hs, lt, ex, pos, pp, cp, wq, asq, easq, liid = (g(k) for k in ("fa", "hs", "lthr", "executp", "pos", "pp", "ev_cpos", "winsdQty", "asq", "ev_asq", "liid"))
    order = (g("nullptr") == 0)
    sa = np.sign(fa)
    cond = np.abs(fa) * ASF >= CR + hs + lt
    lots = lambda x: np.floor(np.minimum(np.abs(x), wq) / 100) * 100  # noqa: E731
    tr = ex - cp
    rule = cond & (lots(tr) >= 100) & (np.sign(tr) == sa)
    tp, fp = rule & order, rule & ~order
    print(f"== {a.date}: rule predicts {int(rule.sum()):,}: {int(tp.sum()):,} orders, {int(fp.sum()):,} false positives; orders outside the rule {int((order & ~rule).sum())}")
    print(f"   asq (decision line) == ev_asq (input line) on {float(np.nanmean(asq == easq)):.4f}; liid distinct values {np.unique(liid[np.isfinite(liid)])[:8]}")
    for name, m in (("orders", tp), ("false positives", fp)):
        if not m.any():
            continue
        print(f"-- {name} ({int(m.sum()):,}); sell share {float((tr[m] < 0).mean()):.3f}")
        print(f"   pos {q(pos[m])} | pp {q(pp[m])} | cpos {q(cp[m])} | executp {q(ex[m])}")
        for nm, v in (("pp - pos", pp - pos), ("pp - cpos", pp - cp), ("pos - cpos", pos - cp), ("pos", pos), ("pp", pp)):
            x = v[m]
            print(f"   {nm:<10} == 0 on {float((x == 0).mean()):.3f}; sign vs sign(fa): same {float((np.sign(x) == sa[m]).mean()):.3f}, opposite {float((np.sign(x) == -sa[m]).mean()):.3f}; values {q(x)}")
        sl = m & (tr < 0)
        print(f"   sells: asq {q(asq[sl])}; asq < 100 on {float((asq[sl] < 100).mean()) if sl.any() else float('nan'):.4f}; asq == 0 on {float((asq[sl] == 0).mean()) if sl.any() else float('nan'):.4f};"
              f" asq < wanted lots on {float((asq[sl] < lots(tr)[sl]).mean()) if sl.any() else float('nan'):.4f}")
        print(f"   winsdQty {q(wq[m])}; winsdQty < 100 on {float((wq[m] < 100).mean()):.4f}")
        for nm, v in (("executp - cpos", ex - cp), ("executp - pos", ex - pos), ("executp - pp", ex - pp)):
            x = v[m]
            print(f"   |{nm}| {q(np.abs(x))}; lots >= 100 on {float((lots(v)[m] >= 100).mean()):.4f}; sign == sign(fa) on {float((np.sign(x) == sa[m]).mean()):.4f}")
    terms = {"sell needs asq >= 100": ~((tr < 0) & (asq < 100)), "sell needs asq >= lots": ~((tr < 0) & (asq < lots(tr))), "pp == pos": pp == pos,
             "pp - pos not opposite to fa": np.sign(pp - pos) != -sa, "pp - cpos not opposite to fa": np.sign(pp - cp) != -sa, "pos - cpos not opposite to fa": np.sign(pos - cp) != -sa,
             "pp not opposite to fa": np.sign(pp) != -sa, "lots(executp - pos) >= 100": lots(ex - pos) >= 100, "lots(executp - pp) >= 100": lots(ex - pp) >= 100,
             "sign(executp - pos) == sign(fa)": np.sign(ex - pos) == sa, "sign(executp - pp) == sign(fa)": np.sign(ex - pp) == sa, "winsdQty >= 100": wq >= 100,
             "executp != pos": np.abs(ex - pos) > 1e-6}
    print("-- the rule with one extra term (precision / recall of all orders):")
    res = []
    for nm, t in terms.items():
        r = rule & t
        res.append((float((r & order).sum()) / max(r.sum(), 1), float((r & order).sum()) / max(order.sum(), 1), nm))
    for p_, r_, nm in sorted(res, reverse=True):
        print(f"   {p_:.4f} / {r_:.4f}  + {nm}")
    best = []
    for k1, k2 in itertools.combinations(terms, 2):
        r = rule & terms[k1] & terms[k2]
        p_, r_ = float((r & order).sum()) / max(r.sum(), 1), float((r & order).sum()) / max(order.sum(), 1)
        best.append((2 * p_ * r_ / max(p_ + r_, 1e-12), p_, r_, f"+ {k1}  + {k2}"))
    print("-- best pairs of extra terms (F1, precision, recall):")
    for f1, p_, r_, nm in sorted(best, reverse=True)[:6]:
        print(f"   {f1:.4f}  {p_:.4f} / {r_:.4f}  {nm}")
    print("[done]")


if __name__ == "__main__":
    main()
