r"""Follow-ups on the decoded alg.cpp against the live log (x = hs, P = cpos; R180b: the band gives executp exactly on 93%).

1  where the decision line is printed. Claim: a snapshot gets a decision line only if vc > 0 or executp != pos or the
   time is past rvts (13:00). Checked two ways: snapshot lines WITHOUT a decision by time of day (should be ~none after
   13:00 and before 13:00 be the in-band ones), and decisions before 13:00 with executp == cpos (should be ~none).
2  the 6.5% of decisions where executp is not the band target: by side of the step (the decoded cFtp only cuts or scales
   BUYS; GTp caps the step by the sellable quantity), the ratio executp step / band step, whether the logged step equals
   -asq (a sell capped by the sellable quantity) or 0 (a buy cut), and by time of day.
3  the false positives of the order rule (|fa| asf >= cr + hs + lthr, lot >= 100, sign): what they look like next to the
   true orders -- seconds since the stock's previous order, the wanted lots, pp against pos (a pending leg), the margin,
   time of day -- to see what the log lacks (top-of-book size, a pending leg opposing the alpha).
Read-only.

    python research/pm_factors/runs_alg2.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

CR, ASF = 0.0004, 0.8


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def hhmm(b: float, step: int = 1800) -> str:
    s = int(b) * step
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    sod_e = ((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod")
    sn = pl.read_parquet(root / "snaps.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).with_columns(sod_e)
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"]).sort("src_epoch")
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "ev_mp", "ev_cpos", "ev_poscst", "ev_vpen", "ev_vr") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num).with_columns((c("nullptr") == 0).alias("order"), sod_e)

    print(f"== 1 {a.date}: where the decision line is printed ({sn.height:,} snapshot lines, {d.height:,} decisions)")
    j = sn.join(d.select("sid", "src_epoch", pl.lit(True).alias("dec")), on=["sid", "src_epoch"], how="left").with_columns(c("dec").fill_null(False))
    for b, n, nd in sorted(j.group_by((c("sod") // 1800).cast(pl.Int64).alias("b")).agg(pl.len().alias("n"), (~c("dec")).sum().alias("nd")).iter_rows()):
        print(f"   {hhmm(b)}  snapshot lines {n:>8,}  without a decision {nd:>8,} ({nd / n:.4f})")
    per = j.group_by("sid").agg((~c("dec")).mean().alias("f"))["f"].to_numpy()
    print(f"   share without a decision per stock q1/10/50/90/99: {q(per)}")
    dup_s = sn.group_by("sid", "src_epoch").len()
    dup_d = d.group_by("sid", "src_epoch").len()
    print(f"   snapshot lines per (stock, source second): 1 on {float((dup_s['len'] == 1).mean()):.4f}, 2 on {float((dup_s['len'] == 2).mean()):.4f}, more on {float((dup_s['len'] > 2).mean()):.4f};"
          f" decision lines per key: 1 on {float((dup_d['len'] == 1).mean()):.4f}, 2 on {float((dup_d['len'] == 2).mean()):.4f}; keys with more snapshot lines than decisions: "
          f"{dup_s.join(dup_d, on=['sid', 'src_epoch'], how='left', suffix='_d').filter(pl.col('len') > pl.col('len_d').fill_null(0)).height:,}")
    j2 = j.sort("sid", "src_epoch").with_columns(c("dec").shift(1).over("sid").alias("prev_dec"), ((c("src_epoch") - c("src_epoch").shift(1).over("sid")) / 1e9).alias("gap"),
                                               (c("fa") == c("fa").shift(1).over("sid")).alias("same_fa"), c("fa").cast(pl.Float64).abs().alias("afa"),
                                               (c("ems_our_vol").cast(pl.Float64) / c("ems_mkt_vol").cast(pl.Float64)).alias("ratio"))
    for name, t in (("with a decision", j2.filter(c("dec"))), ("without", j2.filter(~c("dec")))):
        print(f"   lines {name}: {t.height:,}; previous line of the stock had a decision on {float(t['prev_dec'].mean() or 0):.4f}; seconds since the previous line q10/50/90 {q(t['gap'].drop_nulls().to_numpy(), (10, 50, 90))};"
              f" fa equal to the previous line's on {float(t['same_fa'].mean() or 0):.4f}; |fa| q10/50/90 {q(t['afa'].to_numpy(), (10, 50, 90))}; mktVol == 0 on {float((t['ems_mkt_vol'].cast(pl.Float64) == 0).mean()):.4f}; ourVol / mktVol >= 0.15 on {float((t['ratio'] >= 0.15).mean() or 0):.5f}")
    runs = j2.with_columns((c("dec") != c("prev_dec")).fill_null(True).cum_sum().over("sid").alias("run")).group_by("sid", "run").agg(c("dec").first(), pl.len().alias("n"))
    for flag in (True, False):
        rr = runs.filter(c("dec") == flag)["n"].to_numpy()
        print(f"   runs of consecutive lines {'with' if flag else 'without'} a decision per stock: {rr.size:,} runs, length q10/50/90/max {q(rr, (10, 50, 90, 100))}")
    am = d.filter(c("sod") < 46800)
    pm_ = d.filter(c("sod") >= 46800)
    eq = lambda t: float(((t["executp"] - t["ev_cpos"]).abs() <= 1e-6).mean()) if t.height else float("nan")  # noqa: E731
    print(f"   decisions before 13:00: {am.height:,}, executp == cpos on {eq(am):.5f}; after 13:00: {pm_.height:,}, executp == cpos on {eq(pm_):.5f}")

    g = lambda k: d[k].to_numpy().astype(float)  # noqa: E731
    fa, hs, lt, ex, pos, cp, mp, vp, vr, pc, wq, asq, sod, pp = (g(k) for k in ("fa", "hs", "lthr", "executp", "pos", "ev_cpos", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "winsdQty", "asq", "sod", "pp"))
    order = d["order"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 2 * vp * (vr / 3) * mp
        lo, hi = (fa - (hs + CR) / 1.5) / den - cp, (fa + (hs + CR) / 1.5) / den - cp
    r = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
    tgt = np.clip(cp + r, -pc, pc)
    step_b, step_l = tgt - cp, ex - cp
    ok = np.isfinite(tgt)
    exact = ok & (np.abs(ex - tgt) <= 1e-6 * np.maximum(np.abs(tgt), 1.0))
    miss = ok & ~exact
    print(f"\n== 2 executp against the band target: exact on {float(exact[ok].mean()):.4f}; {int(miss.sum()):,} misses")
    for name, m in (("band step > 0 (buy)", miss & (step_b > 0)), ("band step < 0 (sell)", miss & (step_b < 0)), ("band step == 0", miss & (step_b == 0))):
        if not m.any():
            print(f"   {name}: 0")
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = step_l[m] / step_b[m]
        tot = (ok & ((step_b > 0) if "buy" in name else (step_b < 0) if "sell" in name else (step_b == 0))).sum()
        print(f"   {name}: {int(m.sum()):,} misses of {int(tot):,} ({m.sum() / max(tot, 1):.4f}); logged step / band step {q(ratio)}; logged step == 0 on {float((step_l[m] == 0).mean()):.4f};"
              f" |logged step| == asq on {float((np.abs(np.abs(step_l[m]) - asq[m]) <= 1e-6).mean()):.4f}; logged executp == 0 on {float((ex[m] == 0).mean()):.4f}; |logged step| < |band step| on {float((np.abs(step_l[m]) < np.abs(step_b[m])).mean()):.4f}")
        print(f"      by half hour: " + ", ".join(f"{hhmm(b)} {int(n)}" for b, n in sorted(zip(*np.unique(sod[m] // 1800, return_counts=True), strict=True))))
        print(f"      asq q1/10/50/90/99 {q(asq[m])}; cpos {q(cp[m])}; band step {q(step_b[m])}; logged step {q(step_l[m])}")

    print("\n== 3 the order rule's false positives")
    tr = ex - cp
    lot = np.floor(np.minimum(np.abs(tr), wq) / 100) * 100
    rule = (np.abs(fa) * ASF >= CR + hs + lt) & (lot >= 100) & (np.sign(tr) == np.sign(fa))
    o = d.filter(c("order")).select("sid", c("src_epoch").alias("prev_order"))
    since = d.join_asof(o, left_on="src_epoch", right_on="prev_order", by="sid", strategy="backward", allow_exact_matches=False)
    since = ((since["src_epoch"] - since["prev_order"]) / 1e9).fill_null(1e9).to_numpy()
    tp, fp = rule & order, rule & ~order
    print(f"   rule predicts {int(rule.sum()):,}: {int(tp.sum()):,} orders, {int(fp.sum()):,} false positives; orders missed {int((order & ~rule).sum())}")
    margin = (np.abs(fa) * ASF - CR - hs - lt) / hs
    for name, m in (("orders", tp), ("false positives", fp)):
        if not m.any():
            continue
        print(f"   {name}: seconds since the stock's previous order q10/50/90 {q(np.where(since[m] > 1e8, np.nan, since[m]), (10, 50, 90))} (none before on {float((since[m] > 1e8).mean()):.3f}); lots {q(lot[m], (10, 50, 90))};"
              f" margin / hs {q(margin[m], (10, 50, 90))}; pp == pos on {float((pp[m] == pos[m]).mean()):.3f}; pos == cpos on {float((pos[m] == cp[m]).mean()):.3f}; buy share {float((tr[m] > 0).mean()):.3f}; asq < lot on {float((asq[m] < lot[m]).mean()):.3f}")
        print(f"      by half hour: " + ", ".join(f"{hhmm(b)} {int(n)}" for b, n in sorted(zip(*np.unique(sod[m] // 1800, return_counts=True), strict=True))))
    for name, m2 in (("within 1 s of the previous order", since < 1), ("1-3 s", (since >= 1) & (since < 3)), ("3-6 s", (since >= 3) & (since < 6)), ("6-60 s", (since >= 6) & (since < 60)), (">= 60 s or none", since >= 60)):
        mm = rule & m2
        if mm.any():
            print(f"   rule true, {name}: {int(mm.sum()):,}, order on {float(order[mm].mean()):.4f}")
    for name, m2 in (("pp == pos", pp == pos), ("pp != pos", pp != pos), ("pos == cpos", pos == cp), ("pos != cpos", pos != cp)):
        mm = rule & m2
        if mm.any():
            print(f"   rule true, {name}: {int(mm.sum()):,}, order on {float(order[mm].mean()):.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
