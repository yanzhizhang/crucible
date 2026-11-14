r"""runS on a normal day, the second pass: the target executp, the order gate, and what happens to an order after it is sent.

R72 (20261113) settled the units: pos = qty - base_qty (the holding relative to the day's base position, 0.999),
cpos = pos + pp (pp = in flight), asq = yesterday_qty (T+1 sellable), and after 09:31 trade = executp - cpos follows
0.87 x (base - qty) + 0.07 x fa x winsdQty (R^2 0.87): executp is a small TARGET offset from the base and every trade pulls
the holding back toward base + executp. Orders go out only on |trade| >= 100 and on the matching threshold (sell: fa <
lthr 0.99, buy: fa > -lthr 0.85). Read-only on pm_orders' parquet (decisions, po with side, life); prints:

1  executp itself, after 09:31 and away from the caps: its value against fa x winsdQty, (fa - lthr) x winsdQty,
   (|fa| - hs)+ x winsdQty, x vr, x vpen, x asf, with and without rounding to lots; the rows where executp == 0 against
   |fa| < -lthr (a dead band?); the 09:30 burst on its own (lthr == 0: executp == +asq or == -asq - cpos?);
2  the gate: which of |trade| >= 100, the side's threshold, nullptr (= an order already live) explains order / no order
   (a decision tree of depth 3 written as shares), and the order qty against |trade| (whole trade, one lot, capped);
3  the order: price against the snapshot (mp, mp x (1 +- hs) = the touch?), per side;
4  its life: sent -> cancel confirmed (OnECM) / filled (OnEOM), per side: share cancelled / filled / both, time to the
   cancel and to the fill, fill qty against order qty, fill price against order price.

    python research/pm_factors/runs_exec.py --orders /work/crucible_data/pm_orders --poslog /work/crucible_data/chrono_poslog --date 20261113
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_poslog import read_log  # noqa: E402


def q(x, qs=(1, 10, 50, 90, 99)) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:.4g}" for v in np.percentile(x, qs)) if x.size else "-"


def share(m) -> str:
    m = np.asarray(m, bool)
    return f"{m.mean():.4f} ({int(m.sum()):,} of {m.size:,})" if m.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--poslog", type=Path, default=Path("/work/crucible_data/chrono_poslog"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"), (((c("src_epoch") / 1e9 + 28800) % 86400) // 60).alias("minute"))
    g = lambda df, k: df[k].to_numpy().astype(float)  # noqa: E731

    print(f"==== {a.date}: {d.height:,} decisions")
    print("-- 1 executp")
    for name, s in (("09:30", d.filter(c("minute") < 571)), ("later", d.filter(c("minute") >= 571))):
        ex, fa, w, lt, hs, vr, vp = (g(s, k) for k in ("executp", "fa", "winsdQty", "lthr", "hs", "ev_vr", "ev_vpen"))
        asq, cp = g(s, "asq"), g(s, "ev_cpos")
        nz = ex != 0
        print(f"  {name}: {s.height:,} rows; executp == 0 {share(~nz)}; integer {share(ex[nz] == np.round(ex[nz]))}; lot multiple {share(np.mod(ex[nz], 100) == 0)}")
        print(f"    executp == asq {share(ex == asq)}, == -asq {share(ex == -asq)}, == cpos {share(ex == cp)}, == -asq - cpos... trade == -asq {share(ex - cp == -asq)}")
        with np.errstate(divide="ignore", invalid="ignore"):
            band = np.abs(fa) < -lt
            print(f"    dead band |fa| < -lthr: executp == 0 inside {share(~nz[band])}, outside {share(~nz[~band])}; |fa| < hs: executp == 0 {share(~nz[np.abs(fa) < hs])}")
            for k, den in (("fa w", fa * w), ("(fa - lthr) w", (fa - lt) * w), ("(fa + lthr) w", (fa + lt) * w),
                           ("sign(fa)(|fa| - hs)+ w", np.sign(fa) * np.clip(np.abs(fa) - hs, 0, None) * w),
                           ("sign(fa)(|fa| + lthr)+ w", np.sign(fa) * np.clip(np.abs(fa) + lt, 0, None) * w),
                           ("fa w vr", fa * w * vr), ("fa w vpen", fa * w * vp), ("fa vr", fa * vr), ("fa / vpen", fa / vp)):
                r = ex[nz] / den[nz]
                fin = np.isfinite(r)
                print(f"    executp / {k:26s} on executp != 0: {q(r)}; spread p90/p10 {np.nanpercentile(r[fin], 90) / np.nanpercentile(r[fin], 10) if fin.sum() > 10 else float('nan'):.3g}")
        X = np.column_stack([fa * w, fa * w * vr, fa * vr, fa / np.where(vp > 0, vp, np.nan), lt * w, hs * w * np.sign(fa), cp, np.ones(s.height)])
        ok = np.isfinite(X).all(1) & np.isfinite(ex) & nz
        if ok.sum() > 50:
            b, *_ = np.linalg.lstsq(X[ok], ex[ok], rcond=None)
            r2 = 1 - np.sum((ex[ok] - X[ok] @ b) ** 2) / np.sum((ex[ok] - ex[ok].mean()) ** 2)
            names = ["fa_w", "fa_w_vr", "fa_vr", "fa/vpen", "lthr_w", "hs_w_sgn", "cpos", "one"]
            print(f"    least squares of executp on executp != 0 ({int(ok.sum()):,}): R^2 {r2:.3f}: " + ", ".join(f"{k} {v:+.4g}" for k, v in zip(names, b, strict=True)))
            for k, col in zip(names[:-1], X.T, strict=False):
                m = ok & np.isfinite(col)
                if col[m].std() > 0:
                    print(f"      corr(executp, {k}) {np.corrcoef(col[m], ex[m])[0, 1]:+.3f}; on log |.|: {np.corrcoef(np.log(np.abs(col[m]) + 1e-300), np.log(np.abs(ex[m])))[0, 1]:+.3f}")

    print("-- 2 the gate (after 09:31)")
    s = d.filter(c("minute") >= 571)
    tr, fa, lt, nul = g(s, "trade"), g(s, "fa"), g(s, "lthr"), g(s, "nullptr")
    made = nul == 0
    lot = np.abs(tr) >= 100
    side_ok = np.where(tr < 0, fa < lt, np.where(tr > 0, fa > -lt, False))
    for k1, m1 in (("|trade| >= 100", lot), ("|trade| < 100", ~lot)):
        for k2, m2 in (("side threshold met", side_ok), ("side threshold not met", ~side_ok)):
            m = m1 & m2
            print(f"  {k1:15s} & {k2:22s}: {int(m.sum()):>9,} rows, order made {share(made[m])}")
    m = lot & side_ok
    if "pp" in s.columns:
        pp = g(s, "pp")
        print(f"  among |trade| >= 100 & threshold met: pp != 0 (in flight) {share(pp[m] != 0)}; made with pp == 0 {share(made[m & (pp == 0)])}, with pp != 0 {share(made[m & (pp != 0)])}")
    print("  the rows that made an order and still are not explained: " + share(made & ~(lot & side_ok)))

    po_f = root / "po.parquet"
    if not po_f.exists():
        print(f"no {po_f}")
        return
    po = pl.read_parquet(po_f).with_columns(c("sid").cast(pl.Int64)).sort("po_ts")
    if "side" not in po.columns:
        print("po.parquet has no side column: rerun pm_orders (from 1113 it reads the place_order lines)")
        return
    dd = d.sort("src_epoch").select("sid", c("src_epoch").alias("dec_ts"), "fa", "executp", "ev_cpos", "trade", "asq", "lthr", "hs", "ev_mp", "winsdQty")
    j = po.join_asof(dd, left_on="po_ts", right_on="dec_ts", by="sid", strategy="backward")
    print(f"-- 3 the order: {po.height:,} sent, sides {dict(j.group_by('side').len().iter_rows())}; tied to a decision {share(j['dec_ts'].is_not_null().to_numpy())}")
    for sd in ("buy", "sell"):
        o = j.filter(c("side") == sd)
        if o.is_empty():
            continue
        px, mp, hs, qty, tr = g(o, "px"), g(o, "ev_mp"), g(o, "hs"), g(o, "qty"), g(o, "trade")
        touch = mp * (1 + hs) if sd == "buy" else mp * (1 - hs)
        far = mp * (1 - hs) if sd == "buy" else mp * (1 + hs)
        with np.errstate(divide="ignore", invalid="ignore"):
            print(f"  {sd}: n {o.height:,}; price at the far touch (buy at ask / sell at bid) {share(np.abs(px - touch) < 0.0011)}, at the near touch {share(np.abs(px - far) < 0.0011)}, "
                  f"at mp {share(np.abs(px - mp) < 0.0011)}; (px - mp) / (mp hs) {q((px - mp) / (mp * hs))}")
            print(f"     qty {q(qty)}; qty == |trade| {share(qty == np.abs(tr))}, qty == one lot {share(qty == 100)}, qty < |trade| {share(qty < np.abs(tr))}; qty / |trade| {q(qty / np.abs(tr))}")

    lf = root / "life.parquet"
    if not lf.exists():
        print(f"no {lf}")
        return
    life = pl.read_parquet(lf)
    print(f"-- 4 order life: events {dict(life.group_by('ev').len().iter_rows())}")
    po2 = po.with_columns(c("f3").alias("exoid"))
    first = lambda ev: life.filter(c("ev") == ev).group_by("exoid").agg(c("ts").min().alias(ev))  # noqa: E731
    fills = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("ts").min().alias("fill_ts"), c("fill_qty").sum().alias("fill_qty"),
                                                                  (c("fill_qty") * c("fill_px")).sum().alias("fill_val"))
    L = po2.join(first("OnECM"), on="exoid", how="left").join(first("cxl_req"), on="exoid", how="left").join(fills, on="exoid", how="left")
    for sd in ("buy", "sell"):
        o = L.filter(c("side") == sd)
        if o.is_empty():
            continue
        t0, cx, rq, ft = (o[k].to_numpy().astype(float) for k in ("po_ts", "OnECM", "cxl_req", "fill_ts"))
        fq, fv, qty, px = g(o, "fill_qty"), g(o, "fill_val"), g(o, "qty"), g(o, "px")
        cancelled, filled = np.isfinite(cx), np.isfinite(ft)
        print(f"  {sd}: n {o.height:,}; cancel confirmed {share(cancelled)}, any fill {share(filled)}, both {share(cancelled & filled)}, neither {share(~cancelled & ~filled)}")
        print(f"     sent -> cancel requested (s) {q((rq - t0) / 1e9)}; sent -> cancel confirmed (s) {q((cx - t0) / 1e9)}; sent -> first fill (s) {q((ft - t0) / 1e9)}")
        with np.errstate(divide="ignore", invalid="ignore"):
            print(f"     fill qty / order qty on filled {q(fq[filled] / qty[filled])}; fill vwap - order px {q(fv[filled] / fq[filled] - px[filled])}")
    if "poslog" in a and (a.poslog / a.date).exists():
        files = sorted(a.poslog.glob(f"{a.date}/position_log/*.csv"))
        if files:
            lg = pl.concat([read_log(f) for f in files], how="diagonal_relaxed")
            print(f"  position log: {lg['symbol'].n_unique()} stocks, the day's net traded qty sum |last - first| "
                  f"{lg.group_by('symbol').agg((c('qty').last() - c('qty').first()).abs()).select(c('qty').sum()).item():,.0f}")


if __name__ == "__main__":
    main()
