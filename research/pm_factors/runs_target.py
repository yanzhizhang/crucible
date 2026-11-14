r"""runS and the daily target: does PM steer each stock from yesterday's holding toward the day's base position?

The user's description (2026-11-15): every day PM takes the T-1 position and the base position computed for the index
(``base_position_*.csv``, rolled daily) as the target, and recalibrates toward it. chrono's position log carries both per
stock (``yesterday_qty``, ``base_qty``, ``qty`` every 30 s), pm_orders' decisions carry runS's own numbers (pos, pp,
executp, asq, lthr, hs, fa, winsdQty, cpos). Read-only; prints, per day:

A  the start-of-day gap  gap0 = base_qty - yesterday_qty  (shares; how many stocks, how large);
B  the day's move: net trade = last qty - yesterday_qty against gap0 (share of the gap closed, least-squares slope, sign
   agreement), and what is left at the close;
C  day to day (with the previous date): yesterday_qty == the previous day's last qty, how much base_qty changed, and the
   new gap0 against the previous close's gap plus the base change -- the daily recalibration;
D  runS's units: pos / cpos / executp at each stock's first decision against yesterday_qty, base_qty and the gap;
E  intraday: executp and trade = executp - cpos against the prediction term fa x winsdQty with and without the live gap
   base_qty - qty (least squares, R^2); the thresholds on each side: decisions and orders made by sign(trade), fa against
   lthr / -lthr / hs on the rows that made an order, and whether lthr leans with the gap.

    python research/pm_factors/runs_target.py --dates 20261113,20261114
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pm_poslog import LOG, corr, read_log  # noqa: E402


def q(x: np.ndarray) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:.4g}" for v in np.percentile(x, [1, 10, 50, 90, 99])) if x.size else "-"


def fit(X: np.ndarray, y: np.ndarray, names: list[str]) -> str:
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < 50 or y[ok].std() == 0:
        return "too few rows"
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    r2 = 1 - np.sum((y[ok] - X[ok] @ b) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
    return f"R^2 {r2:.3f} over {int(ok.sum()):,} rows: " + ", ".join(f"{k} {v:+.4g}" for k, v in zip(names, b, strict=True))


def load_log(poslog: Path, d: str) -> pl.DataFrame | None:
    files = sorted(poslog.glob(f"{d}/position_log/*.csv"))
    if not files:
        return None
    lg = pl.concat([read_log(f) for f in files], how="diagonal_relaxed")
    return lg.with_columns(pl.col("symbol").cast(pl.String).str.slice(0, 6).cast(pl.Int64).alias("sid"), pl.col("ts").cast(pl.Int64),
                           *[pl.col(c).cast(pl.Float64) for c in LOG if c in lg.columns]).sort("sid", "ts")


def per_stock(lg: pl.DataFrame) -> pl.DataFrame:
    """One row per stock: yesterday_qty and base_qty of the first dump, qty of the last one."""
    return lg.group_by("sid").agg(pl.col("yesterday_qty").first().alias("y"), pl.col("base_qty").first().alias("base"),
                                  pl.col("base_qty").n_unique().alias("n_base"), pl.col("qty").first().alias("q0"), pl.col("qty").last().alias("q1"),
                                  pl.col("strategy").n_unique().alias("n_strat") if "strategy" in lg.columns else pl.lit(1).alias("n_strat")).sort("sid")


def day(poslog: Path, orders: Path, d: str, prev: pl.DataFrame | None) -> pl.DataFrame | None:
    lg = load_log(poslog, d)
    if lg is None:
        print(f"\n==== {d}: no position log under {poslog}/{d}")
        return None
    if "strategy" in lg.columns and lg["strategy"].n_unique() > 1:
        top = lg.group_by("strategy").len().sort("len", descending=True)
        print(f"\n==== {d}: strategies in the position log {dict(top.iter_rows())}; using {top['strategy'][0]}")
        lg = lg.filter(pl.col("strategy") == top["strategy"][0])
    ps = per_stock(lg)
    y, b, q0, q1 = (ps[c].to_numpy() for c in ("y", "base", "q0", "q1"))
    gap0, net, gap1 = b - y, q1 - y, b - q1
    print(f"\n==== {d}: {ps.height} stocks, {lg['ts'].n_unique()} dumps; base_qty constant within the day for {np.mean(ps['n_base'].to_numpy() == 1):.3f} of stocks")
    print(f"A  gap0 = base - yesterday: != 0 for {np.mean(gap0 != 0):.3f} of stocks; shares p1/10/50/90/99 {q(gap0)}; sum |gap0| {np.abs(gap0).sum():,.0f} "
          f"against sum base {b.sum():,.0f} and sum yesterday {y.sum():,.0f}; first-dump qty == yesterday {np.mean(q0 == y):.3f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        nz = gap0 != 0
        print(f"B  net trade = last qty - yesterday: != 0 for {np.mean(net != 0):.3f}; sum |net| {np.abs(net).sum():,.0f}; sign(net) == sign(gap0) on stocks "
              f"that traded and had a gap {np.mean(np.sign(net[nz & (net != 0)]) == np.sign(gap0[nz & (net != 0)])) if (nz & (net != 0)).any() else float('nan'):.3f}")
        print(f"   net on [gap0, 1]: {fit(np.column_stack([gap0, np.ones_like(gap0)]), net, ['gap0', 'one'])}")
        print(f"   gap left at the close: sum |base - last qty| {np.abs(gap1).sum():,.0f} = {np.abs(gap1).sum() / max(np.abs(gap0).sum(), 1):.3f} of sum |gap0|; "
              f"net / gap0 per stock p1/10/50/90/99 {q((net / gap0)[nz])}")
    if prev is not None:
        j = ps.join(prev, on="sid", how="inner", suffix="_p")
        yy, bb, qp, bp = (j[c].to_numpy() for c in ("y", "base", "q1_p", "base_p"))
        with np.errstate(divide="ignore", invalid="ignore"):
            print(f"C  against the previous day ({j.height} stocks in both): yesterday_qty == previous last qty {np.mean(yy == qp):.3f} (within 100: "
                  f"{np.mean(np.abs(yy - qp) <= 100):.3f}); base changed for {np.mean(bb != bp):.3f}, base / previous base p1/10/50/90/99 {q((bb / bp)[bp != 0])}")
            print(f"   new gap0 on [previous close gap, base change, 1]: {fit(np.column_stack([bp - qp, bb - bp, np.ones(j.height)]), bb - yy, ['prev_gap', 'd_base', 'one'])}")
            print(f"   sum |previous close gap| {np.abs(bp - qp).sum():,.0f} -> sum |new gap0| {np.abs(bb - yy).sum():,.0f}; sum |base change| {np.abs(bb - bp).sum():,.0f}")
    f = orders / f"date={d}" / "decisions.parquet"
    if not f.exists():
        print(f"   no {f}")
        return ps
    dec = pl.read_parquet(f).with_columns(pl.col("sid").cast(pl.Int64), pl.col("src_epoch").cast(pl.Int64)).drop_nulls("src_epoch").sort("sid", "src_epoch")
    print(f"   {dec.height:,} decisions, columns {dec.columns}")
    cp = "ev_cpos" if "ev_cpos" in dec.columns else None
    first = dec.group_by("sid", maintain_order=True).first().join(ps, on="sid", how="inner")
    fy, fb = first["y"].to_numpy(), first["base"].to_numpy()
    print(f"D  first decision of each stock ({first.height} stocks), exact / within 100 / corr:")
    for name in [c for c in ("pos", cp, "executp", "asq") if c]:
        v = first[name].to_numpy().astype(float)
        res = [f"{k} {np.mean(v == x):.3f}/{np.mean(np.abs(v - x) <= 100):.3f}/r {corr(v, x):+.3f}"
               for k, x in (("yesterday", fy), ("base", fb), ("yesterday - base", fy - fb), ("base - yesterday", fb - fy))]
        print(f"   {name:8s} " + "; ".join(res) + f"; values {q(v)}")
        print(f"   {name:8s} on [yesterday, base, 1]: {fit(np.column_stack([fy, fb, np.ones(first.height)]), v, ['yesterday', 'base', 'one'])}")
    j = dec.join_asof(lg.select("sid", "ts", *[c for c in LOG if c in lg.columns]), left_on="src_epoch", right_on="ts", by="sid", strategy="backward").drop_nulls("qty")
    if j.is_empty():
        print("E  no decision has a position-log row before it")
        return ps
    g = lambda c: j[c].to_numpy().astype(float)  # noqa: E731
    ex, fa, w, gap = g("executp"), g("fa"), g("winsdQty") if "winsdQty" in j.columns else np.ones(j.height), g("base_qty") - g("qty")
    cpos = g(cp) if cp else g("pos") + (g("pp") if "pp" in j.columns else 0)
    trade, one = ex - cpos, np.ones(j.height)
    minute = ((g("src_epoch") / 1e9 + 28800) % 86400) // 60
    print(f"E  {j.height:,} decisions with the holding known; live gap = base - qty != 0 on {np.mean(gap != 0):.3f}; corr(pos, qty - base) {corr(g('pos'), -gap):+.3f}, "
          f"corr(pos, qty) {corr(g('pos'), g('qty')):+.3f}, pos == qty - base {np.mean(g('pos') == -gap):.3f}, pos == qty {np.mean(g('pos') == g('qty')):.3f}")
    for name, m in (("09:30-09:31", minute < 571), ("later", minute >= 571)):
        if m.sum() < 50:
            continue
        print(f"   {name} ({int(m.sum()):,} rows): executp {q(ex[m])}; trade {q(trade[m])}")
        for yname, yv in (("executp", ex), ("trade", trade)):
            print(f"     {yname} on [fa x w, 1]:        {fit(np.column_stack([fa * w, one])[m], yv[m], ['fa_w', 'one'])}")
            print(f"     {yname} on [fa x w, gap, 1]:   {fit(np.column_stack([fa * w, gap, one])[m], yv[m], ['fa_w', 'gap', 'one'])}")
            print(f"     {yname} on [fa x w, gap, cpos, 1]: {fit(np.column_stack([fa * w, gap, cpos, one])[m], yv[m], ['fa_w', 'gap', 'cpos', 'one'])}")
    made = (g("nullptr") == 0) if "nullptr" in j.columns else np.zeros(j.height, bool)
    lthr, hs = (g(c) if c in j.columns else np.full(j.height, np.nan) for c in ("lthr", "hs"))
    print("   by side of trade = executp - cpos (rows, order made, then on the rows that made an order):")
    for name, m in (("sell (trade < 0)", trade < 0), ("buy  (trade > 0)", trade > 0), ("none (trade = 0)", trade == 0)):
        mm = m & made
        tail = (f"fa {q(fa[mm])}; fa < lthr {np.mean(fa[mm] < lthr[mm]):.3f}, fa > -lthr {np.mean(fa[mm] > -lthr[mm]):.3f}, |fa| > hs {np.mean(np.abs(fa[mm]) > hs[mm]):.3f}; "
                f"sign(gap) -/0/+ {np.mean(gap[mm] < 0):.2f}/{np.mean(gap[mm] == 0):.2f}/{np.mean(gap[mm] > 0):.2f}; |trade| {q(np.abs(trade[mm]))}") if mm.any() else ""
        print(f"     {name}: {int(m.sum()):,} rows, made {int(mm.sum()):,} ({mm.sum() / max(m.sum(), 1):.4f}); {tail}")
    with np.errstate(divide="ignore", invalid="ignore"):
        r = lthr / hs
        print(f"   lthr / hs {q(r)}; corr(lthr / hs, gap / winsdQty) {corr(r, gap / w):+.3f}; corr(lthr, gap) {corr(lthr, gap):+.3f}; lthr / hs by sign(gap) -/0/+: "
              + " | ".join(q(r[m]) if m.any() else "-" for m in (gap < 0, gap == 0, gap > 0)))
    return ps


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poslog", type=Path, default=Path("/work/crucible_data/chrono_poslog"))
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--dates", required=True, help="comma list in time order: each day is also compared with the one before it")
    a = ap.parse_args()
    prev = None
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        prev = day(a.poslog, a.orders, d, prev)


if __name__ == "__main__":
    main()
