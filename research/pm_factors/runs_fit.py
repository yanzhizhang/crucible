r"""runS: is executp a TARGET position (relative to base) rather than a trade? Read-only on pm_orders' decisions.

From R33b's samples: cpos == pos + pp (pp = in flight), and on large rows executp - cpos == -asq exactly (300059, 600036,
601166), while ordinary rows have a tiny executp next to a sizeable cpos (600415: cpos 36, executp 0.34). So test:

* trade = executp - cpos: share == -asq, == +asq, |trade| <= asq, and its sign against fa;
* executp itself as the target: its scale against winsdQty, fa, hs, lthr, vpen, vr (ratios, quantiles, least squares on the rows
  whose trade is not capped by asq), by time of day (the 09:30 burst against later);
* the order gate: nullptr against |trade| (>= one lot of 100?), asq and lthr.

    python research/pm_factors/runs_fit.py --orders /work/crucible_data/pm_orders --date 20261112
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def q(x: np.ndarray) -> str:
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:.4g}" for v in np.percentile(x, [1, 10, 50, 90, 99])) if x.size else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    d = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet")
    c = pl.col
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"), (c("ev_cpos") - c("pos")).alias("cpos_m_pos"),
                       (((c("src_epoch") / 1e9 + 28800) % 86400) // 60).alias("minute"))
    n = d.height
    print(f"{n:,} decisions; cpos == pos + pp: {d.select(((c('ev_cpos') - c('pos') - c('pp')).abs() < 1e-6).mean()).item():.4f}")
    t = d["trade"].to_numpy()
    asq = d["asq"].to_numpy()
    fa = d["fa"].to_numpy()
    print(f"trade = executp - cpos: == -asq {np.mean(np.abs(t + asq) < 1e-6):.4f}, == +asq {np.mean(np.abs(t - asq) < 1e-6):.4f}, "
          f"|trade| <= asq {np.mean(np.abs(t) <= asq + 1e-6):.4f}, sign(trade) == sign(fa) {np.mean(np.sign(t) == np.sign(fa)):.4f}")
    print(f"  trade p1/10/50/90/99: {q(t)};  |executp| {q(np.abs(d['executp'].to_numpy()))}")
    for name, m in (("09:30-09:31", d["minute"] < 571), ("later", d["minute"] >= 571)):
        s = d.filter(m)
        if s.height:
            tt, aa = s["trade"].to_numpy(), s["asq"].to_numpy()
            print(f"  {name}: {s.height:,} rows; trade == -asq {np.mean(np.abs(tt + aa) < 1e-6):.3f}, == +asq {np.mean(np.abs(tt - aa) < 1e-6):.3f}; "
                  f"|executp| {q(np.abs(s['executp'].to_numpy()))}; lthr == 0 {np.mean(s['lthr'].to_numpy() == 0):.3f}")
    # executp as the target: its scale
    e = d["executp"].to_numpy()
    w = d["winsdQty"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        print(f"\nexecutp / (fa x winsdQty): {q(e / (fa * w))}")
        print(f"executp / ((|fa| - hs)+ x sign(fa) x winsdQty): {q(e / (np.sign(fa) * np.clip(np.abs(fa) - d['hs'].to_numpy(), 0, None) * w))}")
        print(f"executp / ((fa - lthr) x winsdQty): {q(e / ((fa - d['lthr'].to_numpy()) * w))}")
    free = d.filter((c("trade").abs() < c("asq") - 1e-6) & c("executp").is_finite())
    terms = {"fa_w": free["fa"] * free["winsdQty"], "fa_w_vpen": free["fa"] * free["winsdQty"] * free["ev_vpen"],
             "fa_w_vr": free["fa"] * free["winsdQty"] * free["ev_vr"], "cpos": free["ev_cpos"], "pos": free["pos"], "pp": free["pp"],
             "hs_w": free["hs"] * free["winsdQty"] * np.sign(free["fa"]), "lthr_w": free["lthr"] * free["winsdQty"], "one": pl.Series([1.0] * free.height)}
    X = np.column_stack([v.to_numpy().astype(float) for v in terms.values()])
    y = free["executp"].to_numpy().astype(float)
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() > 100:
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        r2 = 1 - np.sum((y[ok] - X[ok] @ b) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2)
        print(f"\nleast squares of executp on {list(terms)} over {ok.sum():,} uncapped rows: R^2 {r2:.3f}")
        print("  " + ", ".join(f"{k} {v:+.4g}" for k, v in zip(terms, b, strict=True)))
        for k in terms:
            if k == "one":
                continue
            x = terms[k].to_numpy().astype(float)
            m = ok & np.isfinite(x)
            if x[m].std() > 0:
                print(f"  corr(executp, {k}) {np.corrcoef(x[m], y[m])[0, 1]:+.3f}")
    # the order gate
    if "nullptr" in d.columns:
        g = d.with_columns(c("trade").abs().alias("at"))
        print("\norder made (nullptr 0) by |trade|:")
        for lo, hi in ((0, 1), (1, 100), (100, 200), (200, 1000), (1000, 1e9)):
            s = g.filter((c("at") >= lo) & (c("at") < hi))
            print(f"  [{lo}, {hi}): {s.height:,} rows, made {float((s['nullptr'] == 0).mean()) if s.height else float('nan'):.4f}")
        print("order made by asq == 0 / > 0, and by lthr == 0 / < 0:")
        for name, m in (("asq 0", c("asq") == 0), ("asq > 0", c("asq") > 0), ("lthr 0", c("lthr") == 0), ("lthr < 0", c("lthr") < 0)):
            s = g.filter(m)
            print(f"  {name}: {s.height:,} rows, made {float((s['nullptr'] == 0).mean()) if s.height else float('nan'):.4f}")


if __name__ == "__main__":
    main()
