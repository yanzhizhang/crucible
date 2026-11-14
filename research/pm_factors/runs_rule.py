r"""runS on 20261113, third pass: executp and lthr as formulas, and the condition that still holds orders back.

R73: after 09:31 log|executp| follows log|fa / vpen| (corr 0.85) and log|lthr x winsdQty| (-0.89); of the decisions with
|trade| >= 100 and the side's threshold met only 2.6% placed an order. Read-only on pm_orders' parquet; prints:

1  log|executp| (after 09:31, executp != 0) by least squares on the logs of |fa|, vpen, winsdQty, -lthr, hs, vr, mp,
   poscst and asq + 1, then the same on the two or three strongest terms; the sign of executp against fa and cpos;
2  log(-lthr) (lthr < 0) the same way, and where lthr == 0 falls in the day;
3  the gate: each decision with |trade| >= 100 and the threshold met, against the stock's own orders -- is one of them
   live (sent, not yet cancel-confirmed or fully filled), how long since the stock's last order was sent / ended, and
   the order rate by those bins; plus the whole-account rate (orders sent in the second before) for a global pacer.

    python research/pm_factors/runs_rule.py --orders /work/crucible_data/pm_orders --date 20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def q(x, qs=(1, 10, 50, 90, 99)) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return " / ".join(f"{v:.4g}" for v in np.percentile(x, qs)) if x.size else "-"


def lsq(y: np.ndarray, terms: dict[str, np.ndarray]) -> tuple[float, dict[str, float]]:
    X = np.column_stack([*terms.values(), np.ones(y.size)])
    ok = np.isfinite(X).all(1) & np.isfinite(y)
    if ok.sum() < 100:
        return float("nan"), {}
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    r = y[ok] - X[ok] @ b
    return 1 - np.sum(r**2) / np.sum((y[ok] - y[ok].mean()) ** 2), dict(zip([*terms, "one"], b, strict=True))


def show(name: str, y: np.ndarray, terms: dict[str, np.ndarray]) -> None:
    r2, b = lsq(y, terms)
    print(f"  {name}: R^2 {r2:.4f}; " + ", ".join(f"{k} {v:+.4g}" for k, v in b.items()))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"), (((c("src_epoch") / 1e9 + 28800) % 86400) // 60).alias("minute")).sort("src_epoch")
    g = lambda df, k: df[k].to_numpy().astype(float)  # noqa: E731
    L = lambda x: np.log(np.abs(x))  # noqa: E731

    late = d.filter(c("minute") >= 571)
    ex, fa, vp, w, lt, hs, vr, mp, pc, asq, cp = (g(late, k) for k in ("executp", "fa", "ev_vpen", "winsdQty", "lthr", "hs", "ev_vr", "ev_mp", "ev_poscst", "asq", "ev_cpos"))
    print(f"==== {a.date}: {d.height:,} decisions, {late.height:,} after 09:31")
    print("-- 1 log|executp| on executp != 0")
    m = (ex != 0) & (vp > 0) & (lt < 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        full = {"log|fa|": L(fa), "log vpen": np.log(vp), "log w": np.log(w), "log -lthr": np.log(-lt), "log hs": np.log(hs), "log vr": np.log(vr),
                "log mp": np.log(mp), "log poscst": np.log(pc), "log(asq+1)": np.log(asq + 1)}
        y = L(ex)
        show("all terms", y[m], {k: v[m] for k, v in full.items()})
        for keep in (["log|fa|", "log vpen"], ["log|fa|", "log vpen", "log w"], ["log|fa|", "log -lthr"], ["log|fa|", "log vpen", "log -lthr"],
                     ["log|fa|", "log vpen", "log hs"], ["log|fa|", "log vpen", "log vr"]):
            show(" + ".join(keep), y[m], {k: full[k][m] for k in keep})
        nz = ex != 0
        print(f"  sign(executp) == sign(fa) {np.mean(np.sign(ex[nz]) == np.sign(fa[nz])):.4f}; executp == 0 with lthr == 0 {np.mean(ex[lt == 0] == 0):.4f}, with lthr < 0 {np.mean(ex[lt < 0] == 0):.4f}")
        print(f"  |executp| / (|fa| / vpen) on executp != 0: {q(np.abs(ex[m]) / (np.abs(fa[m]) / vp[m]))}")
        print("-- 2 log(-lthr) on lthr < 0")
        k = lt < 0
        ly = np.log(-lt)
        full2 = {"log hs": np.log(hs), "log vr": np.log(vr), "log vpen": np.log(np.where(vp > 0, vp, np.nan)), "log w": np.log(w), "log mp": np.log(mp),
                 "log poscst": np.log(pc), "log(asq+1)": np.log(asq + 1), "log(|cpos|+1)": np.log(np.abs(cp) + 1)}
        show("all terms", ly[k], {kk: v[k] for kk, v in full2.items()})
        for keep in (["log hs"], ["log hs", "log vr"], ["log hs", "log vpen"], ["log hs", "log vr", "log vpen"], ["log vpen"], ["log vpen", "log w"]):
            show(" + ".join(keep), ly[k], {kk: full2[kk][k] for kk in keep})
        mn = g(d, "minute")
        z = g(d, "lthr") == 0
        print(f"  lthr == 0 by minute of day (HH:MM, share): " + " ".join(f"{int(x) // 60:02d}:{int(x) % 60:02d} {z[mn == x].mean():.2f}" for x in np.unique(mn)[::20]))
        print(f"  lthr / hs {q(lt[k] / hs[k])}; -lthr - hs {q(-lt[k] - hs[k])}; -lthr / vpen {q(-lt[k] / vp[k])}")

    print("-- 3 the gate")
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("f3").alias("exoid")).sort("po_ts")
    life = pl.read_parquet(root / "life.parquet")
    end_c = life.filter(c("ev") == "OnECM").group_by("exoid").agg(c("ts").min().alias("t_cxl"))
    fills = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("fill_qty").sum().alias("fq"), c("ts").max().alias("t_fill"))
    o = po.join(end_c, on="exoid", how="left").join(fills, on="exoid", how="left")
    o = o.with_columns(pl.when(c("t_cxl").is_not_null()).then(c("t_cxl")).when(c("fq") >= c("qty")).then(c("t_fill")).otherwise(None).alias("t_end"))
    big = 2**62
    o = o.with_columns(c("t_end").fill_null(big))
    s = late.filter((c("trade").abs() >= 100) & pl.when(c("trade") < 0).then(c("fa") < c("lthr")).otherwise(c("fa") > -c("lthr")))
    made = s.with_columns((c("nullptr") == 0).alias("made")).select("sid", "src_epoch", "made", "trade")
    last = o.select("sid", c("po_ts").alias("last_po"), c("t_end").alias("last_end"))
    j = made.join_asof(last, left_on="src_epoch", right_on="last_po", by="sid", strategy="backward")
    j = j.with_columns(((c("last_end") > c("src_epoch")) & c("last_po").is_not_null()).alias("live"),
                       ((c("src_epoch") - c("last_po")) / 1e9).alias("since_po"),
                       pl.when(c("last_end") < big).then((c("src_epoch") - c("last_end")) / 1e9).otherwise(None).alias("since_end"))
    print(f"  {j.height:,} decisions with |trade| >= 100 and the threshold met; made {j['made'].mean():.4f}")
    for live, n, r in j.group_by("live").agg(pl.len().alias("n"), c("made").mean().alias("r")).sort("live").iter_rows():
        print(f"    stock's last order still live = {live}: {n:,} rows, made {r:.4f}")
    nl = j.filter(~c("live"))
    for col, cuts in (("since_po", [0.5, 1, 2, 5, 10, 30, 60, 300]), ("since_end", [0.1, 0.5, 1, 2, 5, 10, 30, 60, 300])):
        t = nl.drop_nulls(col).with_columns(c(col).cut(cuts).alias("bin")).group_by("bin").agg(pl.len().alias("n"), c("made").mean().alias("r")).sort("bin")
        print(f"    not live, by seconds {col} (no earlier order: {nl[col].null_count():,} rows, made {nl.filter(c(col).is_null())['made'].mean() if nl[col].null_count() else float('nan'):.4f}): "
              + " | ".join(f"{b} {r:.4f} ({n:,})" for b, n, r in t.iter_rows()))
    sent = po["po_ts"].to_numpy()
    st = s["src_epoch"].to_numpy()
    n1 = np.searchsorted(sent, st, side="left") - np.searchsorted(sent, st - 1_000_000_000, side="left")
    mk = (g(s, "nullptr") == 0)
    print("    by account orders sent in the second before: " + " | ".join(f"{lo}-{hi} {mk[(n1 >= lo) & (n1 <= hi)].mean() if ((n1 >= lo) & (n1 <= hi)).any() else float('nan'):.4f} ({int(((n1 >= lo) & (n1 <= hi)).sum()):,})"
                                                                for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 9), (10, 10**9))))


if __name__ == "__main__":
    main()
