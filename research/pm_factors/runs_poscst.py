r"""poscst (one value per stock per day in the runS decision log) against PM's start-of-day table: R151 found
poscst / sod.mb in [4.95, 5.00], never above 5 and not exact. What is the rest?

Per day, over the stocks (the day's own stats/sod file, as vr = sod.vr(D) exactly):
  - a sample of rows (poscst, mb, msr, vr, first mp, first cpos, asq) and the fine quantiles of poscst / mb;
  - the share where poscst == 5 x mb to 1e-9, and to float32 rounding;
  - the gap g = 5 x mb - poscst: its steadiest ratios against the table columns and their products with the config
    values, the first price, the lot value (100 x mp), the first position's value;
  - least squares of poscst on mb, msr, vr, their products and a constant;
  - poscst against other files of the table (the previous day's, the next day's) in case mb is from another day.
Read-only.

    python research/pm_factors/runs_poscst.py --orders /work/crucible_data/pm_orders --dates 20261112,20261113 --sts /work/crucible_data/pm_stsfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs_sod import CFG, steadiest  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--dates", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/pm_stsfile"))
    a = ap.parse_args()
    from ic_score import pm_daily
    c = pl.col
    for d in [x for x in a.dates.split(",") if x]:
        f = a.orders / f"date={d}" / "decisions.parquet"
        if not f.exists():
            continue
        t = pl.read_parquet(f)
        if "ev_poscst" not in t.columns:
            continue
        t = t.with_columns(c("sid").cast(pl.Int64), *[c(k).cast(pl.Float64) for k in ("ev_poscst", "ev_mp", "ev_cpos", "ev_asq", "ev_vr") if k in t.columns]).sort("src_epoch")
        per = t.group_by("sid").agg(c("ev_poscst").first().alias("poscst"), c("ev_poscst").n_unique().alias("n_pc"), c("ev_poscst").min().alias("pc_min"),
                                    c("ev_poscst").max().alias("pc_max"), c("ev_mp").first().alias("mp0"), c("ev_cpos").first().alias("cpos0"),
                                    c("ev_asq").first().alias("asq0"), c("ev_asq").max().alias("asq_max"), c("ev_cpos").abs().max().alias("cpos_max")).sort("sid")
        print(f"\n== {d}: {per.height} stocks; poscst single-valued on {float((per['n_pc'] == 1).mean()):.3f}; where not, max/min q50/q99 "
              f"{' / '.join(f'{v:.6g}' for v in np.percentile((per['pc_max'] / per['pc_min']).drop_nulls().to_numpy(), [50, 99]))}")
        for rule, pick in (("on", None), ("before", None)):
            pd_ = pm_daily(a.sts, d, per["sid"].to_list(), rule, pick)
            if pd_ is None or "pm.mb" not in pd_.columns:
                print(f"   rule {rule}: no table")
                continue
            j = per.join(pd_, on="sid", how="left")
            g = lambda k: j[k].to_numpy().astype(float)  # noqa: E731
            pc, mb = g("poscst"), g("pm.mb")
            r = pc / mb
            ok = np.isfinite(r)
            print(f"   rule {rule}: poscst / mb quantiles 0/1/5/25/50/75/95/99/100: {' / '.join(f'{v:.6f}' for v in np.percentile(r[ok], [0, 1, 5, 25, 50, 75, 95, 99, 100]))}")
            print(f"      == 5 x mb to 1e-9: {float((np.abs(r[ok] - 5) < 1e-9).mean()):.3f}; float32(5 x mb): {float((np.float32(5 * mb[ok]) == np.float32(pc[ok])).mean()):.3f};"
                  f" poscst an integer: {float((pc[ok] == np.round(pc[ok])).mean()):.3f}; a multiple of 100: {float((np.mod(pc[ok], 100) == 0).mean()):.3f};"
                  f" mb an integer: {float((mb[ok] == np.round(mb[ok])).mean()):.3f}")
            if rule == "on":
                print("      sample (poscst, mb, msr, vr, mp0, cpos0, asq0, asq_max, cpos_max, poscst/mb):")
                for row in j.head(8).iter_rows(named=True):
                    print("        " + "  ".join(f"{row.get(k, float('nan')):.8g}" for k in ("poscst", "pm.mb", "pm.msr", "pm.vr", "mp0", "cpos0", "asq0", "asq_max", "cpos_max"))
                          + f"  {row['poscst'] / row['pm.mb']:.6f}" if row.get("pm.mb") else "        -")
            cols = [k for k in pd_.columns if k != "sid"]
            xs: dict[str, np.ndarray] = {"1": np.ones(j.height), "mp0": g("mp0"), "100*mp0": 100 * g("mp0"), "cpos0": g("cpos0"), "cpos0*mp0": g("cpos0") * g("mp0"),
                                         "asq0": g("asq0"), "asq_max": g("asq_max"), "cpos_max": g("cpos_max"), "asq_max*mp0": g("asq_max") * g("mp0"),
                                         "cfg.nmvcst/mp0": CFG["nmvcst"] / g("mp0"), "cfg.tqub/mp0": CFG["tqub"] / g("mp0")}
            for k in cols:
                v = g(k)
                xs[k] = v
                xs[f"{k}*mp0"] = v * g("mp0")
                xs[f"{k}/mp0"] = v / g("mp0")
                for cn in ("tpub", "cr", "asf", "adur"):
                    xs[f"cfg.{cn}*{k}"] = CFG[cn] * v
            print("      poscst itself, steadiest ratios:")
            steadiest(pc, xs, 5)
            print("      gap = 5 x mb - poscst, steadiest ratios:")
            steadiest(5 * mb - pc, xs, 6)
            print("      poscst rounded down to the lot: floor(5 x mb / 100) x 100 equal share "
                  f"{float((np.floor(5 * mb[ok] / 100) * 100 == pc[ok]).mean()):.3f}; round(5 x mb) {float((np.round(5 * mb[ok]) == pc[ok]).mean()):.3f};"
                  f" floor(5 x mb) {float((np.floor(5 * mb[ok]) == pc[ok]).mean()):.3f}")
            for terms in (["pm.mb"], ["pm.mb", "pm.msr"], ["pm.mb", "pm.vr"], ["pm.mb", "pm.msr", "pm.vr"]):
                if not all(k in cols for k in terms):
                    continue
                A = np.column_stack([*(g(k) for k in terms), np.ones(j.height)])
                m = np.isfinite(A).all(1) & np.isfinite(pc)
                b, *_ = np.linalg.lstsq(A[m], pc[m], rcond=None)
                res = pc[m] - A[m] @ b
                print(f"      poscst ~ {' + '.join(terms)} + 1: " + ", ".join(f"{k} {v:.6g}" for k, v in zip([*terms, "one"], b, strict=True))
                      + f"; R^2 {1 - res.var() / pc[m].var():.8f}; max |resid| / poscst {float(np.max(np.abs(res) / np.abs(pc[m]))):.4g}")
    print("[done]")


if __name__ == "__main__":
    main()
