r"""Two follow-ups of R150 on the runS decision log.

1  vr and poscst take ONE value per stock per day, and PM's Alg object holds members mb / vr / msr next to poscst (R150
   inv): they should be per-stock morning inputs. Per stock: the logged vr / poscst against PM's own start-of-day table
   (stsfile stats/sod: msr, mb, vr; index weights) and products with the config values -- the quantiles of y / x over the
   stocks; q90 / q10 = 1.000 is an identity. Both file rules (the latest file before the day, and on the day).
2  vpen = the config's 1e-07 exactly before rvts (13:00) and ~1e-3 after 14:45: the shape in between. Per 5-minute bucket
   from 12:55: vpen's quantiles across decisions, how many distinct values one stock takes inside a bucket, and whether
   vpen at a given second is the same for every stock; then least squares of vpen on the seconds since 13:00 (linear,
   and against 1 / seconds-to-close forms), all decisions and per position sign.
Read-only.

    python research/pm_factors/runs_sod.py --orders /work/crucible_data/pm_orders --dates 20261112,20261113 --sts /work/crucible_data/pm_stsfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

CFG = {"tqub": 1e7, "nmvcst": 1e7, "tpub": 0.09, "adur": 12.0, "vpen": 1e-7, "asf": 0.8, "cr": 0.0004}


def steadiest(y: np.ndarray, xs: dict[str, np.ndarray], top: int) -> None:
    res = []
    for xn, x in xs.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            r = y / x
        r = r[np.isfinite(r) & (r != 0)]
        if r.size < 30:
            continue
        q10, q50, q90 = np.percentile(r, [10, 50, 90])
        if q10 * q90 > 0:
            res.append((abs(np.log(q90 / q10)), xn, r.size, q50, q10, q90, float(np.mean(np.abs(r / q50 - 1) < 1e-6))))
    res.sort()
    for sp, xn, n, q50, q10, q90, eq in res[:top]:
        print(f"        / {xn:<26} median {q50:.8g}  q10 {q10:.6g} q90 {q90:.6g}  q90/q10 {np.exp(sp):.4f}  equal-to-median share {eq:.3f}  n {n}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--dates", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/pm_stsfile"))
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    c = pl.col
    for d in [x for x in a.dates.split(",") if x]:
        f = a.orders / f"date={d}" / "decisions.parquet"
        if not f.exists():
            print(f"== {d}: no {f}")
            continue
        t = pl.read_parquet(f)
        if "ev_vr" not in t.columns:
            print(f"== {d}: no ev_* columns")
            continue
        t = t.with_columns(c("sid").cast(pl.Int64), ((c("src_epoch").cast(pl.Int64) / 1e9 + 28800) % 86400).alias("sod"),
                           *[c(k).cast(pl.Float64) for k in ("ev_vr", "ev_poscst", "ev_mp", "ev_vpen", "ev_cpos") if k in t.columns])
        print(f"\n== {d}: {t.height:,} decisions, {t['sid'].n_unique()} stocks")
        per = t.group_by("sid").agg(c("ev_vr").first().alias("vr"), c("ev_vr").n_unique().alias("n_vr"), c("ev_poscst").first().alias("poscst"),
                                    c("ev_poscst").n_unique().alias("n_pc"), c("ev_mp").first().alias("mp0"), c("ev_mp").median().alias("mp"),
                                    c("ev_cpos").first().alias("cpos0")).sort("sid")
        print(f"  1 per stock: vr single-valued on {float((per['n_vr'] == 1).mean()):.3f}, poscst on {float((per['n_pc'] == 1).mean()):.3f};"
              f" vr q10/50/90 {' / '.join(f'{v:.5g}' for v in np.percentile(per['vr'].drop_nulls().to_numpy(), [10, 50, 90]))};"
              f" poscst q10/50/90 {' / '.join(f'{v:.5g}' for v in np.percentile(per['poscst'].drop_nulls().to_numpy(), [10, 50, 90]))}")
        try:
            from ic_score import pm_daily
            for rule in ("before", "on"):
                pd_ = pm_daily(a.sts, d, per["sid"].to_list(), rule)
                if pd_ is None:
                    print(f"     rule {rule}: no stsfile table")
                    continue
                j = per.join(pd_, on="sid", how="left")
                g = lambda k: j[k].to_numpy().astype(float)  # noqa: E731
                cols = [k for k in pd_.columns if k != "sid"]
                xs: dict[str, np.ndarray] = {"1": np.ones(j.height), "mp0": g("mp0"), "cpos0": g("cpos0"), "cpos0*mp0": g("cpos0") * g("mp0")}
                for k in cols:
                    v = g(k)
                    xs[k] = v
                    xs[f"{k}*mp0"] = v * g("mp0")
                    xs[f"{k}/mp0"] = v / g("mp0")
                    xs[f"sqrt({k})"] = np.sqrt(np.where(v > 0, v, np.nan))
                    for cn in ("tpub", "cr", "asf", "adur", "nmvcst"):
                        xs[f"cfg.{cn}*{k}"] = CFG[cn] * v
                for k1 in ("pm.msr", "pm.mb", "pm.vr"):
                    for k2 in ("pm.msr", "pm.mb", "pm.vr"):
                        if k1 != k2 and k1 in cols and k2 in cols:
                            xs[f"{k1}/{k2}"] = g(k1) / g(k2)
                            xs[f"{k1}*{k2}"] = g(k1) * g(k2)
                print(f"     rule {rule}: {len(cols)} table columns {cols[:12]}; steadiest ratios over stocks:")
                for yn in ("vr", "poscst"):
                    print(f"     {yn}:")
                    steadiest(g(yn), xs, a.top)
        except Exception as e:  # xarray / the table missing: part 2 still runs
            print(f"     stsfile join skipped: {e!r}")
        pm_ = t.filter(c("sod") >= 46500).with_columns(((c("sod") - 46800) // 300).cast(pl.Int64).alias("b"), (c("sod") - 46800).alias("x"))
        print("  2 vpen from 12:55, 5-minute buckets (minutes after 13:00: n, q10 / q50 / q90, distinct values per stock, cross-stock spread within one second):")
        sec = pm_.with_columns(c("sod").floor().alias("s")).group_by("s").agg((c("ev_vpen").max() / c("ev_vpen").min()).alias("sp"), c("ev_vpen").n_unique().alias("nu"), c("b").first())
        for (b,), sub in sorted(pm_.group_by("b"), key=lambda kv: kv[0][0]):
            v = sub["ev_vpen"].drop_nulls().to_numpy()
            if v.size < 20:
                continue
            ss = sec.filter(c("b") == b)
            print(f"     {int(b) * 5:>4}: n {v.size:>7,}  {' / '.join(f'{x:.4g}' for x in np.percentile(v, [10, 50, 90]))}  distinct/stock {sub.group_by('sid').agg(c('ev_vpen').n_unique())['ev_vpen'].median():.0f}"
                  f"  same-second max/min median {ss['sp'].median():.4f}")
        ramp = pm_.filter((c("x") > 0) & (c("x") < 6300) & c("ev_vpen").is_not_null())
        if ramp.height > 100:
            x, y, cp = ramp["x"].to_numpy(), ramp["ev_vpen"].to_numpy(), ramp["ev_cpos"].to_numpy()
            forms = {"linear in s since 13:00": x, "s^2": x * x, "1 / (s to 14:57)": 1.0 / (7020 - x), "1 / (s to 15:00)": 1.0 / (7200 - x),
                     "exp(s / 1000)": np.exp(x / 1000), "log(1 + s)": np.log1p(x)}
            for name, z in forms.items():
                for tag, m in (("all", np.ones(x.size, bool)), ("cpos>0", cp > 0), ("cpos<0", cp < 0), ("cpos=0", cp == 0)):
                    if m.sum() < 100:
                        continue
                    A = np.column_stack([z[m], np.ones(int(m.sum()))])
                    bb, *_ = np.linalg.lstsq(A, y[m], rcond=None)
                    r = y[m] - A @ bb
                    print(f"     vpen ~ {name:<24} {tag:<7} slope {bb[0]:.6g} intercept {bb[1]:.6g}  R^2 {1 - r.var() / y[m].var():.5f}  n {int(m.sum()):,}")
    print("[done]")


if __name__ == "__main__":
    main()
