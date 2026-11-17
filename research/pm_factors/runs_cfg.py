r"""runS decision log against PM's alg config (sc.p3.json): which logged quantity is which config value times what?

Config (read by the supervisor on 224, 2026-11-17): tqub 1e7, nmvcst 1e7, tpub 0.09, adur 12, vpen 1e-07, rvts 130000000
(13:00:00.000), asf 0.8, neg_nmvcst false. The decision line logs mp, fa, vpen, vr, poscst, cpos, asq, hs (ev_* columns of
pm_orders' decisions.parquet). For each of vpen / vr / poscst and each candidate x (products of the config values with
cpos, mp, winsdQty, asq ...): the quantiles of y / x and the spread q90 / q10 -- a spread of 1.000 is an identity, and the
median is the constant. Done per time segment (before 13:00 = rvts, 13:00-14:45, after 14:45) to see where a rule
switches. Also: how many distinct values each quantity takes per stock, and the caps (|cpos x mp| against nmvcst, asq
against tqub). Read-only.

    python research/pm_factors/runs_cfg.py --orders /work/crucible_data/pm_orders --dates 20261112,20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

CFG = {"tqub": 1e7, "nmvcst": 1e7, "tpub": 0.09, "adur": 12.0, "vpen": 1e-7, "asf": 0.8}
SEGS = (("<13:00", 0, 780), ("13:00-14:45", 780, 885), (">=14:45", 885, 1440))


def spread(y: np.ndarray, x: np.ndarray) -> tuple[int, float, float, float]:
    """n, median and q10 / q90 of y / x over the rows where both are finite and x != 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = y / x
    r = r[np.isfinite(r) & (x != 0)]
    if r.size < 50:
        return r.size, np.nan, np.nan, np.nan
    q10, q50, q90 = np.percentile(r, [10, 50, 90])
    return r.size, q50, q10, q90


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--dates", required=True)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    c = pl.col
    for d in [x for x in a.dates.split(",") if x]:
        f = a.orders / f"date={d}" / "decisions.parquet"
        if not f.exists():
            print(f"== {d}: no {f}")
            continue
        t = pl.read_parquet(f).with_columns((((c("src_epoch").cast(pl.Int64) / 1e9 + 28800) % 86400) // 60).alias("minute"))
        have = [k for k in ("fa", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "ev_cpos", "ev_asq", "ev_hs", "winsdQty", "lthr", "executp", "pos", "pp", "asq", "hs") if k in t.columns]
        t = t.with_columns(c(k).cast(pl.Float64) for k in have)
        print(f"\n== {d}: {t.height:,} decisions; columns {have}")
        g = lambda k: t[k].to_numpy().astype(float) if k in t.columns else np.full(t.height, np.nan)  # noqa: E731
        mp, cp, w, vp, vr, pc, fa, hs, asq, minute = (g(k) for k in ("ev_mp", "ev_cpos", "winsdQty", "ev_vpen", "ev_vr", "ev_poscst", "fa", "ev_hs", "ev_asq", "minute"))
        ex, lt = g("executp"), g("lthr")
        mv = cp * mp
        print("  distinct values per stock (median over stocks): " + "  ".join(
            f"{k} {t.group_by('sid').agg(c(k).n_unique())[k].median():.0f}" for k in ("ev_vpen", "ev_vr", "ev_poscst", "ev_cpos", "ev_asq", "ev_hs", "ev_mp") if k in t.columns)
            + f"  (decisions per stock {t.group_by('sid').len()['len'].median():.0f})")
        fin = np.isfinite(mv)
        print(f"  caps: |cpos x mp| max {np.nanmax(np.abs(mv)):.4g}, share > nmvcst 1e7 {float((np.abs(mv[fin]) > 1e7).mean()):.4f}, share of cpos < 0 {float((cp[np.isfinite(cp)] < 0).mean()):.4f};"
              f" asq max {np.nanmax(asq):.4g}, asq x mp max {np.nanmax(asq * mp):.4g}; |executp x mp| max {np.nanmax(np.abs(ex * mp)):.4g}")
        k = CFG
        xs = {"1": np.ones(t.height), "cfg.vpen": np.full(t.height, k["vpen"]), "cpos": cp, "|cpos|": np.abs(cp), "mp": mp, "cpos*mp": mv, "|cpos*mp|": np.abs(mv),
              "cfg.vpen*|cpos|": k["vpen"] * np.abs(cp), "cfg.vpen*|cpos*mp|": k["vpen"] * np.abs(mv), "cfg.vpen*mp": k["vpen"] * mp, "cfg.vpen*mp^2": k["vpen"] * mp * mp,
              "cfg.vpen*w": k["vpen"] * w, "cfg.vpen*w*mp": k["vpen"] * w * mp, "w": w, "w*mp": w * mp, "asq": asq, "asq*mp": asq * mp, "hs": hs, "|fa|": np.abs(fa),
              "cfg.vpen*cpos^2": k["vpen"] * cp * cp, "cfg.vpen*(cpos*mp)^2": k["vpen"] * mv * mv, "0.5*cfg.vpen*(cpos*mp)^2": 0.5 * k["vpen"] * mv * mv,
              "cfg.vpen*cpos^2*mp": k["vpen"] * cp * cp * mp, "cfg.tpub*mp": k["tpub"] * mp, "cfg.tpub": np.full(t.height, k["tpub"]), "cfg.nmvcst": np.full(t.height, k["nmvcst"]),
              "|cpos*mp|/cfg.nmvcst": np.abs(mv) / k["nmvcst"], "cfg.asf": np.full(t.height, k["asf"]), "cfg.adur": np.full(t.height, k["adur"]), "-lthr": -lt, "-lthr*w": -lt * w}
        ys = {"vpen": vp, "vr": vr, "poscst": pc, "hs": hs, "-lthr": -lt}
        ys.update({"poscst/vpen": pc / np.where(vp != 0, vp, np.nan), "vpen/vr": vp / np.where(vr != 0, vr, np.nan), "hs/vr": hs / np.where(vr != 0, vr, np.nan)})
        for name, lo, hi in (("all day", 0, 1440), *SEGS):
            m = (minute >= lo) & (minute < hi)
            if m.sum() < 50:
                print(f"  -- {name}: {int(m.sum())} decisions, skipped")
                continue
            print(f"  -- {name}: {int(m.sum()):,} decisions")
            for yn, y in ys.items():
                yy = y[m]
                fin = yy[np.isfinite(yy)]
                if fin.size < 50:
                    continue
                res = []
                for xn, x in xs.items():
                    if xn == yn or xn == f"|{yn}|":
                        continue
                    n, q50, q10, q90 = spread(yy, x[m])
                    if n >= 50 and q50 != 0 and np.isfinite(q50) and q10 * q90 > 0:
                        res.append((abs(np.log(q90 / q10)), xn, n, q50, q10, q90))
                res.sort()
                print(f"     {yn:<12} q10/50/90 {' / '.join(f'{v:.4g}' for v in np.percentile(fin, [10, 50, 90]))}; steadiest ratios (q90/q10 -> 1 = identity):")
                for sp, xn, n, q50, q10, q90 in res[: a.top]:
                    print(f"        / {xn:<28} median {q50:.6g}  q10 {q10:.6g} q90 {q90:.6g}  q90/q10 {np.exp(sp):.4f}  n {n:,}")
    print("[done]")


if __name__ == "__main__":
    main()
