r"""Exact relations in runS's decision log: lthr and executp as one- or two-term identities of the logged inputs.

The decision line prints its doubles at full precision, so a true formula leaves a residual at rounding level. For
y in (lthr, executp) and a pool of terms built from the line's own values (fa, hs, vpen, vr, poscst, cpos, asq, mp, pos,
pp, winsdQty) and the config constants (cr 0.0004, tpub 0.09, asf 0.8, adur 12), per time segment (before 13:00,
13:00-14:45, after): every single term (y = a x) and every pair (y = a x1 + b x2), least squares on a row sample, ranked
by the 99% quantile of |residual| / scale(y); an identity reads ~1e-12 or less and its coefficients are the constants.
For executp the rows are split into free rows and rows sitting on a bound (executp == +-asq, == -pos, == poscst-type
caps), listed first. Read-only.

    python research/pm_factors/runs_ident.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl

CFG = {"cr": 0.0004, "tpub": 0.09, "asf": 0.8, "adur": 12.0}
SEGS = (("<13:00", 0, 780), ("13:00-14:45", 780, 885), (">=14:45", 885, 1440))


def fits(y: np.ndarray, terms: dict[str, np.ndarray], top: int) -> None:
    sc = np.nanpercentile(np.abs(y), 50) or 1.0
    res = []
    names = list(terms)
    for k in names:
        x = terms[k]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 200 or not np.any(x[ok]):
            continue
        a = float(x[ok] @ y[ok] / (x[ok] @ x[ok]))
        r = np.abs(y[ok] - a * x[ok])
        res.append((float(np.percentile(r, 99)) / sc, f"{a:.12g} x {k}", int(ok.sum())))
    for k1, k2 in itertools.combinations(names, 2):
        A = np.column_stack([terms[k1], terms[k2]])
        ok = np.isfinite(A).all(1) & np.isfinite(y)
        if ok.sum() < 200:
            continue
        b, *_ = np.linalg.lstsq(A[ok], y[ok], rcond=None)
        r = np.abs(y[ok] - A[ok] @ b)
        res.append((float(np.percentile(r, 99)) / sc, f"{b[0]:.12g} x {k1} + {b[1]:.12g} x {k2}", int(ok.sum())))
    res.sort(key=lambda t: t[0])
    for e, f, n in res[:top]:
        print(f"        q99 |resid| / scale {e:.3g}   {f}   (n {n})")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--rows", type=int, default=30000)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    c = pl.col
    t = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet").with_columns((((c("src_epoch").cast(pl.Int64) / 1e9 + 28800) % 86400) // 60).alias("minute"))
    want = ("fa", "hs", "lthr", "executp", "asq", "pos", "pp", "winsdQty", "asf", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "ev_cpos")
    have = [k for k in want if k in t.columns]
    t = t.with_columns(c(k).cast(pl.Float64) for k in have).drop_nulls([k for k in ("lthr", "executp", "fa", "hs") if k in have])
    print(f"== {a.date}: {t.height:,} trade decisions; columns {have}")
    print(f"   lthr == 0: {float((t['lthr'] == 0).mean()):.3f}; lthr < 0: {float((t['lthr'] < 0).mean()):.3f}; distinct lthr per stock per minute (median): "
          f"{t.group_by('sid', 'minute').agg(c('lthr').n_unique())['lthr'].median():.0f}; executp == 0: {float((t['executp'] == 0).mean()):.3f}")
    rng = np.random.default_rng(0)
    for name, lo, hi in SEGS:
        s = t.filter((c("minute") >= lo) & (c("minute") < hi))
        if s.height < 500:
            print(f"-- {name}: {s.height} rows, skipped")
            continue
        if s.height > a.rows:
            s = s[np.sort(rng.choice(s.height, a.rows, replace=False))]
        g = lambda k: s[k].to_numpy().astype(float) if k in s.columns else np.full(s.height, np.nan)  # noqa: E731
        fa, hs, lthr, ex, asq, pos, pp, w, mp, vp, vr, pc, cp = (g(k) for k in ("fa", "hs", "lthr", "executp", "asq", "pos", "pp", "winsdQty", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "ev_cpos"))
        one = np.ones(s.height)
        print(f"-- {name}: {s.height:,} rows (sample); lthr q10/50/90 {' / '.join(f'{v:.4g}' for v in np.nanpercentile(lthr, [10, 50, 90]))};"
              f" hs {' / '.join(f'{v:.4g}' for v in np.nanpercentile(hs, [10, 50, 90]))}; vpen {' / '.join(f'{v:.4g}' for v in np.nanpercentile(vp, [10, 50, 90]))}")
        with np.errstate(divide="ignore", invalid="ignore"):
            base = {"1": one, "hs": hs, "vr": vr, "hs*vr": hs * vr, "cr": CFG["cr"] * one, "vpen": vp, "vpen*cpos": vp * cp, "vpen*|cpos|": vp * np.abs(cp), "vpen*cpos*mp": vp * cp * mp,
                    "vpen*|cpos|*mp": vp * np.abs(cp) * mp, "vpen*poscst": vp * pc, "vpen*poscst*mp": vp * pc * mp, "vpen*w": vp * w, "vpen*w*mp": vp * w * mp, "vpen*asq": vp * asq,
                    "vpen*pos": vp * pos, "vpen*|pos|": vp * np.abs(pos), "vpen*pos*mp": vp * pos * mp, "vpen*pp": vp * pp, "sqrt(vpen)": np.sqrt(vp), "vpen^2": vp * vp,
                    "hs*vpen": hs * vp, "1/mp": 1.0 / mp, "cr/mp": CFG["cr"] / mp, "|cpos|/poscst": np.abs(cp) / pc, "cpos/poscst": cp / pc, "vpen*cpos/poscst": vp * cp / pc,
                    "hs*|cpos|/poscst": hs * np.abs(cp) / pc, "hs*cpos/poscst": hs * cp / pc, "vr*|cpos|/poscst": vr * np.abs(cp) / pc}
            print("     lthr:")
            fits(lthr, base, a.top)
            for tag, m in (("pos > 0", pos > 0), ("pos < 0", pos < 0), ("pos == 0", pos == 0)):
                if m.sum() >= 500:
                    print(f"     lthr, {tag} ({int(m.sum())} rows):")
                    fits(lthr[m], {k: v[m] for k, v in base.items()}, 3)
            bound = {"== +asq": ex == asq, "== -asq": ex == -asq, "== pos": ex == pos, "== cpos": ex == cp, "== 0": ex == 0, "== poscst": ex == pc, "== -poscst": ex == -pc,
                     "== pp": ex == pp, "integer": ex == np.round(ex), "multiple of 100": np.mod(ex, 100) == 0}
            print("     executp: " + ", ".join(f"{k} {float(v.mean()):.3f}" for k, v in bound.items()))
            free = ~(bound["== +asq"] | bound["== -asq"] | bound["== 0"] | bound["== poscst"] | bound["== -poscst"])
            ext = {"1": one, "fa": fa, "fa*w": fa * w, "hs*w": hs * w, "lthr*w": lthr * w, "fa/vpen": fa / vp, "hs/vpen": hs / vp, "lthr/vpen": lthr / vp, "fa/(vpen*mp)": fa / (vp * mp),
                   "hs/(vpen*mp)": hs / (vp * mp), "lthr/(vpen*mp)": lthr / (vp * mp), "fa/(vpen*mp^2)": fa / (vp * mp * mp), "hs/(vpen*mp^2)": hs / (vp * mp * mp),
                   "sign(fa)*hs/vpen": np.sign(fa) * hs / vp, "sign(fa)*hs/(vpen*mp)": np.sign(fa) * hs / (vp * mp), "sign(fa)*hs*w": np.sign(fa) * hs * w, "cpos": cp, "pos": pos, "pp": pp,
                   "asq": asq, "poscst": pc, "w": w, "fa*poscst": fa * pc, "fa*poscst/hs": fa * pc / hs, "fa*w/hs": fa * w / hs, "fa/hs": fa / hs, "fa*vr": fa * vr, "fa*w*vr": fa * w * vr,
                   "fa*asf": CFG["asf"] * fa, "fa*w*asf": CFG["asf"] * fa * w}
            for tag, m in (("free rows, fa > 0", free & (fa > 0)), ("free rows, fa < 0", free & (fa < 0))):
                if m.sum() >= 500:
                    print(f"     executp, {tag} ({int(m.sum())} rows):")
                    fits(ex[m], {kk: v[m] for kk, v in ext.items()}, a.top)
    print("[done]")


if __name__ == "__main__":
    main()
