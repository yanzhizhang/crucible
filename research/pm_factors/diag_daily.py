r"""Stage 4h: the once-a-day EMAs (``zzhj.p``, ``zzui.r``, ...) from the state files alone.

``diag_rest`` found ``zzhj.p`` / ``zzui.r`` updated once a day (at 14:56:30 / 14:56:50) as
``wgt_D = a wgt_{D-1} + c``, ``sum_D = a sum_{D-1} + c x_D``. ``diag_slots`` found ``zzhj.c.0..15``
to be the mid 30 s before each quarter-hour end (the last one 14:56:30), and the state also holds
the 10 s price grids ``zzxo.r.0..29`` / ``zzui.r.0..9`` and the day's prices. So the day's value
``x_D`` can be read off two consecutive state files and matched against expressions of those, with
no ticks at all:

* fit ``a``, ``c`` on the weights (least squares over stocks and days, refitted on the better half);
* implied ``x_D = (sum_D - a sum_{D-1}) / (wgt_D - a wgt_{D-1})``;
* candidates from the day's state: per grid (16 quarter-hour mids, 30 / 10 point 10 s grids) the
  mean, first, last, min, max, log(last / first), log(last / mean), log(max / min), std and log
  sum of squares of the log steps; the day's open / high / low / last / VWAP and log ratios
  between them and the grids; scored as is and up to a constant factor.

    python research/pm_factors/diag_daily.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # default state directory

PREFIXES = ("zzhj.p", "zzui.r", "zzba.p", "zzba.r")
SZSE_PREFIX = ("000", "001", "002", "003", "300", "301")


def load(path: Path) -> dict[str, dict[str, float]]:
    """Raw state: sid -> key -> value."""
    with gzip.open(path) as fh:
        return json.loads(fh.read())


def grid(kv: dict[str, float], name: str, n: int) -> np.ndarray:
    """Read the list key ``name.0 .. name.<n-1>`` as an array (NaN where missing or 0)."""
    a = np.array([kv.get(f"{name}.{k}") or np.nan for k in range(n)], float)
    return np.where(a > 0, a, np.nan)


def candidates(kv: dict[str, float]) -> dict[str, float]:
    """Candidate day values from one stock's day-end state."""
    out: dict[str, float] = {}
    grids = {"hj": grid(kv, "zzhj.c", 16), "xo": grid(kv, "zzxo.r", 30), "ui": grid(kv, "zzui.r", 10)}
    day = {"open": kv.get("zzpk.open"), "high": kv.get("zzpk.high"), "low": kv.get("zzpk.low"), "last": kv.get("zzxu.last"),
           "hm": kv.get("zzmh.hm"),
           "vwap": (kv["zzqj.tpv"] / kv["zzqj.v"]) if kv.get("zzqj.v") else None}
    day = {k: float(v) for k, v in day.items() if v}
    with np.errstate(divide="ignore", invalid="ignore"):
        for g, a in grids.items():
            f = a[np.isfinite(a)]
            if f.size < 2:
                continue
            lr = np.diff(np.log(f))
            out.update({f"{g}_mean": f.mean(), f"{g}_first": f[0], f"{g}_last": f[-1], f"{g}_min": f.min(), f"{g}_max": f.max(),
                        f"{g}_lret": np.log(f[-1] / f[0]), f"{g}_l_last_mean": np.log(f[-1] / f.mean()),
                        f"{g}_lrange": np.log(f.max() / f.min()), f"{g}_sd": lr.std(), f"{g}_lrv": np.log((lr ** 2).sum()),
                        f"{g}_absret_mean": np.abs(lr).mean(), f"{g}_l_last_prev": np.log(f[-1] / f[-2])})
            for dn, dv in day.items():
                out[f"l_{g}last_{dn}"] = np.log(f[-1] / dv)
                out[f"l_{g}mean_{dn}"] = np.log(f.mean() / dv)
        for dn, dv in day.items():
            out[dn] = dv
            for en, ev in day.items():
                if dn < en:
                    out[f"l_{dn}_{en}"] = np.log(dv / ev)
    return {k: float(v) for k, v in out.items() if np.isfinite(v)}


def fit_ac(y0: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares y = a y0 + c, refitted on the better half."""
    keep = np.isfinite(y0) & np.isfinite(y)
    a = c = np.nan
    for _ in range(3):
        if keep.sum() < 10:
            break
        (a, c), *_ = np.linalg.lstsq(np.column_stack([y0[keep], np.ones(keep.sum())]), y[keep], rcond=None)
        r = np.abs(y - a * y0 - c)
        keep = keep & (r <= np.nanquantile(r[keep], 0.5))
    return float(a), float(c)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    files = sorted(p.stem for p in args.dcp_dir.glob("*.jz"))
    rows = []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        prev = [f for f in files if f < d]
        if d not in files or not prev:
            continue
        cur, pre = load(args.dcp_dir / f"{d}.jz"), load(args.dcp_dir / f"{prev[-1]}.jz")
        for sid, kv in cur.items():
            if sid not in pre:
                continue
            cand = candidates(kv)
            szse = f"{int(sid):06d}".startswith(SZSE_PREFIX)
            for p in PREFIXES:
                s, w, s0, w0 = (kv.get(f"{p}.sum"), kv.get(f"{p}.wgt"), pre[sid].get(f"{p}.sum"), pre[sid].get(f"{p}.wgt"))
                if None in (s, w, s0, w0):
                    continue
                rows.append({"d": d, "sid": int(sid), "szse": szse, "p": p, "s": s, "w": w, "s0": s0, "w0": w0, **cand})
        print(f"loaded {d} (prev {prev[-1]})")
    if not rows:
        return
    df = pl.DataFrame(rows, infer_schema_length=None)
    cols = [c for c in df.columns if c not in ("d", "sid", "szse", "p", "s", "w", "s0", "w0")]
    out, fits = [], []
    for p in PREFIXES:
        g = df.filter(pl.col("p") == p)
        if g.height < 30:
            continue
        a, c = fit_ac(g["w0"].to_numpy(), g["w"].to_numpy())
        w, w0, s, s0 = (g[k].to_numpy() for k in ("w", "w0", "s", "s0"))
        resid = np.abs(w - a * w0 - c) / np.abs(w)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (s - a * s0) / (w - a * w0)
        fits.append({"p": p, "a": a, "c": c, "half_days": float(-np.log(2) / np.log(a)) if 0 < a < 1 else None,
                     "w_mre": float(np.nanmedian(resid)), "w_ok4": float(np.mean(resid <= 1e-4)),
                     "x_p10": float(np.nanquantile(x, 0.1)), "x_med": float(np.nanmedian(x)), "x_p90": float(np.nanquantile(x, 0.9))})
        sz = g["szse"].to_numpy()
        for cn in cols:
            v = g[cn].to_numpy().astype(float)
            ok = np.isfinite(v) & np.isfinite(x)
            if ok.sum() < 30:
                continue
            e = np.abs(v[ok] - x[ok]) / np.maximum(np.abs(x[ok]), 1e-12)
            with np.errstate(divide="ignore", invalid="ignore"):
                k = float(np.nanmedian(x[ok] / v[ok]))
            es = np.abs(k * v[ok] - x[ok]) / np.maximum(np.abs(x[ok]), 1e-12)
            rk = lambda z: np.argsort(np.argsort(z)).astype(float)  # noqa: E731 -- ranks for Spearman
            rho = float(np.corrcoef(rk(v[ok]), rk(x[ok]))[0, 1])
            out.append({"p": p, "cand": cn, "mre": float(np.median(e)), "ok4": float(np.mean(e <= 1e-4)),
                        "ok4_szse": float(np.mean((e <= 1e-4)[sz[ok]])) if sz[ok].any() else None,
                        "k": k, "mre_k": float(np.median(es)), "ok4_k": float(np.mean(es <= 1e-4)), "rho": rho, "n": int(ok.sum())})
    res = pl.DataFrame(out)
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=220, fmt_str_lengths=30, float_precision=6):
        print("\n== weight fit wgt_D = a wgt_D-1 + c (half_days = half-life in days) and the implied day value x")
        print(pl.DataFrame(fits))
        print(f"\n== best {args.top} candidates per prefix for x (as is), then up to a factor k")
        print(res.sort("mre").group_by("p", maintain_order=True).head(args.top))
        print(res.sort("mre_k").group_by("p", maintain_order=True).head(args.top).select("p", "cand", "k", "mre_k", "ok4_k", "rho"))


if __name__ == "__main__":
    main()
