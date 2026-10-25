r"""Stage 4i: ``zzhj.p`` as an EMA over the 16 quarter-hour samples of each day.

``diag_daily``: ``zzhj.p``'s weight follows ``wgt_D = 0.3225 wgt_{D-1} + 10.12`` (77 % within
1e-4), i.e. 16 updates a day with a per-update decay r ~ 0.933 (r^16 = 0.33, sum r^k ~ 10) -- the
16 sampling times of ``zzhj.c`` (the mid 30 s before each quarter-hour end, the last at 14:56:30).
Its value is ~1.7e6, volume-sized. So per stock and day:

    wgt_D = r^16 wgt_{D-1} + sum_k r^(15-k)        sum_D = r^16 sum_{D-1} + sum_k r^(15-k) v_k

with v_k a per-window quantity between consecutive sampling times (``win``) or of the fixed
quarter-hour bar (``bar``): volume, amount, trades, buy / sell / net volume, high - low, VWAP, and
from the 16 mids the log return, its absolute value and square. r is scanned on a grid and picked by
the weight; values are scored as is and up to a constant factor.

    python research/pm_factors/diag_qh.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader
import dcp_ema as E  # tick loading

SZSE = 3554
SAMPLE = np.minimum(np.arange(16) * 900.0 + 870.0, 14190.0)  # zzhj sampling times, trading-clock s
QN = ("vol", "amt", "n_tr", "vol_buy", "vol_sell", "net_vol", "abs_net_vol", "hl", "vwap", "lret", "abs_lret", "lret2", "mid", "one")


def load(path: Path) -> dict[str, dict[str, float]]:
    """Raw state: sid -> key -> value."""
    with gzip.open(path) as fh:
        return json.loads(fh.read())


def windows(t: np.ndarray, kind: str) -> np.ndarray:
    """Window index 0..15 of trading-clock times t (-1 outside): between sampling times, or fixed quarter-hour bars."""
    if kind == "win":
        k = np.searchsorted(SAMPLE, t, side="left")  # (s_{k-1}, s_k]; before the first sample = window 0
        return np.where(k < 16, k, -1)
    k = np.floor(t / 900.0).astype(int)
    return np.where((t >= 0) & (k < 16), k, -1)


def per_window(gs: pl.DataFrame, mids: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    """16 values per quantity for one stock-day."""
    t = gs["t_local"].to_numpy().astype(float)
    cont = gs["s"].to_numpy() >= 34200
    vol, amt, side, px = (gs[c].to_numpy().astype(float) for c in ("vol", "amt", "side", "price"))
    k = windows(t, kind)
    ok = (k >= 0) & cont
    out = {}
    for name, w in (("vol", vol), ("amt", amt), ("n_tr", np.ones(t.size)), ("vol_buy", vol * (side > 0)),
                    ("vol_sell", vol * (side < 0)), ("net_vol", vol * side)):
        a = np.zeros(16)
        np.add.at(a, k[ok], w[ok])
        out[name] = a
    out["abs_net_vol"] = np.abs(out["net_vol"])
    hi, lo = np.full(16, -np.inf), np.full(16, np.inf)
    np.maximum.at(hi, k[ok], px[ok])
    np.minimum.at(lo, k[ok], px[ok])
    out["hl"] = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["vwap"] = np.where(out["vol"] > 0, out["amt"] / out["vol"], np.nan)
        lm = np.log(mids)
        out["lret"] = np.diff(lm, prepend=np.nan)
    out["abs_lret"], out["lret2"] = np.abs(out["lret"]), out["lret"] ** 2
    out["mid"], out["one"] = mids, np.ones(16)
    return out


TRAIL = (60, 180, 300, 600, 900, 1800, 3600)
KINDS = ("win", "bar", *[f"last{L}" for L in TRAIL], "cum")


def trailing(gs: pl.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """Flow quantities over the L seconds before each sampling time (``last<L>``) or since the open (``cum``)."""
    t = gs["t_local"].to_numpy().astype(float)
    cont = gs["s"].to_numpy() >= 34200
    vol, amt, side = (gs[c].to_numpy().astype(float) for c in ("vol", "amt", "side"))
    o = np.argsort(t, kind="stable")
    t = t[o]
    out = {}
    for name, w in (("vol", vol), ("amt", amt), ("n_tr", np.ones(t.size)), ("vol_buy", vol * (side > 0)),
                    ("vol_sell", vol * (side < 0)), ("net_vol", vol * side)):
        c = np.concatenate([[0.0], np.cumsum((w * cont)[o])])
        at = lambda x: c[np.searchsorted(t, x, side="right")]  # noqa: E731, B023 -- running total at times x
        out[name] = at(SAMPLE) - (0.0 if kind == "cum" else at(SAMPLE - float(kind[4:])))
    out["abs_net_vol"] = np.abs(out["net_vol"])
    return out


def solve_r(c: float) -> float:
    """Solve sum_{k<16} r^k = c for r (bisection on 0.5 .. 0.9999)."""
    lo, hi = 0.5, 0.9999
    for _ in range(80):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if (1 - m ** 16) / (1 - m) < c else (lo, m)
    return (lo + hi) / 2


def fit_ac(y0: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares y = a y0 + c, refitted on the better half."""
    keep = np.isfinite(y0) & np.isfinite(y)
    a = c = np.nan
    for _ in range(3):
        (a, c), *_ = np.linalg.lstsq(np.column_stack([y0[keep], np.ones(keep.sum())]), y[keep], rcond=None)
        r = np.abs(y - a * y0 - c)
        keep = keep & (r <= np.nanquantile(r[keep], 0.5))
    return float(a), float(c)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--prefix", default="zzhj.p")
    args = ap.parse_args()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    files = sorted(p.stem for p in args.dcp_dir.glob("*.jz"))
    recs = []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        prev = [f for f in files if f < d]
        if d not in files or not prev:
            continue
        cur, pre = load(args.dcp_dir / f"{d}.jz"), load(args.dcp_dir / f"{prev[-1]}.jz")
        pm = B.load_state(args.dcp_dir / f"{d}.jz")
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(int(s) for s in cur)
        up = E.load(con, int(d), sids, stop)
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        for sid in sids:
            kv, kv0, gs = cur[str(sid)], pre.get(str(sid)), slow.get(sid)
            p = args.prefix
            if kv0 is None or gs is None or None in (kv.get(f"{p}.sum"), kv.get(f"{p}.wgt"), kv0.get(f"{p}.sum"), kv0.get(f"{p}.wgt")):
                continue
            mids = np.array([kv.get(f"zzhj.c.{k}") or np.nan for k in range(16)], float)
            recs.append((d, sid, int(gs["mkt"][0]), kv[f"{p}.sum"], kv[f"{p}.wgt"], kv0[f"{p}.sum"], kv0[f"{p}.wgt"],
                         {kind: (per_window(gs, mids, kind) if kind in ("win", "bar") else trailing(gs, kind)) for kind in KINDS}))
        print(f"done {d} (prev {prev[-1]}): {len(recs)} records so far")
    if not recs:
        return
    mkt = np.array([r[2] for r in recs])
    s, w, s0, w0 = (np.array([r[i] for r in recs], float) for i in (3, 4, 5, 6))
    sz = mkt == SZSE
    # weight: wgt = a w0 + c with a free overnight decay; within the day sum_k r^(15-k) = c gives r
    a, c = fit_ac(w0[sz], w[sz])
    r = solve_r(c)
    werr = np.abs(a * w0 + c - w) / np.abs(w)
    print(f"\nweight: wgt = {a:.6f} wgt_prev + {c:.6f}; within-day r = {r:.6f} (r^16 = {r ** 16:.4f}, overnight extra "
          f"{a / r ** 16:.4f}); median error {np.nanmedian(werr[sz]):.2e}, within 1e-4 SZSE {np.mean(werr[sz] <= 1e-4):.3f} / "
          f"SSE {np.mean(werr[~sz] <= 1e-4):.3f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (s - a * s0) / (w - a * w0)  # the day's own weighted value
    print(f"implied day value x: p10 {np.nanquantile(x, 0.1):.4g}, median {np.nanmedian(x):.4g}, p90 {np.nanquantile(x, 0.9):.4g}")
    rows = []
    for rr in (r, r - 0.01, r + 0.01, 1.0):
        Pr = rr ** (15 - np.arange(16))
        for kind in KINDS:
            for q in (QN if kind in ("win", "bar") else ("vol", "amt", "n_tr", "vol_buy", "vol_sell", "net_vol", "abs_net_vol")):
                V = np.vstack([rec[7][kind][q] for rec in recs])
                fin = np.isfinite(V)
                with np.errstate(divide="ignore", invalid="ignore"):
                    pv = (np.where(fin, V, 0.0) @ Pr) / (fin.astype(float) @ Pr)
                    e = np.abs(pv - x) / np.maximum(np.abs(x), 1e-12)
                    kk = float(np.nanmedian((x / pv)[sz]))
                    es = np.abs(kk * pv - x) / np.maximum(np.abs(x), 1e-12)
                ok = np.isfinite(pv) & np.isfinite(x) & sz
                rk = lambda z: np.argsort(np.argsort(z)).astype(float)  # noqa: E731 -- ranks for Spearman
                rho = float(np.corrcoef(rk(pv[ok]), rk(x[ok]))[0, 1]) if ok.sum() > 10 else np.nan
                rows.append({"kind": kind, "q": q, "r": round(float(rr), 5), "mre": float(np.nanmedian(e[sz])),
                             "ok4_SZSE": float(np.nanmean(e[sz] <= 1e-4)), "ok4_SSE": float(np.nanmean(e[~sz] <= 1e-4)), "k": kk,
                             "mre_k": float(np.nanmedian(es[sz])), "ok4_k": float(np.nanmean(es[sz] <= 1e-4)), "rho": rho})
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200, float_precision=6):
        print(f"\n== {args.prefix} day value x against EMA-weighted per-sample quantities (r from the weight, +/- 0.01, and 1 = plain mean)")
        print(res.sort("mre").head(20))
        print("\n== up to a factor k")
        print(res.sort("mre_k").head(16))
        print("\n== by rank correlation")
        print(res.sort("rho", descending=True).head(12))


if __name__ == "__main__":
    main()
