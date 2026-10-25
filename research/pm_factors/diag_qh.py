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
R = np.round(np.arange(0.900, 0.9651, 0.0005), 4)
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


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--prefix", default="zzhj.p")
    args = ap.parse_args()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    files = sorted(p.stem for p in args.dcp_dir.glob("*.jz"))
    P = R[:, None] ** (15 - np.arange(16))[None, :]  # (n_r, 16) weights of the day's 16 samples
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
                         {kind: per_window(gs, mids, kind) for kind in ("win", "bar")}))
        print(f"done {d} (prev {prev[-1]}): {len(recs)} records so far")
    if not recs:
        return
    mkt = np.array([r[2] for r in recs])
    s, w, s0, w0 = (np.array([r[i] for r in recs], float) for i in (3, 4, 5, 6))
    sz = mkt == SZSE
    # weight: wgt = r^16 w0 + sum_k r^(15-k)
    wpred = (R ** 16)[:, None] * w0[None, :] + P.sum(axis=1)[:, None]
    werr = np.abs(wpred - w[None, :]) / np.abs(w[None, :])
    wmre = np.nanmedian(werr[:, sz], axis=1)
    ib = int(np.argmin(wmre))
    print(f"\nweight: best r = {R[ib]} (half-life {np.log(0.5) / np.log(R[ib]):.2f} samples), median error {wmre[ib]:.2e}, "
          f"within 1e-4 SZSE {np.mean(werr[ib, sz] <= 1e-4):.3f} / SSE {np.mean(werr[ib, ~sz] <= 1e-4):.3f}")
    rows = []
    for kind in ("win", "bar"):
        for q in QN:
            V = np.vstack([r[7][kind][q] for r in recs])  # (n, 16)
            fin = np.isfinite(V)
            V0 = np.where(fin, V, 0.0)
            for i in (ib, *[j for j in (ib - 10, ib + 10) if 0 <= j < R.size]):
                num = (R[i] ** 16) * s0 + V0 @ P[i]
                den = (R[i] ** 16) * w0 + fin.astype(float) @ P[i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    pv, pmv = num / den, s / w
                    e = np.abs(pv - pmv) / np.maximum(np.abs(pmv), 1e-12)
                    kk = float(np.nanmedian((pmv / pv)[sz]))
                    es = np.abs(kk * pv - pmv) / np.maximum(np.abs(pmv), 1e-12)
                rows.append({"kind": kind, "q": q, "r": float(R[i]), "mre": float(np.nanmedian(e[sz])), "ok4_SZSE": float(np.nanmean(e[sz] <= 1e-4)),
                             "ok4_SSE": float(np.nanmean(e[~sz] <= 1e-4)), "k": kk, "mre_k": float(np.nanmedian(es[sz])),
                             "ok4_k": float(np.nanmean(es[sz] <= 1e-4))})
    res = pl.DataFrame(rows)
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200, float_precision=6):
        print(f"\n== {args.prefix} value: EMA over the 16 samples of per-window quantities (best r and r +/- 0.005)")
        print(res.sort("mre").head(20))
        print("\n== up to a factor k")
        print(res.sort("mre_k").head(12))


if __name__ == "__main__":
    main()
