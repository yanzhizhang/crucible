r"""Every averaged key of PM's .jz at once: which snapshot quantity, averaged at PM's own half-life, is it proportional to?

last_probe compares a key with the LAST snapshot's quantity, which only works for half-lives of a few seconds. Here the
half-life comes from PM's own weight first: an EMA with weight 1 per update and decay 2^(-dt/h) settles at
wgt = 1 / (1 - 2^(-dt/h)), so h = -dt / log2(1 - 1/wgt) with dt the median snapshot gap (3 s). Then, per key and per
candidate quantity q, q's EMA at that half-life over the stock's snapshots up to the key's .t (weights below 2^-40 dropped)
is held against PM's value: the ratio's median and IQR / median, and the log-log slope / R^2 (squares, roots). A small
spread / slope 1 names the input; the weight-implied half-life is printed too. Keys whose input is cumulative from
yesterday (carried, c0) won't fit -- they show up as a big spread and are listed as such. Read-only.

    python research/pm_factors/ema_sweep.py --date 20261113 --jz /work/crucible_data/seed_probe/HS300/20261113.jz \
        --raw /work/crucible_data/dysl2_raw --skip-verified data/reports/r125/verify_20261113.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

CST_NS = 8 * 3600 * 10**9


def tclock(ns: np.ndarray) -> np.ndarray:
    """UTC epoch ns -> trading clock seconds (09:30 = 0, lunch removed)."""
    s = ((ns + CST_NS) / 1e9) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def candidates(g: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per-snapshot candidate quantities of one stock (hm = running max of the in-session price)."""
    p = g["price"].to_numpy().astype(float)
    ex = ((g["time"].to_numpy() + CST_NS) / 1e9) % 86400
    insess = (ex >= 34200) & (ex < 53820)
    hm = np.maximum.accumulate(np.where(insess, p, -np.inf))
    hm = np.where(np.isfinite(hm), hm, p)
    lm = np.minimum.accumulate(np.where(insess, p, np.inf))
    lm = np.where(np.isfinite(lm), lm, p)
    c = {k: g[k].to_numpy().astype(float) for k in ("pre_close", "open", "high", "low", "b1", "a1", "bv1", "av1", "total_volume", "total_amount")}
    pc, b, a_ = c["pre_close"], c["b1"], c["a1"]
    mid = (a_ + b) / 2
    dv = np.diff(c["total_volume"], prepend=c["total_volume"][0])
    da = np.diff(c["total_amount"], prepend=c["total_amount"][0])
    with np.errstate(divide="ignore", invalid="ignore"):
        q = {"price": p, "price/pc": p / pc, "ret_pc": p / pc - 1, "|ret_pc|": np.abs(p / pc - 1), "ret_open": p / c["open"] - 1,
             "hm-p": hm - p, "(hm-p)/hm": (hm - p) / hm, "((hm-p)/hm)^2": ((hm - p) / hm) ** 2, "p-lm": p - lm, "(p-lm)/lm": (p - lm) / lm,
             "(hm-lm)/p": (hm - lm) / p, "high-low": c["high"] - c["low"], "(high-low)/p": (c["high"] - c["low"]) / p,
             "spread": a_ - b, "spread/mid": (a_ - b) / mid, "half_spread/mid": (a_ - b) / mid / 2, "mid": mid, "mid/pc": mid / pc,
             "micro": (b * c["av1"] + a_ * c["bv1"]) / (c["av1"] + c["bv1"]), "(micro-mid)/mid": ((b * c["av1"] + a_ * c["bv1"]) / (c["av1"] + c["bv1"]) - mid) / mid,
             "imb1": (c["bv1"] - c["av1"]) / (c["bv1"] + c["av1"]), "depth1": c["bv1"] + c["av1"], "depth1_amt": c["bv1"] * b + c["av1"] * a_,
             "bv1": c["bv1"], "av1": c["av1"], "dvol": dv, "damt": da, "|dvol|>0": (dv > 0).astype(float), "dvol*p": dv * p, "vwap_snap": np.where(dv > 0, da / dv, np.nan),
             "vwap_snap/p": np.where(dv > 0, da / dv / p, np.nan), "dlogp": np.diff(np.log(p), prepend=np.log(p[0])), "dlogp^2": np.diff(np.log(p), prepend=np.log(p[0])) ** 2,
             "|dlogp|": np.abs(np.diff(np.log(p), prepend=np.log(p[0]))), "vol_cum": c["total_volume"], "amt_cum": c["total_amount"]}
    return q


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/dysl2_raw"))
    ap.add_argument("--keys", default="auto", help="comma list of key stems, or 'auto' = every key with a .wgt")
    ap.add_argument("--skip-verified", type=Path, default=None, help="verify.py --csv: keys already bit-exact are skipped")
    ap.add_argument("--exact-min", type=float, default=0.987)
    ap.add_argument("--dt", type=float, default=3.0)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    st = {int(s): kv for s, kv in json.loads(gzip.open(a.jz).read()).items()}
    stems = sorted({k[:-4] for kv in st.values() for k in kv if k.endswith(".wgt")}) if a.keys == "auto" else a.keys.split(",")
    if a.skip_verified and a.skip_verified.exists():
        done = set()
        for r in pl.read_csv(a.skip_verified).iter_rows(named=True):
            if float(r.get("all_1e4") or r.get("all") or 0) >= a.exact_min:
                done.add(re.sub(r"\|.*$", "", r["key"]))
        stems = [s for s in stems if s not in done]
    q = (pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={a.date}/part-*.parquet")),
                         columns=["symbol_id", "time", "spider_ts", "price", "pre_close", "open", "high", "low", "bid_px", "ask_px", "bid_vol", "ask_vol",
                                  "total_volume", "total_amount"])
         .filter(pl.col("symbol_id").is_in(list(st)) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts")
         .with_columns(pl.col("bid_px").cast(pl.List(pl.Float64)).list.first().alias("b1"), pl.col("ask_px").cast(pl.List(pl.Float64)).list.first().alias("a1"),
                       pl.col("bid_vol").cast(pl.List(pl.Int64)).list.first().alias("bv1"), pl.col("ask_vol").cast(pl.List(pl.Int64)).list.first().alias("av1")))
    per = {}
    for (sid,), g in q.group_by("symbol_id"):
        per[int(sid)] = (tclock(g["spider_ts"].to_numpy()), candidates(g))
    names = list(next(iter(per.values()))[1])
    print(f"{a.date}: {len(stems)} keys to sweep, {len(per)} stocks, {len(names)} candidate quantities")
    summary = []
    for key in stems:
        w = np.array([kv[f"{key}.wgt"] for kv in st.values() if kv.get(f"{key}.wgt")], float)
        if w.size < 50:
            continue
        wm = float(np.median(w))
        h = -a.dt / np.log2(1 - 1 / wm) if wm > 1 else 0.0
        res = []
        pms, xs = {n: [] for n in names}, {n: [] for n in names}
        for sid, kv in st.items():
            if not kv.get(f"{key}.wgt") or sid not in per or f"{key}.sum" not in kv:
                continue
            t, cq = per[sid]
            tl = kv.get(f"{key}.t", np.inf)
            m = t <= tl + 1e-6
            if not m.any():
                continue
            T = t[m][-1]
            pmv = kv[f"{key}.sum"] / kv[f"{key}.wgt"]
            if h > 0:
                sel = m & (T - t < 40 * h)
                wt = np.power(2.0, -(T - t[sel]) / h)
            else:
                sel = np.zeros_like(m)
                sel[np.nonzero(m)[0][-1]] = True
                wt = np.ones(1)
            for n in names:
                v = cq[n][sel]
                ok = np.isfinite(v)
                if ok.any():
                    pms[n].append(pmv)
                    xs[n].append(float(np.sum(wt[ok] * v[ok]) / np.sum(wt[ok])))
        for n in names:
            pm, x = np.array(pms[n]), np.array(xs[n])
            ok = np.isfinite(x) & np.isfinite(pm) & (x != 0) & (pm != 0)
            if ok.sum() < 50:
                continue
            ratio = pm[ok] / x[ok]
            med = np.median(ratio)
            spread = (np.quantile(ratio, .75) - np.quantile(ratio, .25)) / abs(med) if med else np.inf
            pos = ok & (x > 0) & (pm > 0)
            slope = r2 = np.nan
            if pos.sum() >= 50:
                lx, ly = np.log(x[pos]), np.log(pm[pos])
                if np.var(lx) > 0:
                    b, c0 = np.polyfit(lx, ly, 1)
                    slope, r2 = b, 1 - np.var(ly - (b * lx + c0)) / np.var(ly)
            res.append((spread, n, med, slope, r2, int(ok.sum())))
        res.sort()
        pmall = np.array([kv[f"{key}.sum"] / kv[f"{key}.wgt"] for kv in st.values() if kv.get(f"{key}.wgt") and f"{key}.sum" in kv])
        print(f"-- {key}: wgt median {wm:.4g} -> half-life {h:.4g} s; PM value q10/50/90 {np.quantile(pmall, .1):.5g} {np.median(pmall):.5g} {np.quantile(pmall, .9):.5g}")
        for spread, n, med, slope, r2, k in res[: a.top]:
            print(f"   {n:<18} PM/EMA(q) median {med:.6g}  IQR/median {spread:.4f}  log-log slope {slope:.3f} R2 {r2:.3f}  ({k})")
        if res:
            summary.append((res[0][0], key, h, res[0][1], res[0][2], res[0][3], res[0][4]))
    print("== summary, best candidate per key (spread < 0.01 = proportional; slope ~1 / R2 ~1 = power law fits):")
    for spread, key, h, n, med, slope, r2 in sorted(summary):
        tag = "PINNED?" if spread < 0.01 else "close" if spread < 0.05 else ""
        print(f"   {key:<18} h {h:>9.4g} s  {n:<18} x {med:<12.6g} IQR/med {spread:.4f} slope {slope:.3f} R2 {r2:.3f} {tag}")


if __name__ == "__main__":
    main()
