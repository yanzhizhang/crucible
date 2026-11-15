r"""For fast averages (PM's weight says half-life ~1-5 s): which last-snapshot quantity is PM's value proportional to?

zzmh.r (weight ~1.14) and zzcb.r (~2.8) are close to their last input, so exact-match counting is fragile (one missed or
extra snapshot changes the value). Instead, per candidate quantity q at the last snapshot at or before the key's .t: the
ratio PM value / q across stocks -- its median and spread (IQR / median); a power-law fit log PM = a + b log q (slope,
R^2) catches squares and roots. A small spread / R^2 near 1 names the input. Read-only.

    python research/pm_factors/last_probe.py --date 20261106 --jz /data/prod/h/HS300/p1_intrading_dcp/20261106.jz --raw /work/crucible_data/dysl2_raw
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import polars as pl

CST_NS = 8 * 3600 * 10**9


def tclock(ns: np.ndarray) -> np.ndarray:
    """UTC epoch ns -> trading clock seconds (09:30 = 0, lunch removed)."""
    s = ((ns + CST_NS) / 1e9) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/dysl2_raw"))
    ap.add_argument("--keys", default="zzmh.r,zzcb.r,zzeb.r")
    a = ap.parse_args()
    st = {int(s): kv for s, kv in json.loads(gzip.open(a.jz).read()).items()}
    q = (pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={a.date}/part-*.parquet")),
                         columns=["symbol_id", "time", "spider_ts", "price", "pre_close", "open", "high", "low", "bid_px", "ask_px", "bid_vol", "ask_vol",
                                  "total_volume", "total_amount"])
         .filter(pl.col("symbol_id").is_in(list(st)) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts")
         .with_columns(pl.col("bid_px").cast(pl.List(pl.Float64)).list.first().alias("b1"), pl.col("ask_px").cast(pl.List(pl.Float64)).list.first().alias("a1"),
                       pl.col("bid_vol").cast(pl.List(pl.Int64)).list.first().alias("bv1"), pl.col("ask_vol").cast(pl.List(pl.Int64)).list.first().alias("av1")))
    for key in a.keys.split(","):
        rows = []
        for (sid,), g in q.group_by("symbol_id"):
            kv = st.get(int(sid), {})
            if not kv.get(f"{key}.wgt"):
                continue
            tl = kv.get(f"{key}.t", np.inf)
            t = tclock(g["spider_ts"].to_numpy())
            m = t <= tl + 1e-6
            if not m.any():
                continue
            i = int(np.nonzero(m)[0][-1])
            ex = ((g["time"].to_numpy() + CST_NS) / 1e9) % 86400
            p = g["price"].to_numpy()
            hm = np.max(np.where((ex[: i + 1] >= 34200) & (ex[: i + 1] < 53820), p[: i + 1], -np.inf))
            r = {c: float(g[c][i]) for c in ("price", "pre_close", "open", "high", "low", "b1", "a1", "bv1", "av1", "total_volume", "total_amount")}
            r["hm"], r["pm"], r["sid"] = hm, kv[f"{key}.sum"] / kv[f"{key}.wgt"], int(sid)
            rows.append(r)
        d = pl.DataFrame(rows)
        P, H, L, HM, PC, O, B, A = (pl.col(c) for c in ("price", "high", "low", "hm", "pre_close", "open", "b1", "a1"))
        cands = {"price": P, "hm-p": HM - P, "(hm-p)/p": (HM - P) / P, "(hm-p)/hm": (HM - P) / HM, "(hm-p)/pc": (HM - P) / PC, "high-low": H - L,
                 "(high-low)/p": (H - L) / P, "high-p": H - P, "p-low": P - L, "spread": A - B, "spread/mid": (A - B) / ((A + B) / 2), "|p-pc|": (P - PC).abs(),
                 "|p-pc|/pc": ((P - PC) / PC).abs(), "|p-open|": (P - O).abs(), "depth1": pl.col("bv1") + pl.col("av1"), "bv1": pl.col("bv1"), "av1": pl.col("av1"),
                 "vol": pl.col("total_volume"), "amt": pl.col("total_amount"), "pc": PC, "hm": HM, "p-pc": P - PC, "hm-pc": HM - PC}
        d = d.with_columns([e.alias(f"q.{n}") for n, e in cands.items()])
        pm = d["pm"].to_numpy()
        print(f"-- {a.date} {key}: {d.height} stocks; PM value q10/50/90 {np.quantile(pm, .1):.5g} {np.median(pm):.5g} {np.quantile(pm, .9):.5g}")
        res = []
        for n in cands:
            x = d[f"q.{n}"].to_numpy().astype(float)
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
                b, c0 = np.polyfit(lx, ly, 1)
                slope, r2 = b, 1 - np.var(ly - (b * lx + c0)) / np.var(ly)
            res.append((spread, n, med, slope, r2, int(ok.sum())))
        for spread, n, med, slope, r2, k in sorted(res)[:10]:
            print(f"   {n:<14} PM/q median {med:.6g}  IQR/median {spread:.4f}  log-log slope {slope:.3f} R2 {r2:.3f}  ({k})")


if __name__ == "__main__":
    main()
