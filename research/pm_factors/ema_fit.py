r"""Fit the rule of any decayed average in PM's .jz: clock and half-life from its weight, then its input from its value.

PM's averages (engine/state.Ema) take weight 1 per update, decayed by exp(-dt ln2 / half) on the trading clock, so at the
last update (the key's ``.t``) wgt = sum_i exp(-k (t_last - t_i)) depends only on WHEN it updates: the clock (every
snapshot, snapshots whose price changed, every trade, both) and the half-life. With those fixed, sum = sum_i x_i e^(...)
picks the input x. Both are closed forms, vectorized per stock. For each --keys average: the best clock x start x half
by the share of stocks whose wgt equals PM's (rel 1e-4), then the best of ~20 inputs by the share whose sum / wgt equals
PM's (rel 1e-4). Read-only.

    python research/pm_factors/ema_fit.py --date 20261106 --jz /data/prod/h/HS300/p1_intrading_dcp/20261106.jz --raw /work/crucible_data/dysl2_raw
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

CST_NS = 8 * 3600 * 10**9
KEYS = "zzvm.c.re,zzmh.r,zzhj.p,zzcb.r,zzeb.r,zztx.r,zzui.r,zzup.ps,zzba.p,zzeg.c.rEma,zzqj.r,zzup.pf"
HALVES = [5, 10, 15, 20, 30, 45, 60, 90, 100, 120, 150, 180, 200, 240, 300, 360, 450, 600, 900, 1200, 1800, 3600, 7200, 14400, 28800, 57600, 86400, 172800]


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
    ap.add_argument("--keys", default=KEYS)
    ap.add_argument("--prev-jz", type=Path, default=None, help="yesterday's .jz: the value test also tries the average carried over undecayed (the engine's c0)")
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--by", default="wgt", choices=["wgt", "value"], help="value: rank clock x half by the best input's value match (rel 1e-3), ignoring the weight")
    a = ap.parse_args()
    d = a.date
    st = {int(s): kv for s, kv in json.loads(gzip.open(a.jz).read()).items()}
    pv = {int(s): kv for s, kv in json.loads(gzip.open(a.prev_jz).read()).items()} if a.prev_jz and a.prev_jz.exists() else {}
    sids = sorted(st)
    q = (pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")),
                         columns=["symbol_id", "time", "spider_ts", "price", "high", "low", "bid_px", "ask_px", "bid_vol", "ask_vol", "total_volume", "total_amount"])
         .filter(pl.col("symbol_id").is_in(sids) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts")
         .with_columns(pl.col("bid_px").cast(pl.List(pl.Float64)).list.first().alias("b1"), pl.col("ask_px").cast(pl.List(pl.Float64)).list.first().alias("a1"),
                       pl.col("bid_vol").cast(pl.List(pl.Int64)).list.first().alias("bv1"), pl.col("ask_vol").cast(pl.List(pl.Int64)).list.first().alias("av1"))
         .drop("bid_px", "ask_px", "bid_vol", "ask_vol"))
    tr = (pl.read_parquet(list(a.raw.glob(f"kind=transaction/date={d}/part-*.parquet")), columns=["symbol_id", "time", "spider_ts", "price", "volume", "trade_type", "dir"])
          .filter(pl.col("symbol_id").is_in(sids) & (pl.col("trade_type") == 1) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts"))
    qs = {s: g for (s,), g in q.group_by("symbol_id")}
    ts = {s: g for (s,), g in tr.group_by("symbol_id")}
    keys = [k for k in a.keys.split(",") if any(f"{k}.wgt" in kv for kv in st.values())]
    print(f"{d}: {len(sids)} stocks, {q.height:,} snapshots, {tr.height:,} trades; keys {keys}")

    def events(sid: int, clock: str, start: str) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Update times (trading clock, local arrival) and candidate inputs at them."""
        g = qs.get(sid)
        if g is None:
            return None
        lim = 33300 if start == "0925" else 34200
        out_t, xs = [], {}
        if clock in ("snap", "snapchg", "both"):
            sod = ((g["time"].to_numpy() + CST_NS) / 1e9) % 86400
            m = sod >= lim
            p = g["price"].to_numpy()
            if clock == "snapchg":
                m &= np.concatenate([[True], np.diff(p) != 0])
            b1, a1 = g["b1"].to_numpy().astype(float), g["a1"].to_numpy().astype(float)
            mid = np.where((b1 > 0) & (a1 > 0), (b1 + a1) / 2, p)
            lp = np.log(p)
            vol = np.diff(np.concatenate([[0], g["total_volume"].to_numpy()])).astype(float)
            amt = np.diff(np.concatenate([[0.0], g["total_amount"].to_numpy()]))
            cand = {"price": p, "mid": mid, "spread": a1 - b1, "spread/mid": (a1 - b1) / mid, "hl": g["high"].to_numpy() - g["low"].to_numpy(),
                    "logret": np.concatenate([[0.0], np.diff(lp)]), "|logret|": np.abs(np.concatenate([[0.0], np.diff(lp)])),
                    "logret2": np.concatenate([[0.0], np.diff(lp)]) ** 2, "dvol": vol, "damt": amt, "one": np.ones(p.size),
                    "bv1": g["bv1"].to_numpy().astype(float), "av1": g["av1"].to_numpy().astype(float), "depth1": (g["bv1"] + g["av1"]).to_numpy().astype(float),
                    "|dp|": np.abs(np.concatenate([[0.0], np.diff(p)]))}
            ex = sod
            hm = np.maximum.accumulate(np.where((ex >= 34200) & (ex < 53820), p, -np.inf))  # zzmh.hm: highest snapshot price 09:30-14:57
            hm = np.where(np.isfinite(hm), hm, np.nan)
            bv, av = g["bv1"].to_numpy().astype(float), g["av1"].to_numpy().astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                cand.update({"hm-p": hm - p, "(hm-p)/p": (hm - p) / p, "(hm-p)/hm": (hm - p) / hm, "100(hm/p-1)": 100 * (hm / p - 1),
                             "((hm-p)/p)^2": ((hm - p) / p) ** 2, "log(hm/p)": np.log(hm / p), "spread_ticks": (a1 - b1) / 0.01,
                             "depth1_lots": (bv + av) / 100, "log_depth1": np.log(bv + av), "mid/spread": mid / (a1 - b1),
                             "imb1": (bv - av) / (bv + av), "log_bv1/av1": np.log(bv / av), "p/hm": p / hm})
            out_t.append(tclock(g["spider_ts"].to_numpy())[m])
            for k, v in cand.items():
                xs.setdefault(k, []).append(v[m])
        if clock in ("trade", "both"):
            h = ts.get(sid)
            if h is not None:
                sod = ((h["time"].to_numpy() + CST_NS) / 1e9) % 86400
                m = sod >= lim
                p = h["price"].to_numpy()
                v = h["volume"].to_numpy().astype(float)
                sg = np.where(h["dir"].to_numpy() == 1063, 1.0, np.where(h["dir"].to_numpy() == 1550, -1.0, 0.0))
                lp = np.log(p)
                cand = {"price": p, "tqty": v, "tamt": p * v, "signed_qty": sg * v, "signed_amt": sg * p * v, "one": np.ones(p.size),
                        "logret": np.concatenate([[0.0], np.diff(lp)]), "|logret|": np.abs(np.concatenate([[0.0], np.diff(lp)]))}
                out_t.append(tclock(h["spider_ts"].to_numpy())[m])
                for k, x in cand.items():
                    xs.setdefault(k, []).append(x[m])
        if not out_t:
            return None
        t = np.concatenate(out_t)
        o = np.argsort(t, kind="mergesort")
        return t[o], {k: np.concatenate(v)[o] for k, v in xs.items() if sum(len(z) for z in v) == t.size}

    cache = {}
    for key in keys:
        pm = {s: (kv[f"{key}.sum"], kv[f"{key}.wgt"], kv.get(f"{key}.t")) for s, kv in st.items() if kv.get(f"{key}.wgt")}
        res = []
        for clock in ("snap", "snapchg", "trade", "both"):
            for start in ("0925", "0930"):
                for half in HALVES:
                    k = math.log(2) / half
                    ok = n = 0
                    for s, (_, w, tl) in pm.items():
                        e = cache.get((s, clock, start))
                        if e is None:
                            e = cache[(s, clock, start)] = events(s, clock, start) or (np.empty(0), {})
                        t = e[0]
                        if tl is not None:
                            t = t[t <= tl + 1e-6]
                        if not t.size:
                            continue
                        n += 1
                        ok += abs(np.exp(-k * (t[-1] - t)).sum() - w) <= 1e-4 * w
                    if n:
                        res.append((ok / n, clock, start, half))
        if a.by == "value":  # the weight depends on the feed's update count (DataYes vs PM's colo feed); the value much less
            vres = []
            for clock in ("snap", "snapchg", "trade", "both"):
                for start in ("0930",):
                    for half in HALVES:
                        k = math.log(2) / half
                        hit: dict[str, int] = {}
                        n = 0
                        for s, (sm, w, tl) in pm.items():
                            t, xs = cache[(s, clock, start)]
                            m = t <= (tl if tl is not None else np.inf) + 1e-6
                            if not m.any():
                                continue
                            n += 1
                            wt = np.exp(-k * (t[m][-1] - t[m]))
                            tgt = sm / w
                            kp = pv.get(s, {})
                            s0, w0 = (kp.get(f"{key}.sum", 0.0), kp.get(f"{key}.wgt", 0.0)) if kp.get(f"{key}.wgt") else (0.0, 0.0)
                            for name, x in xs.items():
                                xm = x[m]
                                f = np.isfinite(xm)
                                if not f.any():
                                    continue
                                v = float((xm[f] * wt[f]).sum() / wt[f].sum())
                                if w0:  # carried: yesterday's sum / wgt enter before the first update and decay with it
                                    d0 = math.exp(-k * (t[m][f][-1] - t[m][f][0]))
                                    vc = float(((xm[f] * wt[f]).sum() + s0 * d0) / (wt[f].sum() + w0 * d0))
                                    hit[name + "|c0"] = hit.get(name + "|c0", 0) + (abs(vc - tgt) <= 1e-3 * max(abs(tgt), 1e-12))
                                hit[name] = hit.get(name, 0) + (abs(v - tgt) <= 1e-3 * max(abs(tgt), 1e-12))
                        if n and hit:
                            nm, h = max(hit.items(), key=lambda kv: kv[1])
                            vres.append((h / n, clock, start, half, nm))
            vres.sort(reverse=True)
            print(f"-- {key}: {len(pm)} stocks; best by value equal (rel 1e-3): " + "  ".join(f"{c}/{h}s {nm} {f:.3f}" for f, c, _, h, nm in vres[: a.top]))
            continue
        res.sort(reverse=True)
        print(f"-- {key}: {len(pm)} stocks; best clocks by wgt equal: " + "  ".join(f"{c}/{s}/{h}s {f:.3f}" for f, c, s, h in res[: a.top]))
        if not res or res[0][0] < 0.2:
            continue
        _, clock, start, half = res[0]
        k = math.log(2) / half
        score = {}
        for s, (sm, w, tl) in pm.items():
            t, xs = cache[(s, clock, start)]
            m = t <= (tl if tl is not None else np.inf) + 1e-6
            if not m.any():
                continue
            wt = np.exp(-k * (t[m][-1] - t[m]))
            for name, x in xs.items():
                v = float((x[m] * wt).sum() / wt.sum())
                tgt = sm / w
                score.setdefault(name, []).append(abs(v - tgt) <= 1e-4 * max(abs(tgt), 1e-12))
        best = sorted(((float(np.mean(v)), n) for n, v in score.items()), reverse=True)
        print(f"   inputs at {clock}/{start}/{half}s, value equal: " + "  ".join(f"{n} {f:.3f}" for f, n in best[:6]))


if __name__ == "__main__":
    main()
