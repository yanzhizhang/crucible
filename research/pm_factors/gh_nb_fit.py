r"""Fit zzgh.nb / zzgh2.nb: candidate rules built from the day's trades on a 15 s and a 10 s slot grid, scored against PM's .jz.

PM's two nb values differ (by up to ~0.02) although both grids cover the same 09:30-14:57 window, so nb is built from
slot-level quantities, not from the day's total buy / sell amounts. For every candidate the share of stocks equal to PM at
1e-9 / 1e-6, the median |diff| and the correlation are printed for both terms. Read-only.

    python research/pm_factors/gh_nb_fit.py --date 20261120 --jz /work/crucible_data/seed_probe/HS300/20261120.jz --catalog /work/crucible_data/catalog_dysl2.duckdb
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

LN2 = math.log(2.0)
SESSION = 14220.0  # trading-clock seconds from 09:30 to 14:57


def imb(b, s):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b + s > 0, (b - s) / (b + s), np.nan)


def slot_ema(x: np.ndarray, step: float, half: float) -> float:
    """Day-end value of an average updated once per slot with x (weight 1, decayed by step / half per slot)."""
    d = np.exp(-(len(x) - 1 - np.arange(len(x))) * step * LN2 / half)
    return float((x * d).sum() / d.sum())


def candidates(t: np.ndarray, s: np.ndarray, amt: np.ndarray, vol: np.ndarray, side: np.ndarray, n: int,
               snap: np.ndarray | None = None, snap_new: np.ndarray | None = None) -> dict[str, float]:
    """nb candidates on an n-slot grid of the trading clock t (local receive), s = exchange seconds of day; snap = snapshot
    receive times on the same trading clock (snap_new = those with a new exchange stamp)."""
    step = SESSION / n
    out = {}
    for clk, tt, lo, hi in (("local", t, 0.0, SESSION), ("exch", s - 34200 - np.where(s >= 46800, 5400, 0), 0.0, SESSION)):
        m = (tt >= lo) & (tt < hi)
        k = np.floor(tt[m] / step).astype(int)
        b = np.bincount(k, weights=np.where(side[m] > 0, amt[m], 0.0), minlength=n)[:n]
        sl = np.bincount(k, weights=np.where(side[m] < 0, amt[m], 0.0), minlength=n)[:n]
        bv = np.bincount(k, weights=np.where(side[m] > 0, vol[m], 0.0), minlength=n)[:n]
        sv = np.bincount(k, weights=np.where(side[m] < 0, vol[m], 0.0), minlength=n)[:n]
        r = imb(b, sl)
        out[f"{clk}: total amount imbalance"] = float(imb(b.sum(), sl.sum()))
        out[f"{clk}: total volume imbalance"] = float(imb(bv.sum(), sv.sum()))
        out[f"{clk}: mean slot imbalance (traded slots)"] = float(np.nanmean(r)) if np.isfinite(r).any() else math.nan
        out[f"{clk}: mean slot imbalance (empty = 0)"] = float(np.nan_to_num(r).mean())
        out[f"{clk}: share buy-led minus sell-led slots"] = float(((b > sl).sum() - (sl > b).sum()) / n)
        out[f"{clk}: total imbalance, last slot dropped"] = float(imb(b[:-1].sum(), sl[:-1].sum()))
        out[f"{clk}: total imbalance, first slot dropped"] = float(imb(b[1:].sum(), sl[1:].sum()))
        for cut in range(-60, 1, 1):  # total imbalance of the trades before a fixed end, SESSION + cut
            mm = m & (tt < SESSION + cut)
            out[f"{clk}: total imbalance, trades before end {cut:+d} s"] = float(imb(amt[mm & (side > 0)].sum(), amt[mm & (side < 0)].sum()))
        if snap is not None:  # a slot is booked only once a later snapshot has arrived: drop trades in slots after the last snapshot's slot start
            for sn, st_ in (("last snapshot", snap), ("last snapshot with a new exchange stamp", snap_new)):
                last = st_[st_ < SESSION].max() if (st_ < SESSION).any() else 0.0
                for rule, cutoff in (("its slot start", np.floor(last / step) * step), ("its time", last)):
                    mm = m & (tt < cutoff)
                    out[f"{clk}: total imbalance, trades before the {sn} ({rule})"] = float(imb(amt[mm & (side > 0)].sum(), amt[mm & (side < 0)].sum()))
        if snap is not None:  # slots [0, k) at the stock's last evaluation time T: k = floor(T / step), minus 1 while T is in the first half of its slot
            ends = {"last snapshot": snap, "last snapshot with a new exchange stamp": snap_new, "last snapshot or trade": np.concatenate([snap, t])}
            for en, ev in ends.items():
                for lim_n, lim in (("before 14:57", SESSION), ("any time", np.inf)):
                    e = ev[ev < lim]
                    if not e.size:
                        continue
                    T = min(float(e.max()), SESSION)
                    for rule, kk in (("half-slot rule", int(np.floor(T / step)) - int(math.fmod(T, step) < step / 2)), ("floor", int(np.floor(T / step)))):
                        kk = max(0, min(kk, n))
                        out[f"{clk}: slots before k at the {en} ({lim_n}), {rule}"] = float(imb(b[:kk].sum(), sl[:kk].sum()))
        for h in (30, 60, 120, 300, 600, 900, 1800, 3600, 7200):
            out[f"{clk}: slot-EMA buy / sell amount, half {h} s"] = float(imb(slot_ema(b, step, h), slot_ema(sl, step, h)))
            out[f"{clk}: slot-EMA of slot imbalance (empty = 0), half {h} s"] = slot_ema(np.nan_to_num(r), step, h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--grids", default="zzgh=948,zzgh2=1422", help="term=slot count (the .jz's own zzgh.c / zzgh2.c lengths)")
    ap.add_argument("--pm-mp", type=Path, default=None, help="mp_load cache dir of the day: run on PM's own feed")
    a = ap.parse_args()
    import dcp_ema as E
    import duckdb
    from stream import universe_from_state

    with gzip.open(a.jz) as f:
        pm = {int(s): kv for s, kv in json.load(f).items()}
    sids, stop = universe_from_state(a.jz)
    con = duckdb.connect(str(a.catalog), read_only=True)
    if a.pm_mp:  # PM's own feed (mp_load cache) instead of the catalog
        import mp_load
        up = mp_load.load(a.pm_mp, sids, stop)
    else:
        up = E.load(con, int(a.date), sids, stop, extra=True)  # extra: buy / sell order seq ids
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    grids = [(x.split("=")[0], int(x.split("=")[1])) for x in a.grids.split(",")]
    print(f"== {a.date}: {len(sids)} stocks, {len(slow)} with trades")
    for term, n in grids:
        res = {}
        for sid in sids:
            g, p = slow.get(sid), pm.get(sid, {}).get(f"{term}.nb")
            if g is None or not isinstance(p, (int, float)):
                continue
            q = fast.get(sid)
            st_, nw = (None, None) if q is None else (q["t_local"].to_numpy().astype(float), q["s"].to_numpy().astype(float))
            st_new = None if q is None else st_[np.concatenate([[True], np.diff(nw) > 0])]
            c = candidates(g["t_local"].to_numpy().astype(float), g["s"].to_numpy().astype(float), g["amt"].to_numpy().astype(float),
                           g["vol"].to_numpy().astype(float), g["side"].to_numpy().astype(float), n, st_, st_new)
            for k, v in c.items():
                res.setdefault(k, []).append((v, float(p)))
        rows = []
        for k, vs in res.items():
            x, y = np.array(vs).T
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            d = np.abs(x[ok] - y[ok])
            rows.append((float(np.mean(d <= 1e-9)), float(np.mean(d <= 1e-6)), float(np.median(d)), float(np.corrcoef(x[ok], y[ok])[0, 1]),
                         float(np.mean(x[ok] - y[ok])), k))
        print(f"\n-- {term}.nb ({n} slots): share exact 1e-9 / 1e-6, median |diff|, corr, mean (ours - PM) -- best 14")
        for e9, e6, md, r, bias, k in sorted(rows, key=lambda z: (-z[0], -z[1], z[2]))[:14]:
            print(f"   {e9:.3f} {e6:.3f}  med {md:.2e}  corr {r:.5f}  bias {bias:+.5f}  {k}")
        side_variants(term, n, sids, slow, fast, pm)
    print("[done]")


def side_variants(term: str, n: int, sids: list[int], slow: dict, fast: dict, pm: dict) -> None:
    """The best window (local clock, last slot dropped) with other trade-side rules, split by market (SH 6xxxxx / SZ 0, 3xxxxx)."""
    step = SESSION / n
    res = {}
    for sid in sids:
        g, q, p = slow.get(sid), fast.get(sid), pm.get(sid, {}).get(f"{term}.nb")
        if g is None or q is None or not isinstance(p, (int, float)):
            continue
        t, px, amt, side = (g[c].to_numpy().astype(float) for c in ("t_local", "price", "amt", "side"))
        ls_t, ls_q = g["ls"].to_numpy(), q["ls"].to_numpy()
        j = np.searchsorted(ls_q, ls_t, side="right") - 1  # the last snapshot received before the trade
        b1 = np.where(j >= 0, q["b1"].to_numpy().astype(float)[np.maximum(j, 0)], np.nan)
        a1 = np.where(j >= 0, q["a1"].to_numpy().astype(float)[np.maximum(j, 0)], np.nan)
        mid = (b1 + a1) / 2
        prev = np.concatenate([[np.nan], px[:-1]])
        tick = np.sign(px - prev)
        for i in range(1, len(tick)):  # tick rule: an unchanged price keeps the previous trade's side
            if tick[i] == 0 or np.isnan(tick[i]):
                tick[i] = tick[i - 1]
        rules = {"feed side": side,
                 "quote rule (>= ask1 buy, <= bid1 sell, else 0)": np.where(px >= a1, 1.0, np.where(px <= b1, -1.0, 0.0)),
                 "mid rule (> mid buy, < mid sell, else tick)": np.where(px > mid, 1.0, np.where(px < mid, -1.0, np.nan_to_num(tick))),
                 "tick rule": np.nan_to_num(tick),
                 "feed side, unknown -> tick": np.where(side != 0, side, np.nan_to_num(tick))}
        if "bsq" in g.columns:  # r222: SH's unflagged trades take the later order's side (zztx.r SH 1.0000)
            bsq, ssq = g["bsq"].to_numpy().astype(float), g["ssq"].to_numpy().astype(float)
            rules["feed side, unknown -> later order"] = np.where(side != 0, side, np.where(bsq > ssq, 1.0, -1.0))
        mkt = "SH" if sid >= 600000 else "SZ"
        keep = (t > 0) & (t < (n - 1) * step)  # slots [0, n - 1): the best window so far, with the code's strict t > 0
        for rn, sd in rules.items():
            v = float(imb(amt[keep & (sd > 0)].sum(), amt[keep & (sd < 0)].sum()))
            res.setdefault((rn, mkt), []).append((v, float(p)))
            res.setdefault((rn, "all"), []).append((v, float(p)))
    print(f"   side rules on 'local, slots [0, n-1), t > 0' -- exact 1e-9 / 1e-6, median |diff| per market")
    for (rn, mkt), vs in sorted(res.items()):
        x, y = np.array(vs).T
        ok = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[ok] - y[ok])
        print(f"   {mkt:<3} {np.mean(d <= 1e-9):.3f} {np.mean(d <= 1e-6):.3f}  med {np.median(d):.2e}  n {ok.sum():>3}  {rn}")


if __name__ == "__main__":
    main()
