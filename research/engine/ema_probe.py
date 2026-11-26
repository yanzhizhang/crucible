r"""Pin the clock and half-life of PM averages from their weight alone, then try inputs for the ones with a lead.

PM's average state is sum / wgt / t with weight 1 per update decayed by exp(-dt ln2 / half). The weight does not depend on
the input, so matching it pins the clock (which events update the average) and the half-life without guessing the input.
For every target key and every (clock, half-life, stop) the day-end weight is rebuilt per stock as the sum of
exp(-(T - t_i) ln2 / half) over the updates t_i up to the stop (T = the last of them) and compared with PM's .wgt:
  clocks: snap (every snapshot received), trade (every trade), both (snapshots and trades), each on the local trading clock;
          grid n s (an update every n s of the trading clock with the last snapshot's value; ", new snapshot" = only steps
          a snapshot arrived in) -- the runtime's cfg.json gives hl per term, and hl / step fixes the weight, so a weight fit
          at half h on the 3 s snapshot clock is also hl = h x n / 3 on an n s grid;
  stops : PM's own .t of the stock (where the average stopped, whatever the rule), and the day's stop;
then, where the weight matches, input candidates are scored on the value sum / wgt (exact share at 1e-6, median PM / ours):
  snapshot-clock keys (zzhj.p, zzui.r, zzcb.r, zzeb.r): price, mid, 1 / price, amount / volume since the previous snapshot,
  cumulative amount / volume, amount per minute, snapshot VWAP, amount over the last 30 / 60 / 180 / 300 s, level-1 sizes,
  top imbalance, spread (ticks, / mid), and 0/1 indicators (volume / price / mid unchanged), at half-lives near R200's weights;
  zzvm.c.re  trade price, running VWAP from 09:30. Besides the exact share, the tightest PM / ours ratios are printed: a
  ratio constant across stocks is a scale factor on the right input. Clocks also come in event-set variants (snapshots with a
  new exchange stamp / a volume change / from 09:30; trades from 09:30 / 09:30-14:57).
Read-only.

    python research/engine/ema_probe.py --date 20261120 --jz /work/crucible_data/seed_probe/HS300/20261120.jz --catalog /work/crucible_data/catalog_s224.duckdb
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

LN2 = math.log(2.0)
HALVES = (1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30, 40, 45, 50, 60, 90, 100, 120, 150, 180, 200, 240, 300, 450, 600, 900, 1200, 1800,
          2700, 3600, 5400, 7200)
# value candidates: R200's weight fits plus the runtime cfg.json's hl of each term (zzhj 300, zzui 10, zzcb 30, zzeb 100, zztx 100, zzvm 300)
VHALVES = {"zzhj.p": (30, 300), "zzui.r": (3, 10), "zzcb.r": (30,), "zzeb.r": (90, 100), "zztx.r": (100,), "zzvm.c.re": (300,)}
GRIDS = (5, 10, 15, 20, 30, 60)  # fixed-step clocks on the trading clock: an average updated every n s with the last snapshot's value
VCLOCKS = ("snap", "grid 10 s", "grid 30 s")  # --value-clocks: the clocks the snapshot-key inputs are averaged on
KEYS = ("zzhj.p", "zzui.r", "zzvm.c.re", "zzcb.r", "zzeb.r", "zztx.r")


def end_state(t: np.ndarray, x: np.ndarray, half: float, stop: float) -> tuple[float, float, float]:
    """(sum, wgt, last update time) of the average after the updates at t <= stop (NaN inputs skipped)."""
    m = (t <= stop) & np.isfinite(x)
    if not m.any():
        return math.nan, math.nan, math.nan
    tt, xx = t[m], x[m]
    d = np.exp(-(tt[-1] - tt) * LN2 / half)
    return float((xx * d).sum()), float(d.sum()), float(tt[-1])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--keys", default=",".join(KEYS))
    ap.add_argument("--value-clocks", default=";".join(VCLOCKS), help="';'-separated clock names, e.g. 'snap;grid 10 s;snap, exchange clock'")
    a = ap.parse_args()
    vclocks = [c.strip() for c in a.value_clocks.split(";") if c.strip()]
    import dcp_ema as E
    import duckdb
    from stream import universe_from_state

    with gzip.open(a.jz) as f:
        pm = {int(s): kv for s, kv in json.load(f).items()}
    keys = [k for k in a.keys.split(",") if k]
    sids, stop = universe_from_state(a.jz)
    con = duckdb.connect(str(a.catalog), read_only=True)
    up = E.load(con, int(a.date), sids, stop)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    stop_tc = (stop - 34200) if stop <= 41400 else (7200.0 if stop < 46800 else stop - 46800 + 7200)
    print(f"== {a.date}: {len(sids)} stocks, stop {stop:.3f} local (trading clock {stop_tc:.3f})")
    wres = {k: {} for k in keys}
    vres = {k: {} for k in keys}
    for sid in sids:
        gf, gs = fast.get(sid), slow.get(sid)
        if gf is None or gs is None:
            continue
        tf, ts = gf["t_local"].to_numpy().astype(float), gs["t_local"].to_numpy().astype(float)
        sf = gf["s"].to_numpy().astype(float)
        sx = gf["t_exch"].to_numpy().astype(float) if "t_exch" in gf.columns else sf
        cvv = gf["cv"].to_numpy().astype(float)
        newx = np.concatenate([[True], np.diff(sf) > 0])  # the exchange stamp moved on
        newv = np.concatenate([[True], np.diff(cvv) != 0])  # the volume moved on
        ss0 = gs["s"].to_numpy().astype(float)
        clocks = {"snap": tf, "snap, exchange clock": sx, "trade, exchange clock": gs["t_exch"].to_numpy().astype(float), "snap, new exchange stamp": tf[newx], "snap, volume changed": tf[newv], "snap from 09:30": tf[sf >= 34200],
                  "trade": ts, "trade from 09:30": ts[ss0 >= 34200], "trade 09:30-14:57": ts[(ss0 >= 34200) & (ss0 < 53820)],
                  "both": np.sort(np.concatenate([tf, ts]), kind="stable")}
        at = {}  # grid clock -> index of the last snapshot at or before each grid time (to sample snapshot inputs on it)
        for n in GRIDS:
            g = np.arange(n, stop_tc + n, n, dtype=float)
            g = g[g >= tf[0]] if tf.size else g[:0]
            j = np.searchsorted(tf, g, side="right") - 1
            clocks[f"grid {n} s"], at[f"grid {n} s"] = g, j
            fresh = np.concatenate([[True], np.diff(j) > 0])  # a snapshot arrived inside the step
            clocks[f"grid {n} s, new snapshot"], at[f"grid {n} s, new snapshot"] = g[fresh], j[fresh]
        px = gf["price"].to_numpy().astype(float)
        px = np.where(px > 0, px, np.nan)
        b1, a1 = gf["b1"].to_numpy().astype(float), gf["a1"].to_numpy().astype(float)
        mid = np.where((b1 > 0) & (a1 > 0), (b1 + a1) / 2, px)
        ca = gf["ca"].to_numpy().astype(float)
        da = np.diff(ca, prepend=0.0)
        roll = {}
        for n in (3, 6, 9, 12, 15, 18, 20, 21, 24, 27, 30, 45, 60, 180, 300):
            j = np.searchsorted(sf, sf - n, side="right") - 1
            roll[n] = ca - np.where(j >= 0, ca[np.maximum(j, 0)], 0.0)
        tp = gs["price"].to_numpy().astype(float)
        tp = np.where(tp > 0, tp, np.nan)
        ss_ = gs["s"].to_numpy().astype(float)
        cv, cam = np.cumsum(np.where(ss_ >= 34200, gs["vol"].to_numpy(), 0.0)), np.cumsum(np.where(ss_ >= 34200, gs["amt"].to_numpy(), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(cv > 0, cam / cv, np.nan)
        dv = np.diff(cvv, prepend=0.0)
        slots = {}  # amount in the current / the last complete slot of the exchange clock from 09:30 (zzfb's 20 s, zzgh's 15 s grids)
        for w in (15, 20, 30, 60):
            k_ = np.floor((sf - 34200) / w)
            start = 34200 + k_ * w
            j0 = np.searchsorted(sf, start, side="left") - 1
            ca_start = np.where(j0 >= 0, ca[np.maximum(j0, 0)], 0.0)
            jp = np.searchsorted(sf, start - w, side="left") - 1
            ca_prev = np.where(jp >= 0, ca[np.maximum(jp, 0)], 0.0)
            slots[f"amount in the current {w} s slot"] = ca - ca_start
            slots[f"amount in the last complete {w} s slot"] = ca_start - ca_prev
        dt = np.diff(sf, prepend=np.nan)
        bv, av = gf["bv1"].to_numpy().astype(float), gf["av1"].to_numpy().astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snapc = {"snapshot price": px, "mid": mid, "1 / price": 1.0 / px, "amount since previous snapshot": da, "volume since previous snapshot": dv,
                     "cumulative amount": ca, "cumulative volume": cvv, "amount per minute (da / dt x 60)": da / dt * 60, "snapshot VWAP (da / dv)": np.where(dv > 0, da / dv, np.nan),
                     **{f"amount over last {n} s": v for n, v in roll.items()}, **slots, "bid1 size": bv, "ask1 size": av, "bid1 + ask1 size": bv + av,
                     "top imbalance": (bv - av) / (bv + av), "spread in ticks": np.round((a1 - b1) / 0.01, 6), "spread / mid": (a1 - b1) / mid, "tick / price": 0.01 / px, "1 / price^2": 1.0 / px ** 2,
                     "volume since previous / price": dv / px, "bid1 + ask1 size / price": (bv + av) / px, "amount since previous / price^2": da / px ** 2,
                     "volume unchanged (0/1)": (dv == 0).astype(float), "price unchanged (0/1)": np.concatenate([[0.0], (np.diff(px) == 0).astype(float)]),
                     "mid unchanged (0/1)": np.concatenate([[0.0], (np.diff(mid) == 0).astype(float)])}
        inputs = {k: [(c, snapc) for c in vclocks] for k in ("zzhj.p", "zzui.r", "zzcb.r", "zzeb.r", "zztx.r")}
        inputs["zzvm.c.re"] = [("trade", {"trade price": tp, "running VWAP from 09:30": vwap})]
        kv = pm.get(sid, {})
        for k in keys:
            w_pm, t_pm, s_pm = kv.get(f"{k}.wgt"), kv.get(f"{k}.t"), kv.get(f"{k}.sum")
            if not isinstance(w_pm, (int, float)) or not w_pm:
                continue
            stops = {"PM's own t": (t_pm + 1e-6) if isinstance(t_pm, (int, float)) else stop_tc, "day stop": stop_tc}
            for cn, t in clocks.items():
                for sn, st in stops.items():
                    for h in HALVES:
                        _, w, T = end_state(t, np.ones(len(t)), h, st)
                        r = wres[k].setdefault((cn, sn, h), [0, 0, []])
                        r[0] += 1
                        r[1] += abs(w - w_pm) <= 1e-7 * abs(w_pm)
                        r[2].append(w / w_pm)
            if k in inputs and isinstance(s_pm, (int, float)):
              for cn, cand in inputs[k]:
                t = clocks[cn]
                for nm, x in cand.items():
                    xs = x[at[cn]] if cn in at else x  # a grid clock samples the last snapshot's value
                    for sn, st in stops.items():
                        for h in (VHALVES.get(k) or HALVES):
                            s, w, _ = end_state(t, xs, h, st)
                            v, vp = s / w if w else math.nan, s_pm / w_pm
                            r = vres[k].setdefault((f"{nm} @ {cn}", sn, h), [0, 0, []])
                            r[0] += 1
                            r[1] += abs(v - vp) <= 1e-6 * max(1.0, abs(vp))
                            r[2].append(vp / v if v else math.nan)
    for k in keys:
        rows = sorted(((r[1] / max(r[0], 1), float(np.nanmedian(r[2])), key) for key, r in wres[k].items()), key=lambda x: (-x[0], abs(math.log(abs(x[1]))) if x[1] and x[1] == x[1] else 9))
        print(f"\n-- {k} weight: share of stocks with PM's wgt exactly (1e-7), median ours / PM -- best 6 of {len(rows)} (clock, stop, half-life s)")
        for sh, med, (cn, sn, h) in rows[:6]:
            print(f"   {sh:.3f}  median {med:.6f}  {cn:<6} {sn:<11} half {h:g}")
        if vres[k]:
            def spread(v):
                v = np.asarray(v, dtype=float)
                v = v[np.isfinite(v) & (v != 0)]
                if v.size < 10:
                    return math.nan, math.nan, math.nan
                q10, q50, q90 = np.percentile(v, [10, 50, 90])
                return float(q50), float(q10), float(q90)
            vr = []
            for key, r in vres[k].items():
                q50, q10, q90 = spread(r[2])
                disp = abs(math.log(q90 / q10)) if q10 and q90 and q10 > 0 and q90 > 0 else 9.0
                vr.append((r[1] / max(r[0], 1), disp, q50, q10, q90, key))
            print(f"   value sum / wgt: share exact (1e-6); PM / ours median and q10 / q90 -- best 6 by exact share, then the 8 with the tightest PM / ours (a constant factor)")
            for sh, disp, q50, q10, q90, (nm, sn, h) in sorted(vr, key=lambda x: (-x[0], x[1]))[:6]:
                print(f"   {sh:.3f}  ratio {q50:.6g} ({q10:.6g} .. {q90:.6g})  {nm:<34} {sn:<11} half {h:g}")
            for sh, disp, q50, q10, q90, (nm, sn, h) in sorted(vr, key=lambda x: x[1])[:8]:
                print(f"   tight  ratio {q50:.6g} ({q10:.6g} .. {q90:.6g})  exact {sh:.3f}  {nm:<34} {sn:<11} half {h:g}")
    print("[done]")


if __name__ == "__main__":
    main()
