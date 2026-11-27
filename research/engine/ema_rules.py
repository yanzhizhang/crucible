r"""Replay four averages as the runtime's update slots read (zzhj.p, zztx.r, zzcb.r, zzeb.r) and score sum / wgt / t against PM's .jz.

Each average is the shared half-life rule (weight 1 per update, decay exp(-ln2 dt / hl)), so its day-end state is rebuilt
in closed form from the update times and inputs. Rules (clock = trading clock of the chosen stamp, updates only at t > 0):
  zzhj.p  on the first snapshot of every new step-s slot (k = floor(t / step) > the last k): x = cumulative amount - its
          value at the previous update (0 before the first), t = that snapshot's clock;
  zztx.r  on every trade: buy / sell amount and volume accumulate by the trade's own side, then, once both volumes are
          positive, x = sell VWAP - buy VWAP (cumulative), t = the trade's clock;
  zzcb.r  on every snapshot once the cumulative SELL amount is non-zero: S = running sum of a level-1 size (> 0 only),
          x = S x level-1 price / cumulative sell amount (bid and ask variants);
  zzeb.r  on every snapshot: first the 100 s mid average (zzeb, pinned) takes the mid, then p = mid / that average's sum
          (or its mean), x = -p log2 p.
Read-only.

    python research/engine/ema_rules.py --date 20261120 --jz /work/crucible_data/seed_probe/HS300/20261120.jz --catalog /work/crucible_data/catalog_dysl2.duckdb
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


def end_state(t: np.ndarray, x: np.ndarray, half: float) -> tuple[float, float, float]:
    """(sum, wgt, t) after updates (t_i, x_i) in order; t runs through max() as the runtime keeps it; NaN x skipped."""
    m = np.isfinite(x) & np.isfinite(t)
    if not m.any():
        return math.nan, math.nan, math.nan
    tt = np.maximum.accumulate(t[m])
    d = np.exp(-(tt[-1] - tt) * LN2 / half)
    return float((x[m] * d).sum()), float(d.sum()), float(tt[-1])


def running_sum(t: np.ndarray, x: np.ndarray, half: float) -> np.ndarray:
    """The average's sum right after each update (for zzeb's second stage)."""
    out, s, last = np.empty(len(t)), 0.0, None
    for i, (ti, xi) in enumerate(zip(t, x)):
        if last is not None:
            s *= math.exp(-(max(ti, last) - last) * LN2 / half)
        last = ti if last is None else max(ti, last)
        s += xi
        out[i] = s
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--match", default="", help="print only rows whose 'key variant' matches this regex")
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
        up = E.load(con, int(a.date), sids, stop, extra=True)  # extra: buy / sell order seq ids, for the side-0 variants
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    res = {}

    def score(key: str, variant: str, sid: int, got: tuple[float, float, float]) -> None:
        kv = pm.get(sid, {})
        ref = [kv.get(f"{key}.{f}") for f in ("sum", "wgt", "t")]
        if not all(isinstance(r, (int, float)) for r in ref):
            return
        for var in (variant, f"{variant} | {'SH' if sid >= 600000 else 'SZ'}"):
            r = res.setdefault((key, var), {"n": 0, "sum": 0, "wgt": 0, "t": 0, "ratio": [], "w": []})
            r["n"] += 1
            for f, g_, p_ in zip(("sum", "wgt", "t"), got, ref):
                r[f] += abs(g_ - p_) <= 1e-9 * max(1.0, abs(p_))
            r["ratio"].append(ref[0] / ref[1] / (got[0] / got[1]) if got[0] and got[1] and ref[1] else math.nan)
            r["w"].append(ref[1] / got[1] if got[1] else math.nan)

    for sid in sids:
        q, g = fast.get(sid), slow.get(sid)
        if q is None or g is None:
            continue
        ca = q["ca"].to_numpy().astype(float)
        b1, a1, px = (q[c].to_numpy().astype(float) for c in ("b1", "a1", "price"))
        bv1, av1 = q["bv1"].to_numpy().astype(float), q["av1"].to_numpy().astype(float)
        mid = np.where((b1 > 0) & (a1 > 0) & (a1 >= b1), (b1 + a1) / 2, np.where(a1 > 0, a1, np.where(b1 > 0, b1, px)))
        tp, tv, ta, sd = (g[c].to_numpy().astype(float) for c in ("price", "vol", "amt", "side"))
        ls_q, ls_t = q["ls"].to_numpy(), g["ls"].to_numpy()
        for clk in ("t_local", "t_exch"):
            tq, tt = q[clk].to_numpy().astype(float), g[clk].to_numpy().astype(float)
            # zzhj.p
            for step in (30.0, 45.0, 20.0):
                ok = tq > 0
                k = np.floor(tq / step)
                prev_max = np.concatenate([[-1.0], np.maximum.accumulate(np.where(ok, k, -1.0))[:-1]])  # the last slot updated before
                first = ok & (k > prev_max) & (tq < 14220.0)  # slot index < count: no updates after 14:57
                idx = np.flatnonzero(first)
                for init, x0 in (("prev 0", 0.0),) if clk == "t_local" else ():
                    x = np.diff(np.concatenate([[x0], ca[idx]]))
                    for h in sorted({300.0, 10 * step}):  # weight fits step / hl = 0.1
                        score("zzhj.p", f"{clk}, step {step:g}, half {h:g}, {init}", sid, end_state(tq[idx], x, h))
            if clk == "t_local":  # zzhj.p variant: events are snapshots and trades, cumulative = the trades' own running amount
                tc_all = np.cumsum(ta)
                cum_q = np.concatenate([[0.0], tc_all])[np.searchsorted(ls_t, ls_q, side="right")]
                ev_t = np.concatenate([tq, tt])
                ev_c = np.concatenate([cum_q, tc_all])
                ev_ls = np.concatenate([ls_q, ls_t])
                o = np.argsort(ev_ls, kind="stable")
                ev_t, ev_c = ev_t[o], ev_c[o]
                for evn, keep_ev in (("snapshots + trades", np.ones(len(ev_t), bool)), ("snapshots only", np.concatenate([np.ones(len(tq), bool), np.zeros(len(tt), bool)])[o])):
                    et, ec = ev_t[keep_ev], ev_c[keep_ev]
                    ok2 = et > 0
                    k2 = np.floor(et / 30.0)
                    pm2 = np.concatenate([[-1.0], np.maximum.accumulate(np.where(ok2, k2, -1.0))[:-1]])
                    i2 = np.flatnonzero(ok2 & (k2 > pm2))
                    x = np.diff(np.concatenate([[0.0], ec[i2]]))
                    score("zzhj.p", f"{clk}, step 30, half 300, {evn}, trade-built cumulative", sid, end_state(et[i2], x, 300.0))
            # zztx.r
            okt = tt > 0
            bq, sq = np.cumsum(np.where(okt & (sd > 0), tv, 0.0)), np.cumsum(np.where(okt & (sd < 0), tv, 0.0))
            ba, sa = np.cumsum(np.where(okt & (sd > 0), ta, 0.0)), np.cumsum(np.where(okt & (sd < 0), ta, 0.0))
            both = okt & (bq > 0) & (sq > 0)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = np.where(both, sa / sq - ba / bq, np.nan)
            for h in (100.0, 60.0):
                score("zztx.r", f"{clk}, half {h:g}", sid, end_state(tt, x, h))
            # zzcb.r: cumulative sell amount as of each snapshot, by receive order
            j = np.searchsorted(ls_t, ls_q, side="right")
            sell_cum = np.concatenate([[0.0], np.cumsum(np.where(okt & (sd < 0), ta, 0.0))])[j]
            okq = tq > 0
            vols = {"ask1": av1, "bid1": bv1, "ask1 + bid1": av1 + bv1}
            pxs = {"ask1": a1, "bid1": b1, "mid": mid, "last": px}
            for vn, vol1 in vols.items() if clk == "t_local" else ():  # r218: the local clock fits better
                for from_n, base in (("S from t > 0", okq), ("S from all snapshots", np.ones_like(okq))):
                    S = np.cumsum(np.where(base & (vol1 > 0), vol1, 0.0))
                    for pn, px1 in pxs.items():
                        with np.errstate(divide="ignore", invalid="ignore"):
                            x = np.where(okq & (sell_cum != 0), S * px1 / sell_cum, np.nan)
                        score("zzcb.r", f"{clk}, size {vn}, price {pn}, {from_n}", sid, end_state(tq, x, 30.0))
            if clk == "t_local":  # side variants: trades with no side flag (0) counted as sells / buys; SH is where r220 is off
                n0 = int((okt & (sd == 0)).sum())
                r0 = res.setdefault(("side0", "SH" if sid >= 600000 else "SZ"), {"n": 0, "sum": 0, "wgt": 0, "t": 0, "ratio": []})
                r0["n"] += 1
                r0["ratio"].append(n0)
                tick = np.sign(np.diff(tp, prepend=np.nan))
                for i_ in range(1, len(tick)):  # tick rule: an unchanged price keeps the previous trade's sign
                    if not tick[i_] or np.isnan(tick[i_]):
                        tick[i_] = tick[i_ - 1]
                variants = [("side 0 -> sell", np.where(sd == 0, -1.0, sd)), ("side 0 -> buy", np.where(sd == 0, 1.0, sd)),
                            ("side 0 -> tick rule", np.where(sd == 0, np.nan_to_num(tick), sd))]
                if "bsq" in g.columns:  # the later of the two orders is the aggressor (SZ's rule, applied to SH's unflagged trades)
                    bsq, ssq = g["bsq"].to_numpy().astype(float), g["ssq"].to_numpy().astype(float)
                    variants.append(("side 0 -> later order", np.where(sd == 0, np.where(bsq > ssq, 1.0, -1.0), sd)))
                for sn, sd2 in variants:
                    bq2, sq2 = np.cumsum(np.where(okt & (sd2 > 0), tv, 0.0)), np.cumsum(np.where(okt & (sd2 < 0), tv, 0.0))
                    ba2, sa2 = np.cumsum(np.where(okt & (sd2 > 0), ta, 0.0)), np.cumsum(np.where(okt & (sd2 < 0), ta, 0.0))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        x = np.where(okt & (bq2 > 0) & (sq2 > 0), sa2 / sq2 - ba2 / bq2, np.nan)
                    score("zztx.r", f"{clk}, half 100, {sn}", sid, end_state(tt, x, 100.0))
                    sc2 = np.concatenate([[0.0], np.cumsum(np.where(okt & (sd2 < 0), ta, 0.0))])[j]
                    S = np.cumsum(np.where(okq & (av1 > 0), av1, 0.0))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        x = np.where(okq & (sc2 != 0), S * px / sc2, np.nan)
                    score("zzcb.r", f"{clk}, size ask1, price last, S from t > 0, {sn}", sid, end_state(tq, x, 30.0))
            # zzui: Kaufman adaptive average on the first snapshot of each new 10 s slot, then a 10 s average of it
            if clk == "t_local":
                kk = np.floor(tq / 10.0)
                pm10 = np.concatenate([[-1.0], np.maximum.accumulate(np.where(okq, kk, -1.0))[:-1]])
                ii = np.flatnonzero(okq & (kk > pm10) & np.isfinite(mid) & (tq < 14220.0))  # 1422 slots: none after 14:57
                xs, ts_ = mid[ii], tq[ii]
                for win, fast_n, slow_n in ((10, 2, 30), (10, 2, 20), (9, 2, 30)):
                    a_, b_ = 2 / (fast_n + 1) - 2 / (slow_n + 1), 2 / (slow_n + 1)
                    kama, ks = math.nan, np.empty(len(xs))
                    for i, x_ in enumerate(xs):
                        if i < win:
                            kama = x_ if math.isnan(kama) else kama
                        else:
                            path = np.abs(np.diff(xs[i - win:i + 1])).sum()
                            er = abs(x_ - xs[i - win]) / path if path > 0 else 0.0
                            kama += (er * a_ + b_) ** 2 * (x_ - kama)
                        ks[i] = kama
                    var = f"window {win}, fast {fast_n}, slow {slow_n}"
                    pk = pm.get(sid, {}).get("zzui.k")
                    if isinstance(pk, (int, float)) and ks.size:
                        r = res.setdefault(("zzui.k", var), {"n": 0, "sum": 0, "wgt": 0, "t": 0, "ratio": []})
                        r["n"] += 1
                        r["sum"] += abs(ks[-1] - pk) <= 1e-9 * max(1.0, abs(pk))
                        r["ratio"].append(pk / ks[-1] if ks[-1] else math.nan)
                    score("zzui.r", var, sid, end_state(ts_, ks, 10.0))
            # zzeb.r
            m = okq & np.isfinite(mid)
            s1 = running_sum(tq[m], mid[m], 100.0)
            w1 = running_sum(tq[m], np.ones(m.sum()), 100.0)
            for pn, p in (("p = mid / sum", mid[m] / s1), ("p = mid / mean", mid[m] / (s1 / w1))):
                with np.errstate(divide="ignore", invalid="ignore"):
                    x = -p * np.log2(p)
                for h in (100.0,):
                    score("zzeb.r", f"{clk}, {pn}, half {h:g}", sid, end_state(tq[m], x, h))
    print(f"== {a.date}: {len(sids)} stocks; exact share of sum / wgt / t (1e-9 rel), median PM mean / ours")
    import re
    for (key, var), r in sorted(res.items()):
        if a.match and not re.search(a.match, f"{key} {var}"):
            continue
        n = max(r["n"], 1)
        q10, q50, q90 = np.nanpercentile(r["ratio"], [10, 50, 90])
        w = f"  wgt PM / ours {np.nanmedian(r['w']):.5g}" if r.get("w") else ""
        print(f"   {key:<8} {'value exact' if key == 'zzui.k' else 'wgt'} {(r['sum'] if key == 'zzui.k' else r['wgt']) / n:.3f} t {r['t'] / n:.3f}"
              f"  ratio {q50:.6g} ({q10:.5g} .. {q90:.5g}){w}  {var}")
    print("[done]")


if __name__ == "__main__":
    main()
