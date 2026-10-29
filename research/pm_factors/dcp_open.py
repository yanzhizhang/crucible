r"""Stage 4f: the six day-end state keys still open, each against a fresh candidate list, one run.

Every candidate is computed per stock from our ticks up to PM's own stop and scored across stocks against
PM's value: Spearman ``rho``, the median ratio ``k`` (PM / ours), the IQR of log(PM / ours) (0 = same up
to a constant), and the share within 1e-4 as is and after scaling by ``k``. SZSE stocks only (PM's SSE
trade feed is lossy). Top rows per key.

* ``zzcb.r`` (snapshot EMA, 30 s) and ``zztx.r`` (trade EMA, 100 s): values from the tick size
  (0.01 / price, squared, spread in ticks), and for trades the prevailing quote at the trade --
  effective half-spread ``side (px - mid) / mid``, position inside the spread, signed price change,
  log(px / mid), log(px / vwap);
* ``zzba.p`` / ``zzba.r``: is the weight a count, not a time decay? ``wgt`` against the number of
  snapshots / price changes / volume moves of the day (a count-based EMA saturates at 1 / alpha);
  values at the stamp for half-lives 240 / 300 s against price / mid and the range variants;
* ``zzvx.a``: ratios >= 1 by construction -- high / low (snapshot fields, trades, mid), true range over
  range against the previous close / open, range over the open, and the same from the continuous session;
* ``zzgh.nb``: integer?; number of filled ``zzgh.c`` bins; trade-count / volume / amount imbalance over
  exchange 09:30-14:57, local time to the stop, with and without the auctions; resting-book imbalance;
* ``zzhj.p``: the day's own contribution (as ``diag_qh``: ``x = (sum - a sum_prev) / (wgt - a wgt_prev)``)
  against snapshot-amount flows (``total_amount`` deltas, which is what ``zzfb`` uses) over the last L
  seconds before each quarter-hour sample, on the local and the exchange clock, plus decayed flows.

    python research/pm_factors/dcp_open.py --dates 20260923,20260924,20260929,20260930,20261009
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
import dcp_ema as E  # tick loading and per-update arrays
import diag_qh as Q  # zzhj.p sampling times, weight fit
import feature_intraday as FI  # ema_at

SZSE = 3554
LN2 = np.log(2)
TICK = 0.01


def load_raw(path: Path) -> dict[str, dict[str, float]]:
    """Raw state: sid -> key -> value (histogram bins included)."""
    with gzip.open(path) as fh:
        return json.loads(fh.read())


def quote_at_trades(af: dict, asl: dict) -> dict[str, np.ndarray]:
    """Trade-level candidates from the last snapshot received before each trade."""
    k = np.searchsorted(af["t_local"], asl["t_local"], side="right") - 1
    ok = k >= 0
    k = np.clip(k, 0, None)
    mid, bid, ask = (np.where(ok, af[c][k], np.nan) for c in ("mid", "bid1", "ask1"))
    px, side = asl["price"], asl["side"]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff = side * (px - mid) / mid
        return {"eff": eff, "neff": -eff, "abs_eff": np.abs(eff), "lp_mid": np.log(px / mid), "nlp_mid": -np.log(px / mid),
                "pos_spr": (px - bid) / (ask - bid) - 0.5, "side_dp": side * asl["dp"], "side_dp_rel": side * asl["dp"] / px,
                "lp_vwap": np.log(px / asl["rvwap"]), "nlp_vwap": -np.log(px / asl["rvwap"]), "nabsret": -asl["absret"],
                "nlogret2": -asl["logret2"], "neg_dt": -asl["dt"] if "dt" in asl else np.full(px.size, np.nan)}


def fast_extra(af: dict, g: pl.DataFrame) -> dict[str, np.ndarray]:
    """Snapshot-level candidates: tick-size relatives and range ratios."""
    px, mid = af["price"], af["mid"]
    hi, lo = g["hi"].to_numpy().astype(float), g["lo"].to_numpy().astype(float)
    hi, lo = np.where(hi > 0, hi, np.nan), np.where(lo > 0, lo, np.nan)
    op, pc = g["op"].to_numpy().astype(float), g["pc"].to_numpy().astype(float)
    op, pc = np.where(op > 0, op, np.nan), np.where(pc > 0, pc, np.nan)
    cont = g["s"].to_numpy() >= 34200
    pcont = np.where(cont, px, np.nan)
    hic, loc = np.fmax.accumulate(np.nan_to_num(pcont, nan=-np.inf)), np.fmin.accumulate(np.nan_to_num(pcont, nan=np.inf))
    hic, loc = np.where(np.isfinite(hic), hic, np.nan), np.where(np.isfinite(loc), loc, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"tick_rel": TICK / px, "tick_rel2": (TICK / px) ** 2, "tick_mid": TICK / mid, "spr_ticks": af["spread_abs"] / TICK,
                "spr_ticks2": (af["spread_abs"] / TICK) ** 2, "spread2": af["spread"] ** 2, "half_spread": af["spread"] / 2,
                "hl_ratio": hi / lo, "hl_ratio_c": hic / loc, "lhl_ratio": np.log(hi / lo),
                "tr_pc_over_hl": (np.fmax(hi, pc) - np.fmin(lo, pc)) / (hi - lo), "tr_op_over_hl": (np.fmax(hi, op) - np.fmin(lo, op)) / (hi - lo),
                "hl_over_hlc": (hi - lo) / (hic - loc), "hi_over_op": hi / op, "op_over_lo": op / lo,
                "hl_rel_open": (hi - lo) / op, "hl_rel_pc": (hi - lo) / pc, "hl_rel_lo": (hi - lo) / lo,
                "mid_hi_lo": mid * 2 / (hi + lo), "px_over_lo": px / lo, "hi_over_px": hi / px}


def _put(per: dict, key: str, cands: dict[str, float], pmv: float) -> None:
    """Append one stock's PM value and candidate values under key."""
    dk = per.setdefault(key, {})
    dk.setdefault("pm", []).append(pmv)
    for c, v in cands.items():
        dk.setdefault(c, []).append(v)


def ema_value(a: dict, x: np.ndarray, half: float, stop: float) -> float:
    """EMA (weight 1 per update) of x as of stop."""
    return float(FI.ema_at({"t_local": a["t_local"], "x": x}, "x", half, np.array([stop]))[0])


def score(rows: list[dict], key: str, cands: dict[str, np.ndarray], pm: np.ndarray, extra: dict | None = None) -> None:
    """Score every candidate against PM's values (aligned arrays over SZSE stocks) into rows."""
    pm = np.asarray(pm, float)
    for name, v in cands.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v) & np.isfinite(pm) & (v != 0)
        if ok.sum() < 20:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = np.log(np.abs(pm[ok] / v[ok]))
            k = float(np.nanmedian(pm[ok] / v[ok]))
            rk = lambda z: np.argsort(np.argsort(z)).astype(float)  # noqa: E731 -- ranks for Spearman
            rho = float(np.corrcoef(rk(pm[ok]), rk(v[ok]))[0, 1])
            e, es = np.abs(pm[ok] - v[ok]) / np.maximum(np.abs(pm[ok]), 1e-12), np.abs(pm[ok] - k * v[ok]) / np.maximum(np.abs(pm[ok]), 1e-12)
        rows.append({"key": key, "cand": name, "n": int(ok.sum()), "rho": rho, "k": k,
                     "lr_iqr": float(np.nanquantile(lr, 0.75) - np.nanquantile(lr, 0.25)) if np.isfinite(lr).sum() > 2 else np.nan,
                     "ok4": float(np.mean(e <= 1e-4)), "ok4_k": float(np.mean(es <= 1e-4)), "ok2_k": float(np.mean(es <= 1e-2)), **(extra or {})})


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4f"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    files = sorted(p.stem for p in args.dcp_dir.glob("20*.jz") if p.stem.isdigit())
    rows, qh = [], []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        if d not in files:
            print(f"skip {d}: no state file")
            continue
        cur = load_raw(args.dcp_dir / f"{d}.jz")
        prev = [f for f in files if f < d]
        pre = load_raw(args.dcp_dir / f"{prev[-1]}.jz") if prev else {}
        pm = B.load_state(args.dcp_dir / f"{d}.jz")
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(int(s) for s in cur)
        up = E.load(con, int(d), sids, stop)
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        per: dict[str, dict[str, list]] = {}  # key -> cand -> values; "pm" -> PM's value
        put = lambda key, cands, pmv, per=per: _put(per, key, cands, pmv)  # noqa: E731 -- this day's store
        n_used = 0
        for sid in sids:
            gf, gs, kv = fast.get(sid), slow.get(sid), cur[str(sid)]
            if gf is None or gs is None or gf.height < 3 or gs.height < 3 or int(gf["mkt"][0]) != SZSE:
                continue
            n_used += 1
            af, asl = E.arrays(gf, "fast"), E.arrays(gs, "slow")
            fx, tq = fast_extra(af, gf), quote_at_trades(af, asl)
            t_fast = kv.get("zzba.p.t")
            t_fast = (t_fast if t_fast is not None else stop)
            # zzcb.r: snapshot EMA 30 s
            if kv.get("zzcb.r.sum") is not None and kv.get("zzcb.r.wgt"):
                vals = {k_: af[k_] for k_ in ("spread", "spread_abs", "inv_p", "absret", "mlogret", "absdmid", "dt")} | fx
                put("zzcb.r", {c: ema_value(af, v, 30, t_fast) for c, v in vals.items()}, kv["zzcb.r.sum"] / kv["zzcb.r.wgt"])
            # zztx.r: trade EMA 100 s
            if kv.get("zztx.r.sum") is not None and kv.get("zztx.r.wgt"):
                vals = {k_: asl[k_] for k_ in ("ret", "logret", "absret", "logret2", "side", "svol", "samt", "lvol", "dp", "absdp", "dt")} | tq
                put("zztx.r", {c: ema_value(asl, v, 100, stop) for c, v in vals.items()}, kv["zztx.r.sum"] / kv["zztx.r.wgt"])
            # zzba.p / zzba.r: weight vs counts; value vs price / range at 240 / 300 s
            n_snap = int(gf.height)
            n_px = int(af["set_px_changed"].sum())
            n_vol = int(af["set_vol_moved"].sum())
            n_cont = int((gf["s"].to_numpy() >= 34200).sum())
            if kv.get("zzba.p.wgt"):
                put("zzba.p.wgt", {"n_snap": n_snap, "n_px_changed": n_px, "n_vol_moved": n_vol, "n_cont": n_cont, "one": 1.0,
                                   "n_trades": int(gs.height)}, kv["zzba.p.wgt"])
            if kv.get("zzba.p.sum") is not None and kv.get("zzba.p.wgt"):
                put("zzba.p", {f"{c}@{h}": ema_value(af, af[c], h, t_fast) for c in ("price", "mid", "bid1", "ask1") for h in (240, 300, 600, 1800, 3600)}
                    | {"last": float(af["price"][-1]), "mid_last": float(af["mid"][-1]), "vwap": float(asl["rvwap"][-1])}, kv["zzba.p.sum"] / kv["zzba.p.wgt"])
            if kv.get("zzba.r.sum") is not None and kv.get("zzba.r.wgt"):
                rng = {k_: af[k_] for k_ in ("hl", "tr_pc", "tr_open", "spread_abs")} | {k_: fx[k_] for k_ in ("hl_rel_open", "hl_rel_pc", "hl_rel_lo")}
                put("zzba.r", {f"{c}@{h}": ema_value(af, v, h, t_fast) for c, v in rng.items() for h in (240, 300, 600, 3600)}
                    | {"hl_last": float(af["hl"][-1]), "tr_pc_last": float(af["tr_pc"][-1]), "tr_open_last": float(af["tr_open"][-1])},
                    kv["zzba.r.sum"] / kv["zzba.r.wgt"])
            # zzvx.a: ratios >= 1
            if kv.get("zzvx.a.sum") is not None and kv.get("zzvx.a.wgt"):
                rat = {k_: fx[k_] for k_ in ("hl_ratio", "hl_ratio_c", "tr_pc_over_hl", "tr_op_over_hl", "hl_over_hlc", "hi_over_op",
                                             "op_over_lo", "px_over_lo", "hi_over_px", "mid_hi_lo")}
                put("zzvx.a", {f"{c}@{h}": ema_value(af, v, h, t_fast) for c, v in rat.items() for h in (30, 300)}
                    | {f"{c}@last": float(v[-1]) for c, v in rat.items()}, kv["zzvx.a.sum"] / kv["zzvx.a.wgt"])
            elif kv.get("zzvx.a") is not None:
                rat = {k_: fx[k_] for k_ in ("hl_ratio", "hl_ratio_c", "tr_pc_over_hl", "tr_op_over_hl", "hl_over_hlc", "hi_over_op",
                                             "op_over_lo", "px_over_lo", "hi_over_px", "mid_hi_lo")}
                put("zzvx.a", {f"{c}@last": float(v[-1]) for c, v in rat.items()}
                    | {f"{c}@30": ema_value(af, v, 30, t_fast) for c, v in rat.items()}, kv["zzvx.a"])
            # zzgh.nb
            if kv.get("zzgh.nb") is not None:
                s_ex, tl = gs["s"].to_numpy(), asl["t_local"]
                vol, amt, side = asl["vol"], asl["amt"], asl["side"]
                sets = {"ex_cont": (s_ex >= 34200) & (s_ex < 53820), "ex_all": s_ex >= 0, "ex_open": s_ex < 53820,
                        "loc_stop": tl <= stop, "loc_cont": (tl >= 0) & (s_ex < 53820)}
                c = {}
                for sn, m in sets.items():
                    with np.errstate(divide="ignore", invalid="ignore"):
                        c[f"amt_imb@{sn}"] = (amt * side)[m].sum() / amt[m].sum()
                        c[f"vol_imb@{sn}"] = (vol * side)[m].sum() / vol[m].sum()
                        c[f"n_imb@{sn}"] = side[m].sum() / m.sum()
                        c[f"net_amt@{sn}"] = (amt * side)[m].sum()
                        c[f"net_vol@{sn}"] = (vol * side)[m].sum()
                bins = [v for k_, v in kv.items() if k_.startswith("zzgh.c.")]
                c["bins_filled"] = float(sum(1 for v in bins if v))
                c["bins_total"] = float(len(bins))
                c["n_trades"] = float(gs.height)
                c["book_imb_last"] = float(af["timb"][-1])
                c["book_imb@30"] = ema_value(af, af["timb"], 30, t_fast)
                put("zzgh.nb", c, kv["zzgh.nb"])
            # zzhj.p: snapshot-amount flows at the 16 sampling times
            kv0 = pre.get(str(sid))
            if kv0 and None not in (kv.get("zzhj.p.sum"), kv.get("zzhj.p.wgt"), kv0.get("zzhj.p.sum"), kv0.get("zzhj.p.wgt")):
                flows = {}
                for clock in ("t_local", "t_exch"):
                    t = af[clock]
                    o = np.argsort(t, kind="stable")
                    t = t[o]
                    for name, w in (("samt", af["damt"]), ("svol", af["dvol"]), ("n_snap", np.ones(t.size)), ("n_volmoved", af["set_vol_moved"].astype(float))):
                        cs = np.concatenate([[0.0], np.cumsum(w[o])])
                        at = lambda x: cs[np.searchsorted(t, x, side="right")]  # noqa: E731, B023
                        for L in (10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900):
                            flows[f"{name}@{clock[2:]}_last{L}"] = at(Q.SAMPLE) - at(Q.SAMPLE - L)
                        flows[f"{name}@{clock[2:]}_cum"] = at(Q.SAMPLE)
                        for tau in (30, 60, 120, 300):
                            flows[f"{name}@{clock[2:]}_ema{tau}"] = np.array([(w[o][t <= T] * np.exp(-(T - t[t <= T]) / tau)).sum() for T in Q.SAMPLE])
                qh.append((d, sid, kv["zzhj.p.sum"], kv["zzhj.p.wgt"], kv0["zzhj.p.sum"], kv0["zzhj.p.wgt"], flows))
        for key, dk in per.items():
            pmv = np.array(dk.pop("pm"), float)
            score(rows, key, {c: np.array(v, float) for c, v in dk.items()}, pmv, {"date": d})
        print(f"{d}: {n_used} SZSE stocks scored ({len(per)} keys)")
    if not rows:
        print("nothing scored")
        return
    res = pl.DataFrame(rows)
    res.write_csv(args.out / "dcp_open.csv")
    agg = (res.group_by("key", "cand").agg(pl.len().alias("days"), pl.col("n").sum(), pl.col("rho").mean(), pl.col("k").median(),
                                            pl.col("lr_iqr").mean(), pl.col("ok4").mean(), pl.col("ok4_k").mean(), pl.col("ok2_k").mean())
           .sort("key", "lr_iqr"))
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        for key in agg["key"].unique().sort().to_list():
            t = agg.filter(pl.col("key") == key)
            print(f"\n== {key}: {t.height} candidates, best by log-ratio IQR (0 = same up to a constant k), then by rho")
            print(t.head(8))
            print(t.sort("rho", descending=True).head(4))
    if qh:
        s, w, s0, w0 = (np.array([r[i] for r in qh], float) for i in (2, 3, 4, 5))
        a, c = Q.fit_ac(w0, w)
        r = Q.solve_r(c)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = (s - a * s0) / (w - a * w0)
        print(f"\n== zzhj.p: weight wgt = {a:.4f} wgt_prev + {c:.4f} -> r {r:.4f}; {len(qh)} stock-days; day value x median {np.nanmedian(x):.4g}")
        Pr = r ** (15 - np.arange(16))
        out = []
        for name in qh[0][6]:
            V = np.vstack([rec[6][name] for rec in qh])
            fin = np.isfinite(V)
            with np.errstate(divide="ignore", invalid="ignore"):
                pv = (np.where(fin, V, 0.0) @ Pr) / (fin.astype(float) @ Pr)
            out.append({})
            score(out, "zzhj.p", {name: pv}, x)
        t = pl.DataFrame([o for o in out if o]).sort("lr_iqr")
        with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=200, float_precision=4):
            print(t.head(16))
            print(t.sort("rho", descending=True).head(6))
        t.write_csv(args.out / "zzhj_p.csv")
    print(f"tables: {args.out}")


if __name__ == "__main__":
    main()
