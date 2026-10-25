r"""Stage 4c: PM's day-end EMAs on round half-lives -- scan update rules and values, score exactly.

``diag_iema`` (3 days, SZSE) found every snapshot / trade EMA in the state file to be
``wgt = sum_i exp(-(T - t_i) / tau)`` with weight 1 and ``tau * ln 2`` a round number of seconds on
the trading clock: 10 (``zzmh.r``), 30 (``zzvx.*``, ``zzcb.r``, ``zzqj.r``), 100 (``zzeb.*``,
``zzbd.pEmaS/T``, ``zztx.r``), 200 (``zzbd.rEmaS/T``), 300 (``zzpk.p``, ``zzxu.*``, ``zzab2.r``,
``zzup.pf``), 600 (``zzup.ps``); values: price for ``zzpk.p`` / ``zzab2.r`` / ``zzup.pf`` / ``zzxu.p``,
log price for ``zzbd.pEmaS/T``, mid for ``zzeb.p``. The weight matched to ~1e-6 but not exactly,
worse the shorter the half-life -- the jitter of our arrival times against PM's.

This fixes ``tau = H / ln 2`` with H on a grid of round half-lives and scans, per prefix:

* time: local arrival or exchange time (trading clock either way);
* updates: every snapshot / trade, snapshots where the volume moved or the price changed, trades
  grouped by arrival (one update per batch);
* weight: 1, volume, amount;

then, with the best of those, the value against a wide candidate list (price, mid, log forms,
returns of price and mid, against the open and the previous close, the day's range, ask / bid,
the snapshot's own vwap, trade side). Errors are shown at two tolerances (1e-6 and 1e-4), since our
clock jitter alone moves a 30 s EMA by ~1e-5.

    python research/pm_factors/dcp_ema.py --dates 20260923,20260929,20261009
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog path, time expressions, state loader
import diag_iema as I  # state triples, clock assignment helpers

SZSE = 3554
# round half-lives in seconds; 1e12 = no decay (a plain mean over the day's updates)
HALF = np.array([5, 10, 15, 20, 30, 45, 60, 90, 100, 120, 150, 200, 240, 300, 450, 600, 900, 1200, 1800, 3600,
                 5400, 7200, 10800, 14400, 28800, 1e12], float)
SETS = {"fast": ("all", "vol_moved", "px_changed"), "slow": ("all", "batch")}
WEIGHTS = {"fast": ("one", "dvol", "damt"), "slow": ("one", "vol", "amt")}
VALUES = {
    "fast": ("price", "mid", "logp", "logmid", "ask1", "bid1", "svwap", "ret", "logret", "mret", "mlogret", "absret",
             "ret2", "logret2", "spread", "lp_open", "lp_pc", "ret_open", "ret_pc", "lhl", "lh_open", "ll_open",
             "hl_pc", "lh_pc", "ll_pc", "dp", "absdp", "dmid", "absdmid", "imb1", "limb1", "timb", "ltimb", "dvol",
             "ldvol", "damt", "chg", "one", "hl", "tr_pc", "tr_open", "inv_p", "pprev", "spread_abs",
             "volchg", "bidchg", "askchg", "dt", "dt_exch", "dd_hi", "du_lo", "dd_hi2", "du_lo2", "dd_hm", "dd_hm2",
             "sp_hl", "adp_hl", "stoch", "trr_pc", "trr_open"),
    "slow": ("price", "logp", "ret", "logret", "absret", "ret2", "logret2", "lp_open", "lp_pc", "ret_open", "ret_pc",
             "side", "svol", "samt", "vol", "lvol", "amt", "dp", "absdp", "one", "hl", "tr_pc", "tr_open", "inv_p", "pprev",
             "dt", "dt_exch", "dd_hi", "du_lo", "dd_hi2", "du_lo2", "adp_hl", "stoch", "rvwap", "rvwap_all"),
}


def load(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> dict[str, pl.DataFrame]:
    """Snapshots and trades received by the stop, with both clocks and the fields the values need."""
    S, L = B.SOD, B.LSOD
    con.register("want", pl.DataFrame({"sid": sids}))
    fast = con.sql(f"""SELECT symbol_id AS sid, market_id AS mkt, {B.tc(L)} AS t_local, {B.tc(S)} AS t_exch, spider_ts AS ls, {S} AS s,
        price, q.bid_px[1] AS b1, q.ask_px[1] AS a1, q.bid_vol[1]::DOUBLE AS bv1, q.ask_vol[1]::DOUBLE AS av1,
        list_sum(q.bid_vol[1:5])::DOUBLE AS bv5, list_sum(q.ask_vol[1:5])::DOUBLE AS av5,
        total_bid_volume::DOUBLE AS tbv, total_ask_volume::DOUBLE AS tav, total_volume::DOUBLE AS cv, total_amount::DOUBLE AS ca,
        open::DOUBLE AS op, high::DOUBLE AS hi, low::DOUBLE AS lo, pre_close::DOUBLE AS pc
        FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} AND {L} <= {stop}
        ORDER BY sid, spider_ts""").pl()
    slow = con.sql(f"""SELECT symbol_id AS sid, market_id AS mkt, {B.tc(L)} AS t_local, {B.tc(S)} AS t_exch, spider_ts AS ls, {S} AS s,
        price, volume::DOUBLE AS vol, volume * price AS amt, CASE dir WHEN 1063 THEN 1.0 WHEN 1550 THEN -1.0 ELSE 0.0 END AS side
        FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id
        WHERE r.date = {date} AND r.trade_type = 1 AND {L} <= {stop} ORDER BY sid, spider_ts""").pl()
    ref = con.sql(f"""SELECT symbol_id AS sid, max(open) FILTER (open > 0)::DOUBLE AS op, max(pre_close) FILTER (pre_close > 0)::DOUBLE AS pc
        FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} GROUP BY 1""").pl()
    con.unregister("want")
    return {"fast": fast, "slow": slow.join(ref, on="sid", how="left")}


def arrays(g: pl.DataFrame, kind: str) -> dict[str, np.ndarray]:
    """Per-update arrays for one stock: both clocks, update-set masks, weights and value candidates."""
    f = lambda c: g[c].to_numpy().astype(float)  # noqa: E731 -- column as float array
    px = f("price")
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.where(px > 0, px, np.nan)
        prev = np.concatenate([[np.nan], pos[:-1]])
        a = {"t_local": f("t_local"), "t_exch": f("t_exch"), "price": pos, "logp": np.log(pos),
             "ret": pos / prev - 1, "logret": np.log(pos / prev), "one": np.ones(px.size)}
        a["absret"], a["ret2"], a["logret2"] = np.abs(a["ret"]), a["ret"] ** 2, a["logret"] ** 2
        a["dp"] = pos - prev
        a["absdp"] = np.abs(a["dp"])
        a["inv_p"], a["pprev"] = 1 / pos, prev
        op, pc = f("op"), f("pc")
        op, pc = np.where(op > 0, op, np.nan), np.where(pc > 0, pc, np.nan)
        a["lp_open"], a["lp_pc"], a["ret_open"], a["ret_pc"] = np.log(pos / op), np.log(pos / pc), pos / op - 1, pos / pc - 1
        if kind == "fast":
            b1, a1 = f("b1"), f("a1")
            two = (b1 > 0) & (a1 > 0)
            mid = np.where(two, (b1 + a1) / 2, pos)
            pmid = np.concatenate([[np.nan], mid[:-1]])
            dv = np.diff(f("cv"), prepend=np.nan)
            da = np.diff(f("ca"), prepend=np.nan)
            hi, lo = f("hi"), f("lo")
            hi, lo = np.where(hi > 0, hi, np.nan), np.where(lo > 0, lo, np.nan)
            a.update(mid=mid, logmid=np.log(mid), ask1=np.where(a1 > 0, a1, np.nan), bid1=np.where(b1 > 0, b1, np.nan),
                     mret=mid / pmid - 1, mlogret=np.log(mid / pmid), spread=np.where(two, (a1 - b1) / mid, np.nan),
                     dvol=np.nan_to_num(np.clip(dv, 0, None)), damt=np.nan_to_num(np.clip(da, 0, None)),
                     svwap=np.where(dv > 0, da / dv, np.nan), lhl=np.log(hi / lo), lh_open=np.log(hi / op),
                     ll_open=np.log(lo / op), hl_pc=(hi - lo) / pc, lh_pc=np.log(hi / pc), ll_pc=np.log(lo / pc))
            a["spread_abs"] = np.where(two, a1 - b1, np.nan)
            a["volchg"] = (np.nan_to_num(dv) > 0).astype(float)
            a["bidchg"] = np.concatenate([[0.0], (b1[1:] != b1[:-1]).astype(float)])
            a["askchg"] = np.concatenate([[0.0], (a1[1:] != a1[:-1]).astype(float)])
            bv, av, tb, ta = f("bv1"), f("av1"), f("tbv"), f("tav")
            a.update(dmid=mid - pmid, absdmid=np.abs(mid - pmid), imb1=(bv - av) / (bv + av), limb1=np.log(bv / av),
                     timb=(tb - ta) / (tb + ta), ltimb=np.log(tb / ta), ldvol=np.log1p(a["dvol"]),
                     chg=np.concatenate([[0.0], (px[1:] != px[:-1]).astype(float)]))
            a["set_all"] = np.ones(px.size, bool)
            a["set_vol_moved"] = np.nan_to_num(dv) > 0
            a["set_px_changed"] = np.concatenate([[True], px[1:] != px[:-1]])
        else:
            a.update(vol=f("vol"), amt=f("amt"), side=f("side"), lvol=np.log(f("vol")))
            a["svol"], a["samt"] = a["side"] * a["vol"], a["side"] * a["amt"]
            cont = g["s"].to_numpy() >= 34200  # the day's VWAP so far: from 09:30 (zzqj.v / tpv) or with the open auction
            with np.errstate(divide="ignore", invalid="ignore"):
                a["rvwap"] = np.cumsum(a["amt"] * cont) / np.cumsum(a["vol"] * cont)
                a["rvwap_all"] = np.cumsum(a["amt"]) / np.cumsum(a["vol"])
            hi, lo = np.fmax.accumulate(pos), np.fmin.accumulate(pos)  # the day's range so far, from the trades
            ls = g["ls"].to_numpy()
            a["set_all"] = np.ones(px.size, bool)
            a["set_batch"] = np.concatenate([ls[1:] != ls[:-1], [True]])  # last trade of each arrival batch
        # range in price units (true range against the previous close / the open)
        a["hl"], a["tr_pc"], a["tr_open"] = hi - lo, np.fmax(hi, pc) - np.fmin(lo, pc), np.fmax(hi, op) - np.fmin(lo, op)
        # true range / range: 1 while the previous close (open) sits inside the day's range
        a["trr_pc"] = np.where(a["hl"] > 0, a["tr_pc"] / a["hl"], 1.0)
        a["trr_open"] = np.where(a["hl"] > 0, a["tr_open"] / a["hl"], 1.0)
        a["dd_hi"], a["du_lo"] = np.log(hi / pos), np.log(pos / lo)  # distance below the day high / above the low
        a["dd_hi2"], a["du_lo2"] = a["dd_hi"] ** 2, a["du_lo"] ** 2
        a["adp_hl"], a["stoch"] = a["absdp"] / (hi - lo), (pos - lo) / (hi - lo)
        if kind == "fast":
            hm = np.fmax.accumulate(np.where(g["s"].to_numpy() >= 34200, pos, np.nan))  # running max snapshot price from 09:30
            a["dd_hm"] = np.log(hm / pos)
            a["dd_hm2"] = a["dd_hm"] ** 2
            a["sp_hl"] = a["spread_abs"] / (hi - lo)
        a["dt"] = np.diff(a["t_local"], prepend=np.nan)
        a["dt_exch"] = np.diff(a["t_exch"], prepend=np.nan)
    return a


def weight_table(a: dict, kind: str) -> dict[tuple[str, str], np.ndarray]:
    """(time, set) -> (n_weights, n_half): the EMA weight at the last update for every half-life at once."""
    out = {}
    for time in ("t_local", "t_exch"):
        for sset in SETS[kind]:
            m = a[f"set_{sset}"]
            t = a[time][m]
            if t.size == 0:
                out[(time, sset)] = np.full((len(WEIGHTS[kind]), HALF.size), np.nan)
                continue
            e = np.exp(-(t[-1] - t)[:, None] * np.log(2) / HALF[None, :])
            out[(time, sset)] = np.vstack([a[w][m] for w in WEIGHTS[kind]]) @ e
    return out


def ema(a: dict, time: str, sset: str, w: str, half: float, x: str | None = None) -> float:
    """Weight (x None) or weighted average of x for one stock, half-life ``half`` seconds."""
    m = a[f"set_{sset}"]
    t = a[time][m]
    if t.size == 0:
        return np.nan
    e = a[w][m] * np.exp(-(t[-1] - t) * np.log(2) / half)
    if x is None:
        return float(e.sum())
    v = a[x][m]
    ok = np.isfinite(v)
    return float((e[ok] * v[ok]).sum() / e[ok].sum()) if ok.any() else np.nan


def score(errs: list[float]) -> dict[str, float]:
    """Median error and exact shares at 1e-6 / 1e-4."""
    e = np.array([x for x in errs if np.isfinite(x)])
    if e.size == 0:
        return {"mre": np.inf, "ok6": 0.0, "ok4": 0.0}
    return {"mre": float(np.median(e)), "ok6": float((e <= 1e-6).mean()), "ok4": float((e <= 1e-4).mean())}


def top_corr(v: np.ndarray, cands: dict[str, np.ndarray]) -> dict[str, object]:
    """Find the 3 candidates whose cross-stock ranking matches v best: median ratio k, share within 1e-4 after scaling by k."""
    def rank(z: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(z)).astype(float)
    res = []
    for x, c in cands.items():
        ok = np.isfinite(v) & np.isfinite(c)
        if ok.sum() < 20 or np.std(c[ok]) == 0:
            continue
        rho = float(np.corrcoef(rank(v[ok]), rank(c[ok]))[0, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            k = float(np.nanmedian(v[ok] / c[ok]))
            sc = float(np.mean(np.abs(v[ok] - k * c[ok]) <= 1e-4 * np.abs(v[ok])))
            ki = float(np.nanmedian(v[ok] * c[ok]))  # v = ki / c
            si = float(np.mean(np.abs(v[ok] - ki / c[ok]) <= 1e-4 * np.abs(v[ok])))
        if si > sc or (si == sc and rho < 0):
            x, k, sc = f"1/{x}", ki, si
        res.append((x, rho, k, sc))
    res.sort(key=lambda t: -abs(t[1]))
    out: dict[str, object] = {}
    for i, (x, rho, k, sc) in enumerate(res[:3], 1):
        out[f"c{i}"], out[f"rho{i}"], out[f"k{i}"], out[f"s{i}"] = x, rho, k, sc
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    args = ap.parse_args()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    days = []  # (state, kind_of, {(kind, sid): (mkt, arrays)})
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pm = B.load_state(f)
        st = I.state(pm)
        t = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (t + 34200 if t <= 7200 else t - 7200 + 46800) + 0.005
        up = load(con, int(d), sorted(pm["sid"].unique().to_list()), stop)
        if up["slow"].is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        last = {k: v.group_by("sid").agg(pl.col("t_local").max().alias("T")).with_columns(pl.col("sid").cast(pl.Int64)) for k, v in up.items()}
        kind_of = {}
        for p in st["p"].unique().sort().to_list():
            g = st.filter(pl.col("p") == p)
            share = {k: g.join(last[k], on="sid").select(((pl.col("t") - pl.col("T")).abs() <= B.CLOCK_TOL).mean()).item() or 0 for k in last}
            k = max(share, key=share.get)
            if share[k] >= 0.5:
                kind_of[p] = k
        streams = {}
        for kind, u in up.items():
            for g in u.partition_by("sid"):
                a = arrays(g, kind)
                streams[(kind, int(g["sid"][0]))] = (int(g["mkt"][0]), a, weight_table(a, kind))
        days.append((st, kind_of, streams))
        print(f"{d}: {len(kind_of)} prefixes on a clock, {len(streams)} streams")
    if not days:
        return
    prefixes = sorted({p for _, k, _ in days for p in k})
    rows, cfg_rows, prof = [], [], []
    for p in prefixes:
        kind = next(k[p] for _, k, _ in days if p in k)
        pairs = []  # (market, arrays, weight table, PM wgt, PM sum)
        for st, k, streams in days:
            if k.get(p) != kind:
                continue
            for sid, y, s_ in st.filter(pl.col("p") == p).select("sid", "wgt", "sum").iter_rows():
                q = streams.get((kind, sid))
                if q and y and np.isfinite(y) and q[1]["one"].size >= 2:
                    pairs.append((q[0], q[1], q[2], y, s_))
        sz = [q for q in pairs if q[0] == SZSE]
        sh = [q for q in pairs if q[0] != SZSE]
        if len(sz) < 20:
            print(f"skip {p}: {len(sz)} SZSE stocks")
            continue
        y_sz = np.array([q[3] for q in sz])
        best = None
        for time in ("t_local", "t_exch"):
            for sset in SETS[kind]:
                err = np.abs(np.stack([q[2][(time, sset)] for q in sz]) / y_sz[:, None, None] - 1)  # (stocks, w, half)
                med, ok6, ok4 = np.nanmedian(err, axis=0), (err <= 1e-6).mean(axis=0), (err <= 1e-4).mean(axis=0)
                for wi, w in enumerate(WEIGHTS[kind]):
                    for hi, h in enumerate(HALF):
                        s = {"mre": float(med[wi, hi]), "ok6": float(ok6[wi, hi]), "ok4": float(ok4[wi, hi])}
                        cfg_rows.append({"p": p, "kind": kind, "time": time, "set": sset, "w": w, "half": h, **s})
                        if best is None or (s["mre"], -s["ok6"]) < (best[1]["mre"], -best[1]["ok6"]):
                            best = ((time, sset, w, h), s)
        (time, sset, w, h), s = best
        s_sh = score([abs(ema(a, time, sset, w, h) / y - 1) for _, a, _t, y, _ in sh])
        xs = []
        for x in VALUES[kind]:
            xs.append((x, score([abs(ema(a, time, sset, w, h, x) - s_ / y) / max(abs(s_ / y), 1e-12) for _, a, _t, y, s_ in sz])))
        xs.sort(key=lambda t: (t[1]["mre"], -t[1]["ok6"]))
        (x1, v1), (x2, v2) = xs[0], xs[1]
        if v1["ok4"] < 0.9:  # unsolved: what kind of quantity is PM's value, and which candidate does it track
            pmv = np.array([s_ / y for _, _a, _t, y, s_ in sz])
            prof.append({"p": p, "half": h, "v_p10": np.nanquantile(pmv, 0.1), "v_med": np.nanmedian(pmv), "v_p90": np.nanquantile(pmv, 0.9),
                         "neg": float(np.mean(pmv < 0)), **top_corr(pmv, {x: np.array([ema(a, time, sset, w, h, x) for _, a, _t, _y, _s in sz])
                                                                         for x in VALUES[kind]})})
            # the value may run on its own half-life: rescan the top candidates over every half-life
            hb = (np.inf, None, None, 0.0)
            for x in {str(prof[-1].get(f"c{i}", "")).removeprefix("1/") for i in (1, 2, 3)} - {""}:
                for hx in HALF:
                    e = [abs(ema(a, time, sset, w, hx, x) - s_ / y) / max(abs(s_ / y), 1e-12) for _, a, _t, y, s_ in sz]
                    sc = score(e)
                    if sc["mre"] < hb[0]:
                        hb = (sc["mre"], x, hx, sc["ok4"])
            prof[-1].update(hx_x=hb[1], hx=hb[2], hx_mre=hb[0], hx_ok4=hb[3])
        rows.append({"p": p, "kind": kind, "half": h, "time": time.removeprefix("t_"), "set": sset, "w": w,
                     "mre_w": s["mre"], "ok6_w": s["ok6"], "ok4_w": s["ok4"], "mre_w_sh": s_sh["mre"],
                     "x": x1, "mre_x": v1["mre"], "ok6_x": v1["ok6"], "ok4_x": v1["ok4"], "x2": x2, "mre_x2": v2["mre"]})
        print(f"done {p}")
    out = pl.DataFrame(rows).sort("kind", "half", "p")
    args.out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(cfg_rows).write_csv(args.out / "dcp_ema_configs.csv")
    out.write_csv(args.out / "dcp_ema.csv")
    with pl.Config(tbl_rows=60, tbl_cols=26, tbl_width_chars=280, fmt_str_lengths=14, float_precision=6):
        print("\n== per prefix: best half-life / time / update set / weight on SZSE (ok6 / ok4: share within 1e-6 / 1e-4),"
              " SSE weight error, best value x and runner-up x2")
        print(out)
        if prof:
            print("\n== unsolved values on SZSE: PM value quantiles, share negative, and the 3 candidates it ranks with best"
                  " (rho: Spearman; k: median PM / candidate, or PM x candidate for 1/c; s: share within 1e-4 of k x c or k / c)")
            print(pl.DataFrame(prof))
    print(f"configs: {args.out / 'dcp_ema_configs.csv'}, table: {args.out / 'dcp_ema.csv'}")


if __name__ == "__main__":
    main()
