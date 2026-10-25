r"""Stage 4e: everything still open in PM's day-end state, in one run.

1. **multi-day EMAs** -- every ``<p>.sum / .wgt`` average fitted as

       wgt_D = a * wgt_{D-1} + c * sum_i w_i exp(-(T - t_i) ln2 / H)      sum_D likewise with w_i x_i

   with ``wgt_{D-1}`` PM's own previous-day state, ``a`` (the overnight decay) and ``c`` free and H
   up to days. Updates: snapshots (all / price changed / volume moved), trades, or once a day at a
   trigger -- ``d30`` / ``d50`` the first snapshot at or after 14:56:30 / 14:56:50 (where
   ``zzhj.p.t`` / ``zzui.r.t`` sit), ``dclose`` the close. Values are scanned as is and up to a
   constant factor (``k``): distance from the high (also to mid / bid / ask), spread, the per-update
   EMA increment ``alpha<H> = 1 - exp(-dt ln2 / H)``, change indicators, the 10 s price grid, ...
2. **power fit** of the values still unknown (``zzmh.r``, ``zzcb.r``, ``zzeb.r``, ``zztx.r``):
   ``log |v| = c + sum b_k log q_k`` over day quantities q (one and two at a time, SZSE).
3. **clock stamps** (``<p>.t``) of the averages on no intraday clock.
4. **profile inventory** (``<p>.c.<bin>``): bins, integer or decayed values, day-over-day growth.
   Every bin count is a multiple of 237 = the continuous-session minutes (09:31-11:30, 13:01-14:57),
   so these are per-minute intraday profiles carried across days.
5. **profile fit**: each 237-minute block as ``c_D[m] = a c_{D-1}[m] + b x_D[m]`` with x our
   per-minute quantity (trades, volume, amount, buy / sell splits, snapshots, high - low, 1), both bin
   layouts (block-major / minute-major) and both minute labels (left / right).

    python research/pm_factors/diag_rest.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader, trading clock
import dcp_ema as E  # tick loading and per-update arrays

SZSE = 3554
LN2 = np.log(2)
HALF = np.array([10, 30, 60, 100, 200, 300, 600, 1200, 1800, 3600, 5400, 7200, 10800, 14400, 21600, 28800,
                 43200, 86400, 172800, 1e12])
TIMES = ("t_local", "t_exch")
DAILY = {"d30": 14190.0, "d50": 14210.0, "dclose": None}  # trading-clock trigger of a once-a-day update
CFG = {"fast": (("all", "px_changed", "vol_moved"), ("one", "dvol", "dt")), "slow": (("all",), ("one", "vol", "dt")),
       **dict.fromkeys(DAILY, (("all",), ("one",)))}
ALPHA = (5, 10, 15, 20, 30, 60, 100, 300)
VALS = {
    "fast": ("price", "mid", "logp", "hl", "tr_pc", "tr_open", "dd_hi", "dd_hi2", "dd_hm", "dd_hm2", "ddm2", "ddb2", "dda2",
             "du_lo", "spread", "spread_abs", "sp_hl", "dt", "dt_exch", "chg", "bidchg", "askchg", "volchg", "imb1", "timb",
             "ret", "logret", "absret", "lp_open", "lp_pc", "inv_p", "stoch", "adp_hl", "one", *[f"alpha{h}" for h in ALPHA]),
    "slow": ("price", "logp", "ret", "logret", "absret", "side", "svol", "samt", "vol", "lvol", "rvwap", "rvwap_all", "hl",
             "tr_open", "dd_hi", "du_lo", "dt", "lp_open", "lp_pc", "inv_p", "stoch", "one", *[f"alpha{h}" for h in ALPHA]),
    "daily": ("last", "open", "high", "low", "vwap", "hl", "lp_oc", "lp_pc", "one", "snap", "logsnap", "mid", "lsnap_oc",
              "lsnap_pc", "g_mean", "g_ret", "g_lrv", "g_sd", "p1457"),
}
UNSOLVED = ("zzmh.r", "zzcb.r", "zzeb.r", "zztx.r")
HIST = re.compile(r"^(.+)\.c\.(-?\d+)$")
NMIN = 237  # continuous-session minutes: 09:31-11:30 and 13:01-14:57
QB = ("n_tr", "vol", "amt", "n_buy", "n_sell", "vol_buy", "vol_sell", "amt_buy", "amt_sell", "n_snap", "one", "hl")
QN = (*QB, *[f"cum_{q}" for q in QB[:10]])  # plus running totals along the minutes (zzfb looks cumulative)


def vkind(kind: str) -> str:
    """Value list of a kind (the daily triggers share one)."""
    return "daily" if kind in DAILY else kind


def triples(pm: pl.DataFrame) -> pl.DataFrame:
    """(sid, p, sum, wgt, t) per averaged prefix."""
    x = pm.filter(pl.col("key").str.contains(r"\.(sum|wgt|t)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt|t)$", 1).alias("p"), pl.col("key").str.extract(r"\.(sum|wgt|t)$", 1).alias("f"))
    w = x.pivot(on="f", index=["sid", "p"], values="pm")
    for f in ("sum", "wgt", "t"):
        if f not in w.columns:
            w = w.with_columns(pl.lit(None, pl.Float64).alias(f))
    return w.filter(pl.col("sum").is_not_null() & pl.col("wgt").is_not_null())


def daily_stream(af: dict | None, asl: dict, trig: float | None) -> dict | None:
    """One update per day: at the first snapshot received at or after ``trig`` (trading clock), or at the close."""
    ts = asl["t_local"]
    k = None
    if af is not None:
        tf = af["t_local"]
        k = min(int(np.searchsorted(tf, trig, side="left")), tf.size - 1) if trig is not None else tf.size - 1
    t_at = af["t_local"][k] if k is not None else (trig if trig is not None else ts[-1])
    cut = int(np.searchsorted(ts, t_at, side="right"))  # trades received by the trigger snapshot
    p = asl["price"][:cut]
    p = p[np.isfinite(p)]
    if p.size == 0:
        return None
    last, op = p[-1], p[0]
    pc = asl["price"][-1] / np.exp(asl["lp_pc"][-1]) if np.isfinite(asl["lp_pc"][-1]) else np.nan
    v = {"last": last, "open": op, "high": p.max(), "low": p.min(), "vwap": asl["rvwap"][cut - 1], "hl": p.max() - p.min(),
         "lp_oc": np.log(last / op), "lp_pc": np.log(last / pc), "one": 1.0}
    snap = mid = np.nan
    grid = np.full(10, np.nan)
    p1457 = np.nan
    if af is not None:
        snap, mid = af["price"][k], af["mid"][k]
        tf, pf = af["t_local"], af["price"]
        gi = np.searchsorted(tf, t_at - (9 - np.arange(10)) * 10, side="right") - 1  # 10 s grid ending at the trigger
        grid = np.where(gi >= 0, pf[np.clip(gi, 0, None)], np.nan)
        j = np.searchsorted(tf, 14220, side="right") - 1  # last snapshot received by 14:57
        p1457 = pf[j] if j >= 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.diff(np.log(grid))
        v.update(snap=snap, logsnap=np.log(snap), mid=mid, lsnap_oc=np.log(snap / op), lsnap_pc=np.log(snap / pc),
                 g_mean=np.nanmean(grid), g_ret=np.log(grid[-1] / grid[0]),
                 g_lrv=np.log(np.nansum(lr ** 2)) if np.nansum(lr ** 2) > 0 else np.nan, g_sd=np.nanstd(lr), p1457=p1457)
    return {**{n: np.array([x], float) for n, x in v.items()}, "t_local": np.zeros(1), "t_exch": np.zeros(1),
            "set_all": np.ones(1, bool)}


def tables(a: dict, kind: str) -> dict:
    """(time, set, weight) -> (W[H], S[x, H], Wx[x, H]): the EMA weight, and per value the weighted sum and its weight."""
    sets, weights = CFG[kind]
    n = a["t_local"].size
    X = np.column_stack([a[x] if x in a else np.full(n, np.nan) for x in VALS[vkind(kind)]])
    fin = np.isfinite(X)
    X0 = np.where(fin, X, 0.0)
    out = {}
    for tname in TIMES:
        for sset in sets:
            m = a.get(f"set_{sset}")
            if m is None:
                continue
            t = a[tname][m]
            if t.size == 0:
                continue
            e = np.exp(-(t[-1] - t)[:, None] * LN2 / HALF[None, :])
            for w in weights:
                wv = np.nan_to_num(a[w][m])
                out[(tname, sset, w)] = (wv @ e, (wv[:, None] * X0[m]).T @ e, (wv[:, None] * fin[m]).T @ e)
    return out


def fit_ac(y0: np.ndarray, w: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares y = a y0 + c w (0 <= a <= 1.5, c >= 0), refitted on the better half (a few stocks lose ticks in PM's feed)."""
    keep = np.isfinite(y0) & np.isfinite(w) & np.isfinite(y)
    a = c = 0.0
    for _ in range(3):
        if keep.sum() < 10:
            break
        (a, c), *_ = np.linalg.lstsq(np.column_stack([y0[keep], w[keep]]), y[keep], rcond=None)
        a, c = float(np.clip(a, 0.0, 1.5)), float(max(c, 0.0))
        r = np.abs(y - a * y0 - c * w)
        keep = keep & (r <= np.nanquantile(r[keep], 0.5))
    return a, c


def value_scan(recs: list, tabs: dict, kind: str, key: tuple, hi: int, a: float, c: float, y0: np.ndarray, s0: np.ndarray,
               pm_v: np.ndarray, sz: np.ndarray) -> list[tuple]:
    """Per value x: (name, median error, share within 1e-4, factor k, error after k, share within 1e-4 after k), SZSE."""
    xs = []
    for xi, xn in enumerate(VALS[vkind(kind)]):
        num, den = np.full(len(recs), np.nan), np.full(len(recs), np.nan)
        for i, (d, sid, *_) in enumerate(recs):
            g = tabs[d][sid][1].get(kind, {}).get(key)
            if g is not None:
                num[i], den[i] = g[1][xi, hi], g[2][xi, hi]
        with np.errstate(divide="ignore", invalid="ignore"):
            pv = (a * s0 + c * num) / (a * y0 + c * den)
            e = (np.abs(pv - pm_v) / np.maximum(np.abs(pm_v), 1e-12))[sz]
            kk = float(np.nanmedian((pm_v / pv)[sz]))
            es = (np.abs(kk * pv - pm_v) / np.maximum(np.abs(pm_v), 1e-12))[sz]
        if np.isfinite(e).any():
            xs.append((xn, float(np.nanmedian(e)), float(np.nanmean(e <= 1e-4)), kk, float(np.nanmedian(es)), float(np.nanmean(es <= 1e-4))))
    return xs


def minute_idx(s: np.ndarray, lab: str) -> np.ndarray:
    """Continuous-session minute 0..236 of exchange time-of-day s, left (09:30:00-09:30:59 = 0) or right labelled; -1 outside."""
    m = np.full(s.size, -1)
    if lab == "L":
        am, pm_ = (s >= 34200) & (s < 41400), (s >= 46800) & (s < 53820)
        m[am], m[pm_] = (s[am] - 34200) // 60, 120 + (s[pm_] - 46800) // 60
    else:
        am, pm_ = (s > 34200) & (s <= 41400), (s > 46800) & (s <= 53820)
        m[am], m[pm_] = np.ceil((s[am] - 34200) / 60) - 1, 120 + np.ceil((s[pm_] - 46800) / 60) - 1
    return m


def minute_q(gs: pl.DataFrame, gf: pl.DataFrame | None) -> dict[str, np.ndarray]:
    """Per-minute quantities (QN x 237) of one stock's day, for both minute labels."""
    s, px = gs["s"].to_numpy().astype(float), gs["price"].to_numpy().astype(float)
    vol, amt, side = (gs[c].to_numpy().astype(float) for c in ("vol", "amt", "side"))
    buy, sell = (side > 0).astype(float), (side < 0).astype(float)
    out = {}
    for lab in ("L", "R"):
        m = minute_idx(s, lab)
        ok = m >= 0
        Q = np.zeros((len(QB), NMIN))
        for i, w in enumerate((np.ones(s.size), vol, amt, buy, sell, vol * buy, vol * sell, amt * buy, amt * sell)):
            np.add.at(Q[i], m[ok], w[ok])
        if gf is not None:
            ms = minute_idx(gf["s"].to_numpy().astype(float), lab)
            np.add.at(Q[9], ms[ms >= 0], 1.0)
        Q[10] = 1.0
        hi, lo = np.full(NMIN, -np.inf), np.full(NMIN, np.inf)
        np.maximum.at(hi, m[ok], px[ok])
        np.minimum.at(lo, m[ok], px[ok])
        Q[11] = np.where(np.isfinite(hi - lo), hi - lo, 0.0)
        out[lab] = np.vstack([Q, np.cumsum(Q[:10], axis=1)])
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    files = sorted(p.stem for p in args.dcp_dir.glob("*.jz"))
    obs, tabs, kind_of, qrows, tstamps, profs, mq = [], {}, {}, [], [], {}, {}
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        prev = [f for f in files if f < d]
        if d not in files or not prev:
            print(f"skip {d}: no state file for it or the day before")
            continue
        pm, pm0 = B.load_state(args.dcp_dir / f"{d}.jz"), B.load_state(args.dcp_dir / f"{prev[-1]}.jz")
        st, st0 = triples(pm), triples(pm0)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(pm["sid"].unique().to_list())
        up = E.load(con, int(d), sids, stop)
        if up["slow"].is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        tabs[d], mq[d], last_t, day_q = {}, {}, [], []
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            if gs is None or gs.height < 2:
                continue
            af = E.arrays(gf, "fast") if gf is not None and gf.height >= 2 else None
            asl = E.arrays(gs, "slow")
            streams = {"slow": asl}
            if af is not None:
                hm = af["price"] * np.exp(af["dd_hm"])  # running max snapshot price from 09:30
                with np.errstate(divide="ignore", invalid="ignore"):
                    af["ddm2"], af["ddb2"], af["dda2"] = (np.log(hm / af[r]) ** 2 for r in ("mid", "bid1", "ask1"))
                streams["fast"] = af
            for a in list(streams.values()):
                for h in ALPHA:
                    a[f"alpha{h}"] = 1 - np.exp(-a["dt"] * LN2 / h)
            for k, trig in DAILY.items():
                ds = daily_stream(af, asl, trig)
                if ds is not None:
                    streams[k] = ds
            tabs[d][sid] = (int(gs["mkt"][0]), {k: tables(a, k) for k, a in streams.items()})
            mq[d][sid] = minute_q(gs, gf)
            last_t.append((sid, af["t_local"][-1] if af is not None else np.nan, asl["t_local"][-1]))
            q = {"sid": sid, "date": int(d), "mkt": int(gs["mkt"][0]), "n_tr": float(gs.height),
                 "n_snap": float(gf.height) if gf is not None else np.nan}
            if af is not None:
                tl = af["t_local"]
                late = tl >= tl[-1] - 300
                q["spr_abs"] = float(np.nanmean(af["spread_abs"][late])) if np.isfinite(af["spread_abs"][late]).any() else np.nan
                q["spr_rel"] = float(np.nanmean(af["spread"][late])) if np.isfinite(af["spread"][late]).any() else np.nan
                q["dt_mean"] = float(np.nanmean(np.diff(tl[tl >= tl[-1] - 600]))) if (tl >= tl[-1] - 600).sum() > 1 else np.nan
            day_q.append(q)
        # which clock each prefix runs on (first day decides)
        lt = pl.DataFrame(last_t, schema=["sid", "T_fast", "T_slow"], orient="row").with_columns(pl.col("sid").cast(pl.Int64))
        for p in st["p"].unique().to_list():
            if p in kind_of:
                continue
            g = st.filter(pl.col("p") == p).join(lt, on="sid")
            sh = {k: g.select(((pl.col("t") - pl.col(f"T_{k}")).abs() <= B.CLOCK_TOL).mean()).item() or 0 for k in ("fast", "slow")}
            k = max(sh, key=sh.get)
            kind_of[p] = k if sh[k] >= 0.5 else None
        tstamps.append(st.select("sid", "p", "t").with_columns(pl.lit(int(d)).alias("date")))
        j = st.join(st0.select("sid", "p", pl.col("sum").alias("s0"), pl.col("wgt").alias("y0")), on=["sid", "p"], how="left")
        obs.append(j.with_columns(pl.lit(d).alias("d")))
        qpm = pm.filter(pl.col("key").is_in(["zzxu.last", "zzxu.high", "zzxu.low", "zzmh.hm", "zzqj.v", "zzqj.tpv"])).pivot(
            on="key", index="sid", values="pm")
        qrows.append(pl.DataFrame(day_q).with_columns(pl.col("sid").cast(pl.Int64)).join(qpm, on="sid", how="left"))
        profs[d] = {}
        for f_, tag in ((args.dcp_dir / f"{d}.jz", "D"), (args.dcp_dir / f"{prev[-1]}.jz", "D-1")):
            with gzip.open(f_) as fh:
                raw = json.loads(fh.read())
            cells: dict[str, dict[int, dict[int, float]]] = {}
            for s_, kv in raw.items():
                for k, v in kv.items():
                    if v is not None and (m := HIST.match(k)):
                        cells.setdefault(m.group(1), {}).setdefault(int(s_), {})[int(m.group(2))] = float(v)
            for p, by in cells.items():
                nb = max(max(c) for c in by.values()) + 1
                arr = {}
                for sid, c in by.items():
                    a = np.full(nb, np.nan)
                    a[list(c)] = list(c.values())
                    arr[sid] = a
                profs[d][(tag, p)] = arr
        print(f"{d} (prev {prev[-1]}): {len(tabs[d])} stocks ({time.perf_counter() - t0:.0f}s)")
    if not obs:
        return
    ob = pl.concat(obs).with_columns(pl.col("y0").fill_null(0.0), pl.col("s0").fill_null(0.0))
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. multi-day EMA fit per prefix
    rows = []
    for p in sorted(ob["p"].unique().to_list()):
        o = ob.filter(pl.col("p") == p)
        kinds = [kind_of[p]] if kind_of.get(p) else ["fast", "slow", *DAILY]
        recs = [(r["d"], r["sid"], r["wgt"], r["sum"], r["y0"], r["s0"]) for r in o.iter_rows(named=True)
                if r["sid"] in tabs.get(r["d"], {}) and r["wgt"] and np.isfinite(r["wgt"]) and r["wgt"] != 0]
        if len(recs) < 30:
            continue
        mkt = np.array([tabs[d][sid][0] for d, sid, *_ in recs])
        y, s, y0, s0 = (np.array([r[i] for r in recs], float) for i in (2, 3, 4, 5))
        sz = mkt == SZSE
        pm_v = s / y
        best_kind = {}  # per update kind, its best weight fit
        for kind in kinds:
            keys = {k for d, sid, *_ in recs for k in tabs[d][sid][1].get(kind, {})}
            for key in sorted(keys):
                got = [tabs[d][sid][1].get(kind, {}).get(key) for d, sid, *_ in recs]
                ok = np.array([g is not None for g in got])
                if ok[sz].sum() < 20:
                    continue
                Wm = np.vstack([g[0] if g is not None else np.full(HALF.size, np.nan) for g in got])
                for hi in range(HALF.size):
                    a, c = fit_ac(y0[sz & ok], Wm[sz & ok, hi], y[sz & ok])
                    r = np.abs(y - a * y0 - c * Wm[:, hi]) / np.abs(y)
                    mre = float(np.nanmedian(r[sz & ok]))
                    if kind not in best_kind or mre < best_kind[kind][0]:
                        best_kind[kind] = (mre, kind, key, hi, a, c, float(np.mean(r[sz & ok] <= 1e-4)),
                                           float(np.mean(r[~sz & ok] <= 1e-4)) if (~sz & ok).any() else np.nan)
        # a near-constant intraday weight can imitate a daily update: pick the kind by weight + value error together
        cands = []
        for mre, kind, key, hi, a, c, ok4, ok4_sh in best_kind.values():
            xs = value_scan(recs, tabs, kind, key, hi, a, c, y0, s0, pm_v, sz)
            if not xs:
                continue
            xu, xk = sorted(xs, key=lambda t: t[1]), sorted(xs, key=lambda t: t[4])
            cands.append((mre + min(xu[0][1], xk[0][4]), mre, kind, key, hi, a, c, ok4, ok4_sh, xu, xk))
        if not cands:
            continue
        _, mre, kind, key, hi, a, c, ok4, ok4_sh, xu, xk = min(cands, key=lambda t: t[0])
        gap = float(-np.log(a) * HALF[hi] / LN2) if 0 < a < 1 else (0.0 if a >= 1 else np.inf)
        rows.append({"p": p, "clock": kind_of.get(p) or "-", "kind": kind, "time": key[0].removeprefix("t_"), "set": key[1],
                     "w": key[2], "half": float(HALF[hi]), "a": a, "c": c, "gap_s": gap, "mre_w": mre, "ok4_w": ok4, "ok4_w_SSE": ok4_sh,
                     "x": xu[0][0], "mre_x": xu[0][1], "ok4_x": xu[0][2], "xk": xk[0][0], "k": xk[0][3], "mre_xk": xk[0][4],
                     "ok4_xk": xk[0][5]})
    fit = pl.DataFrame(rows).sort("mre_w")
    fit.write_csv(args.out / "rest_ema.csv")

    # 2. power fit of the unsolved values
    qd = pl.concat(qrows, how="diagonal_relaxed")
    if {"zzxu.high", "zzxu.low", "zzxu.last", "zzmh.hm"} <= set(qd.columns):
        qd = qd.with_columns((pl.col("zzxu.high") - pl.col("zzxu.low")).alias("hl"),
                             (pl.col("zzmh.hm") / pl.col("zzxu.last")).log().alias("dd_hm"),
                             (pl.col("zzxu.high") / pl.col("zzxu.last")).log().alias("dd_hi"))
    qd = qd.rename({k: v for k, v in {"zzxu.last": "px", "zzqj.v": "vol", "zzqj.tpv": "amt"}.items() if k in qd.columns})
    qnames = [c for c in ("px", "hl", "dd_hm", "dd_hi", "vol", "amt", "n_tr", "n_snap", "spr_abs", "spr_rel", "dt_mean") if c in qd.columns]
    vals = ob.filter(pl.col("p").is_in(list(UNSOLVED))).select("sid", pl.col("d").cast(pl.Int64).alias("date"), "p",
                                                               (pl.col("sum") / pl.col("wgt")).alias("v"))
    pw = []
    for p in UNSOLVED:
        jn = vals.filter(pl.col("p") == p).join(qd, on=["sid", "date"]).filter(pl.col("mkt") == SZSE)
        if jn.height < 30:
            continue
        v = jn["v"].to_numpy()
        for combo in [(c,) for c in qnames] + list(itertools.combinations(qnames, 2)):
            Q = np.column_stack([jn[c].to_numpy().astype(float) for c in combo])
            ok = (v != 0) & np.isfinite(v) & np.all(np.isfinite(Q) & (Q > 0), axis=1)
            if ok.sum() < 30:
                continue
            Xd = np.column_stack([np.ones(ok.sum()), np.log(Q[ok])])
            yv = np.log(np.abs(v[ok]))
            coef, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
            r2 = 1 - ((yv - Xd @ coef) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
            pw.append({"p": p, "q": " * ".join(combo), "r2": float(r2), "const": float(np.exp(np.clip(coef[0], -700, 700))),
                       "b1": float(coef[1]), "b2": float(coef[2]) if len(coef) > 2 else None, "n": int(ok.sum()),
                       "neg": float(np.mean(v < 0))})
    pwf = pl.DataFrame(pw).sort(["p", "r2"], descending=[False, True]) if pw else None

    # 3. clock stamps of the prefixes on no intraday clock
    ts = pl.concat(tstamps).filter(pl.col("p").is_in([p for p, k in kind_of.items() if not k]))
    tsum = ts.group_by("p", "date").agg(pl.col("t").quantile(0.1).alias("t_p10"), pl.col("t").median().alias("t_med"),
                                        pl.col("t").quantile(0.9).alias("t_p90"),
                                        ((pl.col("t") - pl.col("t").median()).abs() < 0.01).mean().alias("same")).sort("p", "date")

    # 4. profile inventory and 5. profile fit
    inv, prow = [], []
    pnames = sorted({p for d in profs for (_, p) in profs[d]})
    for p in pnames:
        tot, tot0, ints = [], [], []
        for d in profs:
            cur, pre = profs[d].get(("D", p), {}), profs[d].get(("D-1", p), {})
            for sid, arr in cur.items():
                tot.append(np.nansum(arr))
                ints.append(np.nanmean(arr == np.round(arr)))
                if sid in pre:
                    tot0.append((np.nansum(arr), np.nansum(pre[sid])))
        nb = max(len(a) for d in profs for a in profs[d].get(("D", p), {}).values())
        inc = np.array([x - z for x, z in tot0])
        inv.append({"p": p, "bins": nb, "blocks": nb / NMIN, "integer": float(np.mean(ints)), "tot_med": float(np.median(tot)),
                    "inc_p10": float(np.quantile(inc, 0.1)), "inc_med": float(np.median(inc)), "inc_p90": float(np.quantile(inc, 0.9)),
                    "inc_neg": float(np.mean(inc < 0))})
        if nb % NMIN:
            continue
        nblk = nb // NMIN
        for layout in ("block", "inter"):
            for jb in range(nblk):
                idx = jb * NMIN + np.arange(NMIN) if layout == "block" else np.arange(NMIN) * nblk + jb
                Y, Y0, Xs = [], [], {lab: [] for lab in ("L", "R")}
                for d in profs:
                    cur, pre = profs[d].get(("D", p), {}), profs[d].get(("D-1", p), {})
                    for sid, arr in cur.items():
                        if sid not in pre or sid not in mq.get(d, {}) or tabs[d][sid][0] != SZSE or len(arr) != nb or len(pre[sid]) != nb:
                            continue
                        Y.append(arr[idx])
                        Y0.append(pre[sid][idx])
                        for lab in ("L", "R"):
                            Xs[lab].append(mq[d][sid][lab])
                if len(Y) < 20:
                    continue
                y, y0 = np.concatenate(Y), np.concatenate(Y0)
                var = np.nansum((y - np.nanmean(y)) ** 2)
                best = None
                for lab in ("L", "R"):
                    X = np.concatenate(Xs[lab], axis=1)  # (QN, cells)
                    for qi, qn in enumerate(QN):
                        a, b = fit_ac(y0, X[qi], y)
                        r = y - a * y0 - b * X[qi]
                        r2 = 1 - np.nansum(r ** 2) / var if var > 0 else np.nan
                        okc = float(np.nanmean(np.abs(r) <= np.maximum(0.5, 1e-6 * np.abs(y))))
                        if best is None or (okc, r2) > (best[0], best[1]):
                            best = (okc, r2, lab, qn, a, b)
                okc, r2, lab, qn, a, b = best
                prow.append({"p": p, "block": jb, "layout": layout, "lab": lab, "q": qn, "a": a, "b": b, "r2": r2, "ok": okc,
                             "y_med": float(np.nanmedian(y)), "cells": int(y.size)})
    hsum = pl.DataFrame(inv)
    hsum.write_csv(args.out / "rest_hist.csv")
    pfit = pl.DataFrame(prow) if prow else None
    if pfit is not None:
        pfit.write_csv(args.out / "rest_profile.csv")
        # per prefix keep the layout that fits its blocks best
        lay = pfit.group_by("p", "layout").agg(pl.col("ok").mean().alias("m_ok"), pl.col("r2").mean().alias("m_r2")).sort(
            ["p", "m_ok", "m_r2"], descending=[False, True, True]).group_by("p", maintain_order=True).first()
        pfit = pfit.join(lay.select("p", "layout"), on=["p", "layout"]).sort("p", "block")

    with pl.Config(tbl_rows=80, tbl_cols=20, tbl_width_chars=260, fmt_str_lengths=16, float_precision=5):
        print("\n== 1. multi-day EMA fit (wgt_D = a wgt_D-1 + c * today's EMA; SZSE; x: best value as is, xk: best up to a factor k)")
        print(fit)
        if pwf is not None:
            print("\n== 2. power fit of unsolved values, best 6 each (log|v| = log const + b1 log q1 + b2 log q2, SZSE; neg = share < 0)")
            print(pwf.group_by("p", maintain_order=True).head(6))
        print("\n== 3. clock stamps of the averages on no intraday clock (trading-clock seconds; same = share at the median)")
        print(tsum)
        print("\n== 4. profiles: bins (multiples of 237 minutes), integer values, median total, day-over-day growth of the total")
        print(hsum)
        if pfit is not None:
            print("\n== 5. profile fit per 237-minute block: c_D[m] = a c_D-1[m] + b x[m] (SZSE; ok: share within 0.5 / 1e-6)")
            print(pfit)
    print(f"tables: {args.out}/rest_ema.csv, rest_hist.csv, rest_profile.csv ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
