r"""Stage 4e: everything still open in PM's day-end state, in one run.

1. **multi-day EMAs** -- every ``<p>.sum / .wgt`` average fitted as

       wgt_D = a * wgt_{D-1} + sum_i w_i exp(-(T - t_i) ln2 / H)      sum_D likewise with w_i x_i

   with ``wgt_{D-1}`` PM's own previous-day state, ``a`` free (the overnight decay, whatever the
   clock does across the night) and H up to days. Updates: snapshots (all / price changed), trades,
   or once a day (``daily``: one update at the close, values = day quantities). Aimed at the
   averages no intraday rule fits (``zzba.p/r``, ``zzhj.p``, ``zzui.r``); the solved ones should
   come out with ``a`` ~ 0. The value scan is wide (distance from the high, spread, the per-update
   EMA increment ``alpha<H> = 1 - exp(-dt ln2 / H)``, change indicators, ...).
2. **power fit** of the values still unknown (``zzmh.r``, ``zzcb.r``, ``zzeb.r``, ``zztx.r``):
   ``log |v| = c + sum b_k log q_k`` over day quantities q (one and two at a time, SZSE), so a
   relation like ``v = 4 / range`` or ``v = c dd^2`` shows as round exponents and R^2 ~ 1.
3. **clock stamps** (``<p>.t``) of the averages on no intraday clock: a fixed time of day = an
   update at that moment.
4. **histogram inventory** (``<p>.c.<bin>``): bins, labels, integer or decayed counts, and how
   much the total grows from one day to the next (48 = one count per 5-minute bar).

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
CFG = {"fast": (("all", "px_changed"), ("one", "dvol")), "slow": (("all",), ("one", "vol")), "daily": (("all",), ("one",))}
ALPHA = (10, 30, 100, 300)
VALS = {
    "fast": ("price", "mid", "logp", "hl", "tr_pc", "tr_open", "dd_hi", "dd_hi2", "dd_hm", "dd_hm2", "du_lo", "spread",
             "spread_abs", "sp_hl", "dt", "dt_exch", "chg", "bidchg", "askchg", "volchg", "imb1", "timb", "ret", "logret",
             "absret", "lp_open", "lp_pc", "inv_p", "stoch", "adp_hl", "one", *[f"alpha{h}" for h in ALPHA]),
    "slow": ("price", "logp", "ret", "logret", "absret", "side", "svol", "samt", "vol", "lvol", "rvwap", "rvwap_all", "hl",
             "tr_open", "dd_hi", "du_lo", "dt", "lp_open", "lp_pc", "inv_p", "stoch", "one", *[f"alpha{h}" for h in ALPHA]),
    "daily": ("last", "open", "high", "low", "vwap", "p1457", "hl", "lp_oc", "lp_pc", "one"),
}
UNSOLVED = ("zzmh.r", "zzcb.r", "zzeb.r", "zztx.r")
HIST = re.compile(r"^(.+)\.c\.(-?\d+)$")


def triples(pm: pl.DataFrame) -> pl.DataFrame:
    """(sid, p, sum, wgt, t) per averaged prefix."""
    x = pm.filter(pl.col("key").str.contains(r"\.(sum|wgt|t)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt|t)$", 1).alias("p"), pl.col("key").str.extract(r"\.(sum|wgt|t)$", 1).alias("f"))
    w = x.pivot(on="f", index=["sid", "p"], values="pm")
    for f in ("sum", "wgt", "t"):
        if f not in w.columns:
            w = w.with_columns(pl.lit(None, pl.Float64).alias(f))
    return w.filter(pl.col("sum").is_not_null() & pl.col("wgt").is_not_null())


def daily_stream(af: dict | None, asl: dict) -> dict:
    """One update per day at the close, carrying the day's quantities."""
    px = asl["price"][np.isfinite(asl["price"])]
    last = px[-1]
    pc = last / np.exp(asl["lp_pc"][-1]) if np.isfinite(asl["lp_pc"][-1]) else np.nan
    p1457 = np.nan
    if af is not None:
        k = np.searchsorted(af["t_local"], 14220, side="right") - 1  # last snapshot received by 14:57
        p1457 = af["price"][k] if k >= 0 else np.nan
    v = {"last": last, "open": px[0], "high": px.max(), "low": px.min(), "vwap": asl["rvwap"][-1], "p1457": p1457,
         "hl": px.max() - px.min(), "lp_oc": np.log(last / px[0]), "lp_pc": np.log(last / pc), "one": 1.0}
    return {**{k: np.array([x], float) for k, x in v.items()}, "t_local": np.zeros(1), "t_exch": np.zeros(1),
            "set_all": np.ones(1, bool)}


def tables(a: dict, kind: str) -> dict:
    """(time, set, weight) -> (W[H], S[x, H], Wx[x, H]): the EMA weight, and per value the weighted sum and its weight."""
    sets, weights = CFG[kind]
    n = a["t_local"].size
    X = np.column_stack([a[x] if x in a else np.full(n, np.nan) for x in VALS[kind]])
    fin = np.isfinite(X)
    X0 = np.where(fin, X, 0.0)
    out = {}
    for tname in TIMES:
        for sset in sets:
            m = a[f"set_{sset}"]
            t = a[tname][m]
            if t.size == 0:
                continue
            e = np.exp(-(t[-1] - t)[:, None] * LN2 / HALF[None, :])
            for w in weights:
                wv = np.nan_to_num(a[w][m])
                out[(tname, sset, w)] = (wv @ e, (wv[:, None] * X0[m]).T @ e, (wv[:, None] * fin[m]).T @ e)
    return out


def fit_a(x: np.ndarray, z: np.ndarray) -> float:
    """Least-squares a >= 0 in z = a x, refitted on the better half (a few stocks lose ticks in PM's feed)."""
    keep = np.isfinite(x) & np.isfinite(z)
    a = 0.0
    for _ in range(3):
        den = (x[keep] ** 2).sum()
        if den <= 0 or keep.sum() < 10:
            return a
        a = float(np.clip((x[keep] * z[keep]).sum() / den, 0.0, 1.5))
        r = np.abs(z - a * x)
        keep = keep & (r <= np.nanquantile(r[keep], 0.5))
    return a


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
    obs, tabs, kind_of, qrows, tstamps, hist = [], {}, {}, [], [], []
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
        tabs[d], last_t = {}, []
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            if gs is None or gs.height < 2:
                continue
            af = E.arrays(gf, "fast") if gf is not None and gf.height >= 2 else None
            asl = E.arrays(gs, "slow")
            streams = {"slow": asl, "daily": daily_stream(af, asl)}
            if af is not None:
                streams["fast"] = af
            for a in streams.values():
                if "dt" in a:
                    for h in ALPHA:
                        a[f"alpha{h}"] = 1 - np.exp(-a["dt"] * LN2 / h)
            tabs[d][sid] = (int(gs["mkt"][0]), {k: tables(a, k) for k, a in streams.items()})
            last_t.append((sid, af["t_local"][-1] if af is not None else np.nan, asl["t_local"][-1]))
            q = {"sid": sid, "date": int(d), "mkt": int(gs["mkt"][0]), "n_tr": float(gs.height),
                 "n_snap": float(gf.height) if gf is not None else np.nan}
            if af is not None:
                tl = af["t_local"]
                late = tl >= tl[-1] - 300
                q["spr_abs"] = float(np.nanmean(af["spread_abs"][late])) if np.isfinite(af["spread_abs"][late]).any() else np.nan
                q["spr_rel"] = float(np.nanmean(af["spread"][late])) if np.isfinite(af["spread"][late]).any() else np.nan
                q["dt_mean"] = float(np.nanmean(np.diff(tl[tl >= tl[-1] - 600]))) if (tl >= tl[-1] - 600).sum() > 1 else np.nan
            qrows.append(q)
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
        qrows[-len(tabs[d]):] = [{**r, **(qpm.filter(pl.col("sid") == r["sid"]).to_dicts() or [{}])[0]} for r in qrows[-len(tabs[d]):]]
        for f_, tag in ((args.dcp_dir / f"{d}.jz", "D"), (args.dcp_dir / f"{prev[-1]}.jz", "D-1")):
            with gzip.open(f_) as fh:
                raw = json.loads(fh.read())
            hist += [(int(d), tag, int(s), m.group(1), int(m.group(2)), float(v)) for s, kv in raw.items() for k, v in kv.items()
                     if v is not None and (m := HIST.match(k))]
        print(f"{d} (prev {prev[-1]}): {len(tabs[d])} stocks ({time.perf_counter() - t0:.0f}s)")
    if not obs:
        return
    ob = pl.concat(obs).with_columns(pl.col("y0").fill_null(0.0), pl.col("s0").fill_null(0.0))
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. multi-day EMA fit per prefix
    rows = []
    for p in sorted(ob["p"].unique().to_list()):
        o = ob.filter(pl.col("p") == p)
        kinds = [kind_of[p]] if kind_of.get(p) else ["fast", "slow", "daily"]
        recs = [(r["d"], r["sid"], r["wgt"], r["sum"], r["y0"], r["s0"]) for r in o.iter_rows(named=True)
                if r["sid"] in tabs.get(r["d"], {}) and r["wgt"] and np.isfinite(r["wgt"]) and r["wgt"] != 0]
        if len(recs) < 30:
            continue
        mkt = np.array([tabs[d][sid][0] for d, sid, *_ in recs])
        y, s, y0, s0 = (np.array([r[i] for r in recs], float) for i in (2, 3, 4, 5))
        sz = mkt == SZSE
        best = None
        for kind in kinds:
            keys = {k for d, sid, *_ in recs for k in tabs[d][sid][1].get(kind, {})}
            for key in sorted(keys):
                got = [tabs[d][sid][1].get(kind, {}).get(key) for d, sid, *_ in recs]
                ok = np.array([g is not None for g in got])
                if ok[sz].sum() < 20:
                    continue
                Wm = np.vstack([g[0] if g is not None else np.full(HALF.size, np.nan) for g in got])
                for hi in range(HALF.size):
                    a = fit_a(y0[sz & ok], (y - Wm[:, hi])[sz & ok])
                    r = np.abs(y - a * y0 - Wm[:, hi]) / np.abs(y)
                    mre = float(np.nanmedian(r[sz & ok]))
                    if best is None or mre < best[0]:
                        best = (mre, kind, key, hi, a, float(np.mean(r[sz & ok] <= 1e-4)), float(np.mean(r[~sz & ok] <= 1e-4)) if (~sz & ok).any() else np.nan)
        if best is None:
            continue
        mre, kind, key, hi, a, ok4, ok4_sh = best
        pm_v = s / y
        xs = []
        for xi, xn in enumerate(VALS[kind]):
            err = []
            for (d, sid, *_), a_y0, a_s0, v in zip(recs, a * y0, a * s0, pm_v, strict=True):
                g = tabs[d][sid][1].get(kind, {}).get(key)
                if g is None:
                    err.append(np.nan)
                    continue
                den = a_y0 + g[2][xi, hi]
                err.append(abs((a_s0 + g[1][xi, hi]) / den - v) / max(abs(v), 1e-12) if den else np.nan)
            e = np.array(err)[sz]
            if np.isfinite(e).any():
                xs.append((xn, float(np.nanmedian(e)), float(np.nanmean(e <= 1e-4))))
        xs.sort(key=lambda t: t[1])
        gap = float(-np.log(a) * HALF[hi] / LN2) if 0 < a < 1 else (0.0 if a >= 1 else np.inf)
        rows.append({"p": p, "clock": kind_of.get(p) or "-", "kind": kind, "time": key[0].removeprefix("t_"), "set": key[1],
                     "w": key[2], "half": float(HALF[hi]), "a": a, "gap_s": gap, "mre_w": mre, "ok4_w": ok4, "ok4_w_SSE": ok4_sh,
                     "x": xs[0][0] if xs else None, "mre_x": xs[0][1] if xs else None, "ok4_x": xs[0][2] if xs else None,
                     "x2": xs[1][0] if len(xs) > 1 else None, "mre_x2": xs[1][1] if len(xs) > 1 else None})
    fit = pl.DataFrame(rows).sort("mre_w")
    fit.write_csv(args.out / "rest_ema.csv")

    # 2. power fit of the unsolved values
    qd = pl.DataFrame(qrows).with_columns(pl.col("sid").cast(pl.Int64))
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
        j = vals.filter(pl.col("p") == p).join(qd, on=["sid", "date"]).filter(pl.col("mkt") == SZSE)
        if j.height < 30:
            continue
        v = j["v"].to_numpy()
        for combo in [(c,) for c in qnames] + list(itertools.combinations(qnames, 2)):
            Q = np.column_stack([j[c].to_numpy().astype(float) for c in combo])
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

    # 4. histogram inventory
    hf = pl.DataFrame(hist, schema=["date", "tag", "sid", "p", "bin", "c"], orient="row")
    tot = hf.group_by("date", "tag", "sid", "p").agg(pl.col("c").sum().alias("tot"))
    inc = tot.filter(pl.col("tag") == "D").join(tot.filter(pl.col("tag") == "D-1").select("date", "sid", "p", pl.col("tot").alias("tot0")),
                                                on=["date", "sid", "p"]).with_columns((pl.col("tot") - pl.col("tot0")).alias("inc"))
    hsum = hf.filter(pl.col("tag") == "D").group_by("p").agg(
        pl.col("bin").n_unique().alias("bins"), pl.col("bin").min().alias("bin_min"), pl.col("bin").max().alias("bin_max"),
        (pl.col("c") == pl.col("c").round()).mean().alias("integer")).join(
        tot.filter(pl.col("tag") == "D").group_by("p").agg(pl.col("tot").median().alias("tot_med")), on="p").join(
        inc.group_by("p").agg(pl.col("inc").quantile(0.1).alias("inc_p10"), pl.col("inc").median().alias("inc_med"),
                              pl.col("inc").quantile(0.9).alias("inc_p90"), (pl.col("inc") < 0).mean().alias("inc_neg")), on="p").sort("p")
    hsum.write_csv(args.out / "rest_hist.csv")

    with pl.Config(tbl_rows=80, tbl_cols=18, tbl_width_chars=250, fmt_str_lengths=16, float_precision=5):
        print("\n== 1. multi-day EMA fit (wgt_D = a wgt_D-1 + today's EMA; SZSE; gap_s: overnight time a stands for at that half-life)")
        print(fit)
        if pwf is not None:
            print("\n== 2. power fit of unsolved values, best 6 each (log|v| = log const + b1 log q1 + b2 log q2, SZSE; neg = share < 0)")
            print(pwf.group_by("p", maintain_order=True).head(6))
        print("\n== 3. clock stamps of the averages on no intraday clock (trading-clock seconds; same = share at the median)")
        print(tsum)
        print("\n== 4. histograms: bins, labels, integer counts, median total, day-over-day growth of the total")
        print(hsum)
    print(f"tables: {args.out}/rest_ema.csv, rest_hist.csv ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
