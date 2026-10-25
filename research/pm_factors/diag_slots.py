r"""Stage 4g: PM's per-slot profiles (``<p>.c.<bin>``) as today's intraday time grids, against our ticks.

``diag_profile`` showed the bin counts divide the 14220 s continuous session (09:30-11:30,
13:00-14:57) exactly: zzhj 474 = 30 s, zzfb 711 = 20 s, zzgh / zzjb 948 = 15 s, zzgh2 / zzjb2
1422 = 10 s. Slot k covers trading-clock seconds [k L, (k + 1) L). Tested here:

* **zzfb** = cumulative amount (or volume) by the end of each slot -- exchange or local time,
  with or without the open auction, slot end included or not; exact share per variant;
* **zzjb / zzjb2** (4 in 94 % of slots, else 0) = activity flag -- the zero slots against slots
  with no trade, no snapshot, no price change, no volume change;
* **zzgh / zzgh2** (0..4, ~20 % each) = quintile code of a per-slot quantity -- within the stock's
  day or across stocks in the slot; agreement share (chance = 0.2);
* **zzhj** (mostly 0, price-like values at the start of the array) = an event list -- number of
  non-zero entries against counts of new highs / lows (snapshot and trade), and the share of the
  values found among those events' prices.

    python research/pm_factors/diag_slots.py --dates 20260923,20260929,20261009
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader
import dcp_ema as E  # tick loading

SESSION = 14220.0  # 09:30-11:30 + 13:00-14:57 in trading-clock seconds
SZSE = 3554
HIST = re.compile(r"^(.+)\.c\.(-?\d+)$")


def profiles(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """Map profile -> sid -> bin values."""
    with gzip.open(path) as fh:
        raw = json.loads(fh.read())
    cells: dict[str, dict[int, dict[int, float]]] = {}
    for s, kv in raw.items():
        for k, v in kv.items():
            if v is not None and (m := HIST.match(k)):
                cells.setdefault(m.group(1), {}).setdefault(int(s), {})[int(m.group(2))] = float(v)
    out = {}
    for p, by in cells.items():
        nb = max(max(c) for c in by.values()) + 1
        out[p] = {}
        for sid, c in by.items():
            a = np.zeros(nb)
            a[list(c)] = list(c.values())
            out[p][sid] = a
    return out


def slot_of(t: np.ndarray, n: int, right: bool = False) -> np.ndarray:
    """Slot index 0..n-1 of trading-clock seconds t (-1 before the session, n after it)."""
    L = SESSION / n
    k = np.ceil(t / L) - 1 if right else np.floor(t / L)
    return np.where(t < 0, -1, np.minimum(k, n)).astype(int)


def per_slot(t: np.ndarray, w: np.ndarray, n: int, right: bool = False) -> np.ndarray:
    """Sum of w per slot."""
    k = slot_of(t, n, right)
    ok = (k >= 0) & (k < n)
    out = np.zeros(n)
    np.add.at(out, k[ok], w[ok])
    return out


def last_per_slot(t: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    """Last value received in or before each slot (carried forward)."""
    ends = (np.arange(n) + 1) * SESSION / n
    i = np.searchsorted(t, ends, side="left") - 1
    return np.where(i >= 0, v[np.clip(i, 0, None)], np.nan)


def quintile(x: np.ndarray) -> np.ndarray:
    """Within-array quintile code 0..4 by rank (NaN stays NaN)."""
    out = np.full(x.size, np.nan)
    ok = np.isfinite(x)
    if ok.sum() < 5:
        return out
    r = np.argsort(np.argsort(x[ok], kind="stable"), kind="stable")
    out[ok] = np.floor(5 * r / ok.sum())
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    args = ap.parse_args()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    fb, jb, gh, hj, jbz, hjd = [], [], [], [], [], []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pf = profiles(f)
        pm = B.load_state(f)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(pm["sid"].unique().to_list())
        up = E.load(con, int(d), sids, stop)
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        gh_cs: dict[tuple[str, str], list] = {}  # cross-sectional quintiles need every stock's slot values
        for sid in sids:
            gs, gf = slow.get(sid), fast.get(sid)
            if gs is None or gf is None:
                continue
            mkt = int(gs["mkt"][0])
            te, tl = gs["t_exch"].to_numpy().astype(float), gs["t_local"].to_numpy().astype(float)
            s = gs["s"].to_numpy().astype(float)
            vol, amt, side, px = (gs[c].to_numpy().astype(float) for c in ("vol", "amt", "side", "price"))
            ft_l, ft_e = gf["t_local"].to_numpy().astype(float), gf["t_exch"].to_numpy().astype(float)
            fpx, fcv = gf["price"].to_numpy().astype(float), gf["cv"].to_numpy().astype(float)
            # zzfb: cumulative amount / volume at each slot end
            if "zzfb" in pf and sid in pf["zzfb"]:
                y = pf["zzfb"][sid]
                n = y.size
                ends = (np.arange(n) + 1) * SESSION / n
                for tname, t in (("exch", te), ("local", tl)):
                    o = np.argsort(t, kind="stable")
                    tt = t[o]
                    for auct in (True, False):
                        keep = np.ones(tt.size, bool) if auct else (s[o] >= 34200)
                        for wname, w in (("amt", amt[o]), ("vol", vol[o])):
                            cw = np.cumsum(w * keep)
                            for side_ in ("right", "left"):  # right: slot end included
                                i = np.searchsorted(tt, ends, side=side_) - 1
                                pred = np.where(i >= 0, cw[np.clip(i, 0, None)], 0.0)
                                fb.append((d, sid, mkt, "trades_" + tname, auct, wname, side_,
                                           float(np.mean(np.abs(pred - y) <= np.maximum(0.5, 1e-9 * np.abs(y)))),
                                           float(np.nanmedian(y / np.where(pred > 0, pred, np.nan)))))
                # the snapshot's own running total (total_amount / total_volume) carried to each slot end
                auct_amt = float(amt[s < 34200].sum())
                auct_vol = float(vol[s < 34200].sum())
                fca = gf["ca"].to_numpy().astype(float)
                for tname, ft in (("exch", ft_e), ("local", ft_l)):
                    for wname, cw, a0 in (("amt", fca, auct_amt), ("vol", fcv, auct_vol)):
                        for auct in (True, False):
                            for side_ in ("right", "left"):
                                i = np.searchsorted(ft, ends, side=side_) - 1
                                pred = np.where(i >= 0, cw[np.clip(i, 0, None)], 0.0) - (0.0 if auct else a0)
                                fb.append((d, sid, mkt, "snap_" + tname, auct, wname, side_,
                                           float(np.mean(np.abs(pred - y) <= np.maximum(0.5, 1e-9 * np.abs(y)))),
                                           float(np.nanmedian(y / np.where(pred > 0, pred, np.nan)))))
            # zzjb / zzjb2: which slots are 0
            for p in ("zzjb", "zzjb2"):
                if p not in pf or sid not in pf[p]:
                    continue
                y = pf[p][sid]
                n = y.size
                zero = y == 0
                jbz.append((d, sid, mkt, p, float(zero.mean()), int(gs.height), int(gf.height), sorted(set(y.tolist()))[:6]))
                for tname, t, ft in (("exch", te, ft_e), ("local", tl, ft_l)):
                    flags = {"no_trade": per_slot(t, np.ones(t.size), n) == 0,
                             "no_snap": per_slot(ft, np.ones(ft.size), n) == 0,
                             "no_px_chg": per_slot(ft[1:], (np.diff(fpx) != 0).astype(float), n) == 0,
                             "no_vol_chg": per_slot(ft[1:], (np.diff(fcv) != 0).astype(float), n) == 0}
                    for fn, fl in flags.items():
                        jb.append((d, sid, mkt, p, tname, fn, float(np.mean(fl == zero)), float(zero.mean()), float(fl.mean())))
            # zzgh / zzgh2: quintile codes
            for p in ("zzgh", "zzgh2"):
                if p not in pf or sid not in pf[p]:
                    continue
                y = pf[p][sid]
                n = y.size
                for tname, t, ft in (("exch", te, ft_e), ("local", tl, ft_l)):
                    lp = np.log(last_per_slot(ft, fpx, n))
                    ret = np.diff(lp, prepend=np.nan)
                    v_ = per_slot(t, vol, n)
                    nb_ = per_slot(t, amt * side, n)
                    a_ = per_slot(t, amt, n)
                    qs = {"ret": ret, "absret": np.abs(ret), "vol": v_, "n_tr": per_slot(t, np.ones(t.size), n),
                          "imb": np.where(a_ > 0, nb_ / np.where(a_ > 0, a_, 1), np.nan), "amt": a_,
                          "cumret": lp - lp[np.isfinite(lp)][0] if np.isfinite(lp).any() else lp}
                    n_ = per_slot(t, np.ones(t.size), n)
                    nbn = per_slot(t, side, n)
                    vb = per_slot(t, vol * side, n)
                    imbs = {"amt": (nb_, a_), "vol": (vb, v_), "cnt": (nbn, n_)}
                    for iname, (num, den) in imbs.items():
                        x = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
                        for ename, efill in (("empty_mid", 2.0), ("empty_0", 0.0), ("empty_skip", np.nan)):
                            for bname, edges in (("eq5", np.array([-0.6, -0.2, 0.2, 0.6])), ("sym", np.array([-0.5, -0.1, 0.1, 0.5])),
                                                 ("sign", np.array([-0.999999, -1e-9, 1e-9, 0.999999]))):
                                code = np.where(np.isfinite(x), np.searchsorted(edges, x, side="right").astype(float), efill)
                                ok = np.isfinite(code)
                                gh.append((d, sid, mkt, p, tname, f"imb_{iname}", f"bins_{bname}_{ename}",
                                           float(np.mean(code[ok] == y[ok])) if ok.any() else np.nan))
                    for qn, q in qs.items():
                        code = quintile(q)
                        ok = np.isfinite(code)
                        gh.append((d, sid, mkt, p, tname, qn, "within_day", float(np.mean(code[ok] == y[ok])) if ok.any() else np.nan))
                        gh_cs.setdefault((p, tname, qn), []).append((sid, mkt, q, y))
            # zzhj: event list
            if "zzhj" in pf and sid in pf["zzhj"]:
                y = pf["zzhj"][sid]
                nz = y[y != 0]
                cont = (gf["s"].to_numpy() >= 34200)
                p_ok = np.where(cont & np.isfinite(fpx) & (fpx > 0), fpx, np.nan)
                runmax, runmin = np.fmax.accumulate(p_ok), np.fmin.accumulate(p_ok)
                new_hi = np.isfinite(p_ok) & (np.concatenate([[True], p_ok[1:] > runmax[:-1]]))
                new_lo = np.isfinite(p_ok) & (np.concatenate([[True], p_ok[1:] < runmin[:-1]]))
                tcm = s >= 34200
                tmax, tmin = np.fmax.accumulate(np.where(tcm, px, np.nan)), np.fmin.accumulate(np.where(tcm, px, np.nan))
                t_hi = tcm & np.concatenate([[True], px[1:] > tmax[:-1]])
                t_lo = tcm & np.concatenate([[True], px[1:] < tmin[:-1]])
                ev = {"snap_new_hi": p_ok[new_hi], "snap_new_lo": p_ok[new_lo], "trade_new_hi": px[t_hi], "trade_new_lo": px[t_lo],
                      "snap_hi_or_lo": p_ok[new_hi | new_lo], "trade_hi_or_lo": px[t_hi | t_lo]}
                for en, evp in ev.items():
                    found = float(np.mean(np.isin(np.round(nz, 4), np.round(evp, 4)))) if nz.size else np.nan
                    hj.append((d, sid, mkt, en, int(nz.size), int(evp.size), found))
                if nz.size:
                    pos = np.flatnonzero(y)
                    last, hi_, lo_ = px[-1], np.nanmax(px), np.nanmin(px)
                    hjd.append((d, sid, mkt, int(nz.size), int(pos.min()), int(pos.max()), float(np.median(nz / last)),
                                float(np.min(nz / last)), float(np.max(nz / last)),
                                float(np.mean((nz >= lo_ - 1e-9) & (nz <= hi_ + 1e-9))), float(np.mean(nz == np.round(nz, 2))),
                                int(gs.height), float(amt.sum()), float(np.mean(np.diff(pos) == 1)) if pos.size > 1 else np.nan))
        # cross-sectional quintiles per slot
        for (p, tname, qn), rows in gh_cs.items():
            n = rows[0][3].size
            Q = np.vstack([r[2] if r[2].size == n else np.full(n, np.nan) for r in rows])
            C = np.full(Q.shape, np.nan)
            for k in range(n):
                C[:, k] = quintile(Q[:, k])
            for (sid, mkt, _, y), c in zip(rows, C, strict=True):
                ok = np.isfinite(c)
                gh.append((d, sid, mkt, p, tname, qn, "across_stocks", float(np.mean(c[ok] == y[ok])) if ok.any() else np.nan))
        print(f"done {d}")
    with pl.Config(tbl_rows=80, tbl_cols=12, tbl_width_chars=220, float_precision=4):
        if fb:
            f = pl.DataFrame(fb, schema=["d", "sid", "mkt", "time", "auction", "w", "end", "exact", "ratio"], orient="row")
            print("\n== zzfb: cumulative at each 20 s slot end (exact = share of slots equal; SZSE / SSE)")
            print(f.group_by("time", "auction", "w", "end").agg(
                pl.col("exact").filter(pl.col("mkt") == SZSE).median().alias("exact_SZSE"),
                pl.col("exact").filter(pl.col("mkt") != SZSE).median().alias("exact_SSE"),
                pl.col("ratio").filter(pl.col("mkt") == SZSE).median().alias("ratio_SZSE")).sort("exact_SZSE", descending=True).head(12))
        if jb:
            j = pl.DataFrame(jb, schema=["d", "sid", "mkt", "p", "time", "flag", "agree", "pm_zero", "flag_share"], orient="row")
            print("\n== zzjb / zzjb2: PM's zero slots against activity flags (agree = share of slots where both say the same)")
            print(j.filter(pl.col("mkt") == SZSE).group_by("p", "time", "flag").agg(
                pl.col("agree").median(), pl.col("pm_zero").median(), pl.col("flag_share").median()).sort(["p", "agree"], descending=[False, True]))
        if gh:
            g = pl.DataFrame(gh, schema=["d", "sid", "mkt", "p", "time", "q", "ref", "agree"], orient="row")
            print("\n== zzgh / zzgh2: codes (quintiles or fixed bins of the per-slot imbalance; agree: same code; chance 0.2)")
            print(g.filter(pl.col("mkt") == SZSE).group_by("p", "time", "q", "ref").agg(pl.col("agree").median()).sort(
                ["p", "agree"], descending=[False, True]).group_by("p", maintain_order=True).head(10))
        if hj:
            h = pl.DataFrame(hj, schema=["d", "sid", "mkt", "event", "pm_nonzero", "n_events", "found"], orient="row")
            print("\n== zzhj: non-zero entries against event counts and prices (found = share of PM values among the event prices)")
            print(h.filter(pl.col("mkt") == SZSE).group_by("event").agg(
                pl.col("pm_nonzero").median(), pl.col("n_events").median(),
                (pl.col("pm_nonzero") == pl.col("n_events")).mean().alias("count_equal"), pl.col("found").median()).sort("found", descending=True))
        if jbz:
            z = pl.DataFrame(jbz, schema=["d", "sid", "mkt", "p", "zero", "n_tr", "n_snap", "values"], orient="row")
            print("\n== zzjb / zzjb2: share of zero slots per stock (all-zero stocks vs scattered zeros) and those stocks' activity")
            print(z.group_by("p").agg((pl.col("zero") == 1).mean().alias("all_zero"), (pl.col("zero") == 0).mean().alias("no_zero"),
                                      ((pl.col("zero") > 0) & (pl.col("zero") < 1)).mean().alias("partly"),
                                      pl.col("n_tr").filter(pl.col("zero") == 1).median().alias("n_tr_all_zero"),
                                      pl.col("n_tr").filter(pl.col("zero") == 0).median().alias("n_tr_no_zero"),
                                      (pl.col("mkt") == SZSE).filter(pl.col("zero") == 1).mean().alias("all_zero_SZSE_share")))
            print(z.filter((pl.col("zero") > 0) & (pl.col("zero") < 1)).head(8))
        if hjd:
            hd = pl.DataFrame(hjd, schema=["d", "sid", "mkt", "n_nonzero", "first_pos", "last_pos", "v_over_last_med", "v_over_last_min",
                                           "v_over_last_max", "in_day_range", "two_decimals", "n_tr", "amt", "contiguous"], orient="row")
            print("\n== zzhj: what the non-zero values are (ratio to the last price, inside the day's range, 2-decimal prices, positions)")
            print(hd.select(pl.exclude("d", "sid")).describe())
            print("rank correlation of n_nonzero with:", {c: round(hd.select(pl.corr("n_nonzero", c, method="spearman")).item(), 3)
                                                         for c in ("n_tr", "amt", "v_over_last_med")})


if __name__ == "__main__":
    main()
