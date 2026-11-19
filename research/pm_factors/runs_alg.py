r"""Check the decoded alg.cpp (runS decision path) against the live log of one day.

Claims from the reconstruction (sht-demo feat/run-s 07b0cd0), each tested on the parsed 224 log:
A  participation gate: no decision unless ourVol / mktVol < 0.15 (the 300 s averages printed on every snapshot as
   emsOurVol / emsMktVol); room = (0.15 x mktVol - ourVol) / 0.85, which should be the logged winsdQty.
B  lthr: is the logged lthr the close relaxation v18 = -w x (cr + msr) with w = (vpen / 0.001)^2 (the time fraction since
   13:00) and msr the stock's start-of-day value (stsfile stats/sod)? Zero before 13:00 follows from w = 0.
C  the order condition |fa x asf| >= cr + x + v18 with x = the logged hs: share of orders that satisfy it, and the
   rule "condition and lot-rounded min(|executp - cpos|, winsdQty) >= 100 and trade sign == sign(fa)" as a classifier of
   "order made" (top-of-book size is not in the log, so precision below 1 is expected; recall should be ~1).
D  the no-trade band: lo / hi = (fa -+ (x + cr) / 1.5) / (2 x vpen x (vr / 3) x mp) - pos; target 0 inside the band, else
   the nearer edge, clamped to +-poscst; compared with the logged executp (exact share and ratio quantiles), for x = hs.
Read-only.

    python research/pm_factors/runs_alg.py --orders /work/crucible_data/pm_orders --date 20261118 --sts /work/crucible_data/sts_224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
CR, ASF, PART = 0.0004, 0.8, 0.15


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    return np.abs(a - b) <= tol * np.maximum(np.abs(b), 1e-300)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/sts_224"))
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    sn = pl.read_parquet(root / "snaps.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"])
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "ev_mp", "ev_cpos", "ev_poscst", "ev_vpen", "ev_vr", "ev_asq") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num).with_columns((c("nullptr") == 0).alias("order"), ((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod"))
    print(f"== {a.date}: {sn.height:,} snapshot lines, {d.height:,} decisions, {int(d['order'].sum()):,} with an order")

    print("-- A participation: ourVol / mktVol on every snapshot line, and whether a decision follows")
    j = sn.join(d.select("sid", "src_epoch", pl.lit(True).alias("dec"), "order", "winsdQty"), on=["sid", "src_epoch"], how="left").with_columns(c("dec").fill_null(False), c("order").fill_null(False))
    ov, mv = j["ems_our_vol"].cast(pl.Float64).to_numpy(), j["ems_mkt_vol"].cast(pl.Float64).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ov / mv
    dec, od = j["dec"].to_numpy(), j["order"].to_numpy()
    print(f"   ratio q1/10/50/90/99 {q(r)}; ourVol == 0 on {float((ov == 0).mean()):.3f}; ratio >= 0.15 on {float((r >= PART).mean()):.5f}")
    for lo, hi in ((-1, 0), (0, 0.05), (0.05, 0.10), (0.10, 0.14), (0.14, 0.15), (0.15, 0.16), (0.16, 0.25), (0.25, 1e9)):
        m = (r > lo) & (r <= hi) if lo >= 0 else (r <= 0)
        if m.sum():
            print(f"   ratio in ({lo:g}, {hi:g}]: {int(m.sum()):>9,} snapshots, decision on {float(dec[m].mean()):.4f}, order on {float(od[m].mean()):.5f}")
    print(f"   decisions with ratio >= 0.15: {int((dec & (r >= PART)).sum()):,}; orders with ratio >= 0.15: {int((od & (r >= PART)).sum()):,}; largest ratio on an order {np.nanmax(np.where(od, r, np.nan)):.5f}")
    w = j["winsdQty"].cast(pl.Float64).to_numpy()
    room = (PART * mv - ov) / (1 - PART)
    m = dec & np.isfinite(w)
    print(f"   winsdQty == (0.15 mktVol - ourVol) / 0.85: exact (1e-9) on {float(close(w[m], room[m]).mean()):.4f} of {int(m.sum()):,} decisions; winsdQty / that: {q(w[m] / room[m])}")
    nd = ~dec
    print(f"   snapshots without a decision: {int(nd.sum()):,}; of them ratio >= 0.15: {float((r[nd] >= PART).mean()):.4f}, mktVol == 0: {float((mv[nd] == 0).mean()):.4f}, fa == 0: {float((j['fa'].cast(pl.Float64).to_numpy()[nd] == 0).mean()):.4f}")

    g = lambda k: d[k].to_numpy().astype(float) if k in d.columns else np.full(d.height, np.nan)  # noqa: E731
    fa, hs, lt, ex, pos, cp, mp, vp, vr, pc, wq, asq, sod = (g(k) for k in ("fa", "hs", "lthr", "executp", "pos", "ev_cpos", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "winsdQty", "asq", "sod"))
    order = d["order"].to_numpy()
    print("-- B lthr against the close relaxation v18 = -w (cr + msr), w = (vpen / 0.001)^2")
    wt = (vp / 0.001) ** 2
    aft = (sod >= 46800) & (lt != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = -lt / wt - CR
    print(f"   lthr == 0 before 13:00 on {float((lt[sod < 46800] == 0).mean()):.4f}; after 13:00 lthr != 0 on {float((lt[sod >= 46800] != 0).mean()):.4f}")
    print(f"   after 13:00: -lthr / w - cr (the implied msr) q1/10/50/90/99 {q(implied[aft])}; distinct values per stock (median) "
          f"{pl.DataFrame({'sid': d['sid'].to_numpy()[aft], 'v': np.round(implied[aft], 12)}).group_by('sid').agg(pl.col('v').n_unique())['v'].median():.0f}")
    try:
        from ic_score import pm_daily
        pdv = pm_daily(a.sts, a.date, sorted(set(d["sid"].to_list())), "on")
        msr = d.select("sid").join(pdv.select("sid", "pm.msr", "pm.mb", "pm.vr"), on="sid", how="left")
        for k in ("pm.msr", "pm.mb", "pm.vr"):
            v = msr[k].to_numpy().astype(float)
            for nm, cand in ((f"-w (cr + {k})", -wt * (CR + v)), (f"-w {k}", -wt * v), (f"-sqrt(w) (cr + {k})", -np.sqrt(wt) * (CR + v)), (f"-w (cr + {k} / 2)", -wt * (CR + v / 2))):
                print(f"   lthr == {nm}: exact on {float(close(lt[aft], cand[aft], 1e-6).mean()):.4f}; lthr / that {q(lt[aft] / cand[aft])}")
    except Exception as e:
        print(f"   start-of-day table not read ({e!r}); the implied msr above still tells whether it is one value per stock")

    print("-- C the order condition |fa| x asf >= cr + hs + lthr")
    cond = np.abs(fa) * ASF >= CR + hs + lt
    cond0 = np.abs(fa) * ASF >= CR + hs
    tr = ex - cp
    lot = np.floor(np.minimum(np.abs(tr), wq) / 100) * 100
    agree = np.sign(tr) == np.sign(fa)
    print(f"   condition holds on {float(cond.mean()):.4f} of decisions, on {float(cond[order].mean()):.4f} of orders; without the lthr term on {float(cond0[order].mean()):.4f} of orders")
    print(f"   margin (|fa| asf - cr - hs - lthr) / hs on orders q1/10/50/90/99 {q(((np.abs(fa) * ASF - CR - hs - lt) / hs)[order])}")
    for name, rule in (("condition", cond), ("condition, lot >= 100", cond & (lot >= 100)), ("condition, lot >= 100, sign", cond & (lot >= 100) & agree),
                       ("condition, lot >= 100, sign, executp != pos", cond & (lot >= 100) & agree & (np.abs(ex - pos) > 1e-6)),
                       ("no-lthr condition, lot >= 100, sign", cond0 & (lot >= 100) & agree)):
        tp = float((rule & order).sum())
        print(f"   rule [{name}]: predicts {int(rule.sum()):,}; precision {tp / max(rule.sum(), 1):.4f}, recall {tp / max(order.sum(), 1):.4f}")
    miss = order & ~(cond & (lot >= 100) & agree)
    print(f"   orders outside the last full rule: {int(miss.sum())}; of them condition false {int((miss & ~cond).sum())}, lot < 100 {int((miss & (lot < 100)).sum())}, sign differs {int((miss & ~agree).sum())}, executp == 0 {int((miss & (ex == 0)).sum())}")

    print("-- D the no-trade band against the logged executp (x = hs)")
    with np.errstate(divide="ignore", invalid="ignore"):
        den = 2 * vp * (vr / 3) * mp
        lo, hi = (fa - (hs + CR) / 1.5) / den - pos, (fa + (hs + CR) / 1.5) / den - pos
    for nm, (l_, h_) in (("as given", (lo, hi)), ("edges swapped in sign", (-hi, -lo))):
        tgt = np.where(l_ > 0, l_, np.where(h_ < 0, h_, 0.0))
        tgt = np.clip(tgt, -pc, pc)
        ok = np.isfinite(tgt) & np.isfinite(ex)
        print(f"   {nm}: inside the band (target 0) on {float((tgt[ok] == 0).mean()):.4f} (logged executp == 0 on {float((ex[ok] == 0).mean()):.4f}); both zero or both non-zero on {float(((tgt[ok] == 0) == (ex[ok] == 0)).mean()):.4f};"
              f" exact where both non-zero {float(close(ex[ok & (tgt != 0) & (ex != 0)], tgt[ok & (tgt != 0) & (ex != 0)], 1e-6).mean()) if (ok & (tgt != 0) & (ex != 0)).any() else float('nan'):.4f};"
              f" executp / target there {q((ex / tgt)[ok & (tgt != 0) & (ex != 0)])}")
        for tag, mseg in (("before 13:00", sod < 46800), ("after 13:00", sod >= 46800)):
            mm = ok & mseg
            print(f"      {tag}: zero / non-zero agreement {float(((tgt[mm] == 0) == (ex[mm] == 0)).mean()):.4f}; sign agreement where both non-zero {float((np.sign(tgt) == np.sign(ex))[mm & (tgt != 0) & (ex != 0)].mean()) if (mm & (tgt != 0) & (ex != 0)).any() else float('nan'):.4f}")
    print("[done]")


if __name__ == "__main__":
    main()
