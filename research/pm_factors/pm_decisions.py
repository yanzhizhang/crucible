r"""PM's trade decisions (runS / ``sol/alg.cpp:251``) from ``pm_signals``' state files: what decides when it acts and how much.

A decision line carries ``sid, fa, pos, pp_, this->executp, asq, lthr, hs, liid, winsdQty, asf`` (pm_signals stores them as
``pp``, ``executp``, ...). Every decision on 224 so far is followed by ``order is nullptr``: no order reaches chrono yet (no
base position configured), so these are the rule's intended quantities, not fills.

Per day, printed:

* how many scored snapshots got a decision, and PM's |fa| with and without one (the gap between the largest |fa| without a
  decision and the smallest with one is the entry threshold);
* the constants (asf, liid, lthr == 0) and the quantiles of hs, winsdQty, asq, pos, pp;
* executp against its bounds: share equal to -asq / +asq (a cap), its sign against fa's, executp / winsdQty;
* least squares of executp on candidate terms (fa x winsdQty, (fa -/+ hs) x winsdQty, asq, pp, pos) over the rows that hit
  no cap, for each sign of fa: coefficients and R^2;
* winsdQty / emsMktVol from the matching snapshot (3/17 so far).

    python research/pm_factors/pm_decisions.py --state /work/crucible_data/pm_signals/state --dates 20261106,20261109
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

TERMS = ("fa_w", "fa_m_hs_w", "fa_p_hs_w", "asq", "pp", "pos", "one")


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return "/".join(f"{np.percentile(v, p):.4g}" for p in qs) if v.size else "-"


def ols(df: pl.DataFrame, y: str, terms: tuple[str, ...]) -> tuple[np.ndarray, float]:
    """Coefficients and R^2 of y on the terms (rows with every value finite)."""
    X = np.column_stack([df[t].to_numpy().astype(float) for t in terms])
    Y = df[y].to_numpy().astype(float)
    ok = np.isfinite(X).all(1) & np.isfinite(Y)
    if ok.sum() <= len(terms):
        return np.full(len(terms), np.nan), np.nan
    b, *_ = np.linalg.lstsq(X[ok], Y[ok], rcond=None)
    r = Y[ok] - X[ok] @ b
    return b, 1 - r.var() / Y[ok].var() if Y[ok].var() > 0 else np.nan


def day(path: Path, d: str) -> None:
    s = pl.read_parquet(path)
    snap, dec = s.filter(pl.col("kind") == "snap"), s.filter(pl.col("kind") == "decision")
    print(f"\n== {d}: {snap.height:,} scored snapshots, {dec.height:,} decisions; decision columns: "
          + ", ".join(c for c in dec.columns if dec[c].null_count() < dec.height))
    if dec.is_empty():
        return
    dec = dec.with_columns(pl.col(c).cast(pl.Float64) for c in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "asf")
                           if c in dec.columns)
    has = snap.join(dec.select("src_epoch").unique().with_columns(pl.lit(True).alias("dec")), on="src_epoch", how="left")
    has = has.with_columns(pl.col("dec").fill_null(False), pl.col("fa").abs().alias("afa"))
    a_y, a_n = has.filter(pl.col("dec"))["afa"].to_numpy(), has.filter(~pl.col("dec"))["afa"].to_numpy()
    print(f"  share of snapshots with a decision {has['dec'].mean():.3f}; |fa| with p0/p1/p10/p50 {q(a_y, (0, 1, 10, 50))}, "
          f"without p50/p90/p99/p100 {q(a_n, (50, 90, 99, 100))}")
    for c in ("asf", "liid"):
        if c in dec.columns:
            print(f"  {c}: values " + ", ".join(f"{v:g} x{n}" for v, n in dec.group_by(c).len().sort("len", descending=True).head(4).iter_rows()))
    for c in ("lthr", "hs", "winsdQty", "asq", "pos", "pp", "executp"):
        if c in dec.columns:
            v = dec[c].to_numpy()
            print(f"  {c}: p1/p10/p50/p90/p99 {q(v)}; share == 0 {np.mean(v == 0):.3f}")
    ex, asq, fa = dec["executp"].to_numpy(), dec["asq"].to_numpy(), dec["fa"].to_numpy()
    lo, hi = np.isclose(ex, -asq), np.isclose(ex, asq)
    print(f"  executp == -asq {lo.mean():.3f} (fa < 0 among them {np.mean(fa[lo] < 0) if lo.any() else np.nan:.3f}), "
          f"== +asq {hi.mean():.3f} (fa > 0 among them {np.mean(fa[hi] > 0) if hi.any() else np.nan:.3f}); "
          f"sign(executp) == sign(fa) {np.mean(np.sign(ex) == np.sign(fa)):.3f}; integer executp {np.mean(np.isclose(ex, np.round(ex))):.3f}, "
          f"multiple of 100 {np.mean(np.isclose(ex % 100, 0) | np.isclose(ex % 100, 100)):.3f}")
    pos, pp = dec["pos"].to_numpy(), dec["pp"].to_numpy()
    nz, pz = ex != 0, pos != 0  # zero rows hid everything above: 3/4 of executp and of pos are 0
    lth, hs_ = dec["lthr"].to_numpy(), dec["hs"].to_numpy()
    print(f"  non-zero only: executp {nz.mean():.3f} of rows, pos {pz.mean():.3f}, both {np.mean(nz & pz):.3f}, executp without pos "
          f"{np.mean(nz & ~pz):.3f}, pos without executp {np.mean(pz & ~nz):.3f}")
    if pz.any():
        print(f"  pos != 0: sign(pos) == sign(fa) {np.mean(np.sign(pos[pz]) == np.sign(fa[pz])):.3f}; executp == pos {np.mean(np.isclose(ex[pz], pos[pz])):.3f}, "
              f"== pos - pp {np.mean(np.isclose(ex[pz], pos[pz] - pp[pz])):.3f}, == clip(pos, +-asq) "
              f"{np.mean(np.isclose(ex[pz], np.clip(pos[pz], -asq[pz], asq[pz]))):.3f}; executp / pos p1/p10/p50/p90/p99 {q(ex[pz] / pos[pz])}; "
              f"pos multiple of 100 {np.mean(np.isclose(pos[pz] % 100, 0) | np.isclose(pos[pz] % 100, 100)):.3f}; "
              f"pos / winsdQty p1/p10/p50/p90/p99 {q(pos[pz] / dec['winsdQty'].to_numpy()[pz])}")
        for nm, sel in (("pos > 0", pos > 0), ("pos < 0", pos < 0), ("pos == 0", ~pz)):
            print(f"  {nm} ({sel.sum():,}): fa p10/p50/p90 {q(fa[sel], (10, 50, 90))}; fa - hs {q(fa[sel] - hs_[sel], (10, 50, 90))}; "
                  f"fa + lthr {q(fa[sel] + lth[sel], (10, 50, 90))}; fa - lthr {q(fa[sel] - lth[sel], (10, 50, 90))}; "
                  f"asq > 0 {np.mean(asq[sel] > 0):.3f}")
    if nz.any():
        a_ = asq > 0
        print(f"  executp != 0 and asq > 0 ({np.sum(nz & a_):,}): |executp| == asq {np.mean(np.isclose(np.abs(ex[nz & a_]), asq[nz & a_])):.3f}, "
              f"|executp| < asq {np.mean(np.abs(ex[nz & a_]) < asq[nz & a_] - 1e-9):.3f}; sign(executp) == sign(fa) {np.mean(np.sign(ex[nz]) == np.sign(fa[nz])):.3f}")
    sq = dec.sort("sid", "src_epoch").with_columns(  # per stock over time: does pos carry over, and what moves it
        pl.col("pos").shift(1).over("sid").alias("pos0"), pl.col("executp").shift(1).over("sid").alias("ex0"),
        pl.col("fa").shift(1).over("sid").alias("fa0"), ((pl.col("src_epoch") // 1_000_000_000 + 8 * 3600) % 86400).alias("tod"))
    sq = sq.filter(pl.col("pos0").is_not_null())
    if sq.height:
        dp = (sq["pos"] - sq["pos0"]).to_numpy()
        ch = dp != 0
        e0, tod = sq["ex0"].to_numpy(), sq["tod"].to_numpy()
        print(f"  per stock over time ({sq.height:,} consecutive pairs): pos unchanged {np.mean(~ch):.3f}; on a change: dpos == previous "
              f"executp {np.mean(np.isclose(dp[ch], e0[ch])):.3f}, sign(dpos) == sign(fa) {np.mean(np.sign(dp[ch]) == np.sign(sq['fa'].to_numpy()[ch])):.3f}, "
              f"from 0 {np.mean(sq['pos0'].to_numpy()[ch] == 0):.3f}, to 0 {np.mean(sq['pos'].to_numpy()[ch] == 0):.3f}; dpos p1/p10/p50/p90/p99 {q(dp[ch])}")
        if ch.any():
            hh = (tod[ch] // 3600 * 100 + tod[ch] % 3600 // 60)
            top = pl.Series(hh).value_counts().sort("count", descending=True).head(8)
            print("  pos changes by HHMM (top 8): " + ", ".join(f"{int(a):04d} x{b}" for a, b in top.iter_rows())
                  + f"; second of minute == 0 {np.mean(tod[ch] % 60 == 0):.3f}")
    w = dec["winsdQty"].to_numpy()
    print(f"  executp / winsdQty p1/p10/p50/p90/p99 {q(ex / np.where(w > 0, w, np.nan))}")
    t = dec.with_columns((pl.col("fa") * pl.col("winsdQty")).alias("fa_w"), ((pl.col("fa") - pl.col("hs")) * pl.col("winsdQty")).alias("fa_m_hs_w"),
                         ((pl.col("fa") + pl.col("hs")) * pl.col("winsdQty")).alias("fa_p_hs_w"), pl.lit(1.0).alias("one"))
    free = t.filter(pl.Series(~(lo | hi)))
    for name, part in (("fa > 0, no cap", free.filter(pl.col("fa") > 0)), ("fa < 0, no cap", free.filter(pl.col("fa") < 0))):
        print(f"  {name}: {part.height:,} rows")
        for terms in (("fa_w",), ("fa_m_hs_w",), ("fa_p_hs_w",), ("fa_w", "asq", "pp", "pos", "one"), TERMS):
            b, r2 = ols(part, "executp", terms)
            print(f"    executp ~ {' + '.join(f'{c:.4g} {n}' for c, n in zip(b, terms, strict=True))}: R^2 {r2:.4f}")
    if "ems_mkt_vol" in snap.columns:
        j = dec.drop([c for c in ("ems_mkt_vol",) if c in dec.columns]).join(snap.select("src_epoch", "sid", "ems_mkt_vol").unique(["src_epoch", "sid"]), on=["src_epoch", "sid"], how="inner")
        r = (j["winsdQty"] / j["ems_mkt_vol"]).to_numpy()
        print(f"  winsdQty / emsMktVol over {j.height:,} rows: p1/p50/p99 {q(r, (1, 50, 99))} (3/17 = {3 / 17:.6f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, default=Path("/work/crucible_data/pm_signals/state"))
    ap.add_argument("--dates", required=True, help="YYYYMMDD,...")
    a = ap.parse_args()
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        f = a.state / f"date={d}.parquet"
        if f.exists():
            day(f, d)
        else:
            print(f"skip {d}: no {f}")


if __name__ == "__main__":
    main()
