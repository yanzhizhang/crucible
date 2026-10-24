r"""Stage 5d: rebuild the solved state every 5 minutes from our ticks, check it at the close, match features.

Joins the two lines of work. ``dcp_ema`` found the rules of PM's day-end EMAs (weight 1 per
update received, trading clock, a round half-life, a value); ``feature_match`` ranks expressions of
a family's state against the model fingerprint, but only on the close-of-day state. The model was
trained on features taken through the day, so this rebuilds the state at every 5-minute mark
(09:35 ... 14:55, local arrival time) and matches on that:

1. **rebuild** at each mark: the solved EMAs (``RULES``), running scalars (day high / low / open /
   last, volume and amount from 09:30, ``zzmh.hm``, ``zzgh.nb``) and the 10 s price lists
   ``zzxo.r`` / ``zzui.r``, written with the state file's own key names;
2. **check** the same code at PM's stop against PM's day-end state (SZSE: share within 1e-4), so a
   wrong rule shows up before it pollutes the match;
3. **match** every fingerprint feature whose stem has rebuilt keys: ``feature_match``'s candidate
   expressions, pooled over stocks x marks x days, ranked by distance to the threshold quantiles --
   or, with ``--calib`` (``fingerprint_calib.py``), by where the thresholds fall in each candidate's
   distribution against where they fall in true feature values (the trees split more in the tails,
   so raw threshold quantiles run 2-16x wider than the values).

    python research/pm_factors/feature_intraday.py --dates 20260923,20260929,20261009
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog, state loader
import dcp_ema as E  # tick loading and per-update arrays
import feature_match as FM  # candidate expressions and fingerprint scoring

SZSE = 3554
# prefix -> (updates, half-life s, value); fitted on 3 days by dcp_ema / diag_iema (docs/PM_REPRO.md)
RULES = {
    "zzpk.p": ("fast", 300, "price"), "zzab2.r": ("fast", 300, "price"), "zzup.pf": ("fast", 300, "price"),
    "zzxu.p": ("fast", 300, "price"), "zzup.ps": ("fast", 600, "price"), "zzxu.a": ("fast", 300, "hl"),
    "zzvx.trEmaD": ("fast", 30, "hl"), "zzvx.trEmaO": ("fast", 30, "hl"), "zzvx.trDEmaD": ("fast", 30, "hl"),
    "zzvx.trDEmaO": ("fast", 30, "hl"), "zzvx.rEmaD": ("fast", 30, "one"), "zzvx.rEmaO": ("fast", 30, "one"),
    "zzbd.pEmaS": ("fast", 100, "logp"), "zzeb.p": ("fast", 100, "mid"), "zzbd.rEmaS": ("fast", 200, "logret"),
    "zzba.p": ("fast", 240, "price"), "zzba.r": ("fast", 300, "tr_open"),  # value half-lives; weight rule still open
    "zzbd.pEmaT": ("slow", 100, "logp"), "zzbd.rEmaT": ("slow", 200, "logret"), "zzqj.r": ("slow", 30, "rvwap"),
}
MARKS = np.arange(300.0, 14101.0, 300.0)  # trading-clock seconds: 09:35 ... 14:55


def ema_at(a: dict, x: str, half: float, marks: np.ndarray) -> np.ndarray:
    """EMA of x (weight 1 per update, local arrival on the trading clock) as of each mark; NaN before any update."""
    t, v = a["t_local"], a[x]
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    lam = np.log(2) / half
    out = np.full(marks.size, np.nan)
    for i, (m, k) in enumerate(zip(marks, np.searchsorted(t, marks, side="right"), strict=True)):
        if k:
            lo = np.searchsorted(t, m - 60 / lam)  # older updates weigh < e^-60
            e = np.exp(-(m - t[lo:k]) * lam)
            out[i] = (e * v[lo:k]).sum() / e.sum()
    return out


def at_marks(t: np.ndarray, v: np.ndarray, marks: np.ndarray) -> np.ndarray:
    """Last value received at or before each mark."""
    k = np.searchsorted(t, marks, side="right") - 1
    return np.where(k >= 0, v[np.clip(k, 0, None)], np.nan)


def rebuild(g_fast: pl.DataFrame | None, g_slow: pl.DataFrame | None, marks: np.ndarray) -> dict[str, np.ndarray]:
    """One stock: key -> values at each mark, keys named as in PM's state file (EMAs as <p>.sum with <p>.wgt = 1)."""
    out: dict[str, np.ndarray] = {}
    af = E.arrays(g_fast, "fast") if g_fast is not None and g_fast.height >= 2 else None
    asl = E.arrays(g_slow, "slow") if g_slow is not None and g_slow.height >= 2 else None
    for p, (kind, half, x) in RULES.items():
        a = af if kind == "fast" else asl
        if a is not None:
            out[f"{p}.sum"], out[f"{p}.wgt"] = ema_at(a, x, half, marks), np.ones(marks.size)
    if asl is not None:
        t, px = asl["t_local"], asl["price"]
        s = g_slow["s"].to_numpy()
        hi, lo = np.fmax.accumulate(px), np.fmin.accumulate(px)
        for k in ("zzpk.high", "zzvx.high", "zzxu.high"):
            out[k] = at_marks(t, hi, marks)
        for k in ("zzpk.low", "zzvx.low", "zzxu.low"):
            out[k] = at_marks(t, lo, marks)
        out["zzpk.open"] = np.where(np.searchsorted(t, marks, side="right") > 0, px[0], np.nan)
        out["zzxu.last"] = out["zzxo.last"] = at_marks(t, px, marks)
        c = s >= 34200
        cv, ca = np.cumsum(asl["vol"] * c), np.cumsum(asl["amt"] * c)
        out["zzqj.v"], out["zzqj.tpv"] = at_marks(t, cv, marks), at_marks(t, ca, marks)
        # trade counts so far (zzgh.b's thresholds sit at a typical day's count): from 09:30 / all
        out["zzgh.ntr"], out["zzgh.ntr_all"] = at_marks(t, np.cumsum(c).astype(float), marks), at_marks(t, np.arange(1.0, t.size + 1), marks)
        cont = c & (s < 53820)
        buy = np.cumsum(asl["amt"] * cont * (asl["side"] > 0))
        sell = np.cumsum(asl["amt"] * cont * (asl["side"] < 0))
        tot = np.cumsum(asl["amt"] * cont)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["zzgh.nb"] = at_marks(t, (buy - sell) / tot, marks)
    if af is not None:
        t, px = af["t_local"], af["price"]
        s = g_fast["s"].to_numpy()
        out["zzmh.hm"] = at_marks(t, np.fmax.accumulate(np.where((s >= 34200) & (s < 53820), px, np.nan)), marks)
        okp = np.isfinite(px)
        for fam, n in (("zzxo", 30), ("zzui", 10)):  # last snapshot received by mark - (n-1-k) * 10 s
            for k in range(n):
                out[f"{fam}.r.{k}"] = at_marks(t[okp], px[okp], marks - (n - 1 - k) * 10)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--calib", type=Path, help="fingerprint_calib.json: score by calibrated threshold levels")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    frames: list[pl.DataFrame] = []  # one pseudo state per (day, mark)
    checks = []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pm = B.load_state(f)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
        sids = sorted(pm["sid"].unique().to_list())
        up = E.load(con, int(d), sids, stop)
        if up["slow"].is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
        slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
        close = np.array([tz])
        rows, chk = [], []
        for sid in sids:
            gf, gs = fast.get(sid), slow.get(sid)
            built = rebuild(gf, gs, MARKS)
            mkt = int((gf if gf is not None else gs)["mkt"][0]) if (gf is not None or gs is not None) else 0
            for k, v in built.items():
                rows.append(pl.DataFrame({"sid": sid, "mark": MARKS, "key": k, "pm": v}))
            fin = rebuild(gf, gs, close)  # the same code at PM's stop, against PM's own state
            for p in RULES:
                if f"{p}.sum" in fin:
                    chk.append((sid, mkt, p, float(fin[f"{p}.sum"][0])))
        long = pl.concat(rows).filter(pl.col("pm").is_finite()).with_columns(pl.col("sid").cast(pl.Int64))
        frames += [g.drop("mark") for g in long.partition_by("mark")]
        ours = pl.DataFrame(chk, schema=["sid", "mkt", "p", "ours"], orient="row").with_columns(pl.col("sid").cast(pl.Int64))
        pmv = pm_values(pm)
        c = ours.join(pmv, on=["sid", "p"]).with_columns(((pl.col("ours") - pl.col("pm")).abs()
                                                          / pl.col("pm").abs().clip(lower_bound=1e-12)).alias("err"))
        checks.append(c.with_columns(pl.lit(int(d)).alias("date")))
        print(f"{d}: rebuilt {len(sids)} stocks x {MARKS.size} marks ({time.perf_counter() - t0:.0f}s)")
    if not frames:
        return
    chk = pl.concat(checks)
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=5):
        print("\n== close check: the rebuild at PM's stop against PM's day-end state (share within 1e-4 / median rel. error)")
        print(chk.group_by("p").agg((pl.col("err") <= 1e-4).filter(pl.col("mkt") == SZSE).mean().alias("ok4_SZSE"),
                                    (pl.col("err") <= 1e-4).filter(pl.col("mkt") != SZSE).mean().alias("ok4_SSE"),
                                    pl.col("err").filter(pl.col("mkt") == SZSE).median().alias("mre_SZSE"))
              .sort("p"))
    fp = pl.read_parquet(args.fingerprint).with_columns(pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"))
    keys = set(frames[0]["key"].unique().to_list())
    stems = sorted(s for s in fp["stem"].unique().to_list() if any(k.startswith(s + ".") for k in keys))
    print(f"\nstems rebuilt: {stems}")
    out = []
    levels = FM.load_levels(args.calib)
    for stem in stems:
        pooled: dict[str, list[np.ndarray]] = {}
        for fr in frames:
            for k, v in FM.candidates(FM.atoms(fr, stem)).items():
                pooled.setdefault(k, []).append(v)
        cand = FM.pool(pooled)
        out += FM.rank(fp.filter(pl.col("stem") == stem), cand, args.top, levels)
        print(f"matched {stem}: {len(cand)} candidates ({time.perf_counter() - t0:.0f}s)")
    if not out:
        return
    res = pl.DataFrame(out).sort(["gain_share", "rank"], descending=[True, False])
    args.out.mkdir(parents=True, exist_ok=True)
    res.write_csv(args.out / "feature_intraday.csv")
    chk.write_csv(args.out / "feature_intraday_check.csv")
    with pl.Config(tbl_rows=200, tbl_cols=14, tbl_width_chars=250, fmt_str_lengths=40, float_precision=5):
        print(f"\n== best {args.top} candidates per feature on the 5-minute rebuild (score: |quantile gap| / threshold span,"
              " 0 = same; in_range: share inside the training range)")
        print(res.select("feature", "gain_share", "kind", "rank", "candidate", "score", "in_range",
                         "t_p10", "c_p10", "t_med", "c_med", "t_p90", "c_p90"))
    print(f"tables: {args.out}/feature_intraday.csv, feature_intraday_check.csv")


def pm_values(pm: pl.DataFrame) -> pl.DataFrame:
    """PM's day-end EMA values (sid, p, pm = sum / wgt)."""
    x = pm.filter(pl.col("key").str.contains(r"\.(sum|wgt)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt)$", 1).alias("p"), pl.col("key").str.extract(r"\.(sum|wgt)$", 1).alias("f"))
    w = x.pivot(on="f", index=["sid", "p"], values="pm")
    return w.filter(pl.col("wgt") != 0).select("sid", "p", (pl.col("sum") / pl.col("wgt")).alias("pm"))


if __name__ == "__main__":
    main()
