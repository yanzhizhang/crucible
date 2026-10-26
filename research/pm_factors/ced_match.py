r"""Stage 5f: which model features come from the static (daily) CED data.

A feature built from daily data -- size, past returns, historical volatility, average turnover,
Barra exposures -- is constant through the day, so no intraday candidate can match it; and an
intraday quantity scaled by a daily baseline (amount so far / average daily amount) needs the baseline.
This builds both kinds per stock and day from the crucible CED store (``research/ced``, read at
T-1 as CED does for EOD / Barra, SOD at T) and ranks them against every model feature's exact
training CDF (``pm_models_probe`` -> ``cdf.parquet``):

* daily: log market cap / float cap, 1 / 5 / 20-day returns, 20 / 60-day volatility of daily
  returns, log 20-day average amount, 20-day turnover, 20-day average range, the limit band width,
  every Barra exposure and the Barra stats (HBeta, PBeta, SRisk, TRisk, Capt);
* intraday over daily (``--ticks``): amount / volume traded since 09:30 at each 5-minute mark over the
  20-day average daily amount / volume, and volume over float shares;
* each as is, negated, as a cross-sectional rank in [0, 1] and [-0.5, 0.5], and cross-sectionally
  z-scored; plus, per feature, its distance to a uniform CDF (a rank feature shows up there).

    python research/pm_factors/ced_match.py --dates 20260922,20260923,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dcp_build as B  # state loader (the day's universe and PM's stop)
import feature_match as FM  # cdf_gap
from ced import store as S  # the CED store


def sid_of(sym: pl.Expr) -> pl.Expr:
    """Wind code ``600000.SH`` -> 600000."""
    return sym.str.extract(r"^(\d+)", 1).cast(pl.Int64)


def daily_candidates(d: str, sids: list[int]) -> pl.DataFrame | None:
    """One row per stock: the daily candidates of day d (history up to d-1)."""
    eod_days = [x for x in S.dates("daily_eod") if x < d]
    if len(eod_days) < 21:
        print(f"skip {d}: {len(eod_days)} daily_eod days before it in {S.STORE} (need 21+; run research/run_ced.py all-hist)")
        return None
    hist = S.read("daily_eod", eod_days[-min(60, len(eod_days))], eod_days[-1]).collect().with_columns(sid_of(pl.col("symbol")).alias("sid"))
    hist = hist.filter(pl.col("sid").is_in(sids)).sort("sid", "date").with_columns(
        (pl.col("close") * pl.col("adj_factor")).log().diff().over("sid").alias("lr"))
    last20 = hist.filter(pl.col("date") >= eod_days[-20])
    last5 = hist.filter(pl.col("date") >= eod_days[-5])
    sod_day = d if d in S.dates("daily_sod_hist") else None
    sod = S.read("daily_sod_hist" if sod_day else "daily_sod", d, d).collect().with_columns(sid_of(pl.col("symbol")).alias("sid"))
    agg = hist.group_by("sid").agg(pl.col("lr").std().alias("vol60"), pl.col("close").last().alias("close"),
                                   pl.col("preclose").last().alias("preclose"), pl.col("lr").last().alias("ret1"))
    a20 = last20.group_by("sid").agg(pl.col("lr").std().alias("vol20"), pl.col("lr").sum().alias("ret20"), pl.col("tot").mean().alias("adv20"),
                                     pl.col("vol").mean().alias("avgvol20"), ((pl.col("high") - pl.col("low")) / pl.col("preclose")).mean().alias("range20"))
    a5 = last5.group_by("sid").agg(pl.col("lr").sum().alias("ret5"))
    out = agg.join(a20, on="sid", how="left").join(a5, on="sid", how="left").join(
        sod.select("sid", "freeshare", "totshare", "maxp_allowed"), on="sid", how="left")
    out = out.with_columns((pl.col("close") * pl.col("totshare")).log().alias("lmcap"), (pl.col("close") * pl.col("freeshare")).log().alias("lfcap"),
                           pl.col("adv20").log().alias("ladv20"), (pl.col("avgvol20") / pl.col("freeshare")).alias("turn20"),
                           (pl.col("maxp_allowed") / pl.col("close")).log().alias("band"))
    for ds in ("barra_exp", "barra_stats"):
        bd = [x for x in S.dates(ds) if x < d]
        if bd:
            b = S.read(ds, bd[-1], bd[-1]).collect().with_columns(sid_of(pl.col("symbol")).alias("sid"))
            num = [c for c, t in b.schema.items() if t.is_numeric() and c not in ("sid",)]
            out = out.join(b.select("sid", *[pl.col(c).alias(f"{ds[6:]}.{c}") for c in num]), on="sid", how="left")
    return out.filter(pl.col("sid").is_in(sids)).drop("close", "preclose", "maxp_allowed")


def transforms(v: np.ndarray) -> dict[str, np.ndarray]:
    """As is, negated, cross-sectional rank [0, 1] / [-0.5, 0.5], cross-sectional z-score."""
    f = np.isfinite(v)
    r = np.full(v.size, np.nan)
    if f.sum() > 1:
        r[f] = np.argsort(np.argsort(v[f])) / (f.sum() - 1)
    z = (v - np.nanmean(v)) / np.nanstd(v) if f.sum() > 1 and np.nanstd(v) > 0 else np.full(v.size, np.nan)
    return {"": v, "-": -v, "rank:": r, "crank:": r - 0.5, "z:": z}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--cdf", type=Path, default=Path("data/reports/stage5/cdf.parquet"))
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--ticks", action=argparse.BooleanOptionalAction, default=True, help="also intraday amount / volume over daily baselines")
    ap.add_argument("--max-gap", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    args = ap.parse_args()
    cdf = {g["feature"][0]: (g["t"].to_numpy(), g["p"].to_numpy())
           for g in pl.read_parquet(args.cdf).filter(pl.col("nan_left") == 0).partition_by("feature") if g.height >= 8}
    gain = dict(pl.read_parquet(args.fingerprint).select("feature", "gain_share").iter_rows())
    print(f"CED store {S.STORE}; training-CDF points for {len(cdf)} features")
    pooled: dict[str, list[np.ndarray]] = {}
    con = None
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pm = B.load_state(f)
        sids = sorted(pm["sid"].unique().to_list())
        dc = daily_candidates(d, sids)
        if dc is None:
            continue
        cols = [c for c in dc.columns if c != "sid"]
        for c in cols:
            for pre, v in transforms(dc[c].to_numpy().astype(float)).items():
                pooled.setdefault(f"{pre}ced.{c}", []).append(v)
        if args.ticks:
            import dcp_ema as E
            import duckdb
            import feature_intraday as FI
            import ic_score as IC
            con = con or duckdb.connect(str(B.CATALOG), read_only=True)
            tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
            stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
            up = E.load(con, int(d), sids, stop)
            slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
            base = {r["sid"]: r for r in dc.select("sid", "adv20", "avgvol20", "freeshare").iter_rows(named=True)}
            per_mark: dict[str, list[np.ndarray]] = {}
            order = [s for s in sids if s in slow and s in base]
            for s in order:
                built = FI.rebuild(None, slow[s], IC.MARKS)
                b = base[s]
                with np.errstate(divide="ignore", invalid="ignore"):
                    vals = {"amt_adv20": built["zzqj.tpv"] / b["adv20"], "vol_avg20": built["zzqj.v"] / b["avgvol20"],
                            "vol_float": built["zzqj.v"] / b["freeshare"]}
                for k, v in vals.items():
                    per_mark.setdefault(k, []).append(np.asarray(v, float))
            for k, rows in per_mark.items():
                M = np.vstack(rows)  # (stocks, marks)
                for m in range(M.shape[1]):
                    for pre, v in transforms(M[:, m]).items():
                        pooled.setdefault(f"{pre}ced.{k}", []).append(v)
        print(f"done {d}: {len(cols)} daily fields")
    cand = FM.pool(pooled)
    if not cand:
        print("no candidates (CED store empty for these days?)")
        return
    rows = []
    for feat, pts in cdf.items():
        scored = sorted(((FM.cdf_gap(v, pts), k) for k, v in cand.items()), key=lambda t: t[0])
        # distance to a rank's CDF: uniform on [0, 1] (F(t) = t) or on [-0.5, 0.5] (F(t) = t + 0.5)
        u = float(min(np.mean(np.abs(np.clip(pts[0], 0, 1) - pts[1])), np.mean(np.abs(np.clip(pts[0] + 0.5, 0, 1) - pts[1]))))
        rows.append({"feature": feat, "gain_share": gain.get(feat, 0.0), "cdf_points": pts[0].size, "best_gap": scored[0][0],
                     "best": "; ".join(f"{k} {g:.3f}" for g, k in scored[:3]), "t_min": float(pts[0].min()), "t_max": float(pts[0].max()),
                     "uniform_gap": u})
    res = pl.DataFrame(rows).sort("gain_share", descending=True)
    args.out.mkdir(parents=True, exist_ok=True)
    res.write_csv(args.out / "ced_match.csv")
    with pl.Config(tbl_rows=120, tbl_cols=8, tbl_width_chars=240, fmt_str_lengths=110, float_precision=4):
        print(f"\n== model features whose training CDF a CED candidate matches within {args.max_gap} (by model gain)")
        print(res.filter(pl.col("best_gap") <= args.max_gap).select("feature", "gain_share", "best_gap", "best"))
        print("\n== rank-like features: CDF close to a uniform rank on [0, 1] or [-0.5, 0.5] (the CDF cannot say WHAT is ranked; IC decides)")
        print(res.filter((pl.col("uniform_gap") <= 0.02) & (pl.col("t_min") >= -0.6) & (pl.col("t_max") <= 1.05))
              .select("feature", "gain_share", "t_min", "t_max", "uniform_gap", "best"))
    print(f"table: {args.out / 'ced_match.csv'}")


if __name__ == "__main__":
    main()
