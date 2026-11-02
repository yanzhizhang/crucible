r"""Stage 4: what kind of quantity each PM sampler column is, read off the column's OWN statistics -- no tick data involved.

Per (file, V): range and sign, integer share and the value step (quantization: 100 = shares, 0.01 = price ticks),
cross-sectional elasticities of log value on log price level and log daily amount (price-like = (1, 0),
amount-like = (0, 1), share-like = (-1, 1), ratio-like = (0, 0)), the intraday profile (trend with time, U-shape,
monotone-cumulative share), the day-to-day memory (slope of pm_d on pm_{d-1} cell by cell), and the algebraic
identities that hold between columns of one file (V3 = V1 + V2, V1 * V2, V1 / V2, (V1 - V2) / (V1 + V2), V1 = k V2)
and between files (same column name, equal or proportional). The type guess is a rule over these fingerprints.

Price level = ZZDS c (per stock and day), daily amount = ZZMS n at the day's last bar, when those files are exported.

    python research/pm_factors/fingerprint.py --dates 20260923,20260924,20260929 --exports data/pm_export --out data/reports/stage4
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import pm_frame

REL = 1e-6


def load_family(exports: Path, name: str, dates: list[str]) -> tuple[pl.DataFrame | None, str]:
    """All days of one family as (code, minute, day, V, x); the I kind."""
    parts, kind = [], ""
    for d in dates:
        f = exports / f"{name}_{d}.parquet"
        if not f.exists():
            continue
        pm, vals, kind = pm_frame(f, None, None, None, None, None)
        parts.append(pm.select("code", "minute", pl.lit(d).alias("day"), "V", pl.col(vals[0]).alias("x")))
    return (pl.concat(parts) if parts else None), kind


def step_of(v: np.ndarray) -> float:
    """The value grid: the 5th percentile of the gaps between sorted distinct values (0 = continuous)."""
    u = np.unique(v[np.isfinite(v)])
    if u.size < 3:
        return 0.0
    g = np.diff(u)
    g = g[g > 1e-12]
    return float(np.quantile(g, 0.05)) if g.size else 0.0


def quant_share(v: np.ndarray, q: float) -> float:
    """Share of values that are integer multiples of q."""
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    r = np.abs(v / q - np.round(v / q))
    return float(np.mean(r < 1e-6))


def elasticity(t: pl.DataFrame, price: pl.DataFrame | None, amount: pl.DataFrame | None) -> tuple[float, float, float]:
    """Cross-sectional regression of log |value| (per stock and day, averaged over bars) on log price and log amount."""
    if price is None or amount is None:
        return float("nan"), float("nan"), float("nan")
    a = (t.filter(pl.col("x").is_finite() & (pl.col("x") != 0)).group_by("code", "day").agg(pl.col("x").abs().mean().alias("v"))
         .join(price, on=["code", "day"]).join(amount, on=["code", "day"]).filter((pl.col("v") > 0) & (pl.col("p") > 0) & (pl.col("a") > 0)))
    if a.height < 50:
        return float("nan"), float("nan"), float("nan")
    y, p, am = np.log(a["v"].to_numpy()), np.log(a["p"].to_numpy()), np.log(a["a"].to_numpy())
    X = np.column_stack([np.ones(y.size), p, am])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - float(((y - X @ coef) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    return float(coef[1]), float(coef[2]), r2


def profile(t: pl.DataFrame) -> tuple[float, float, float]:
    """Intraday shape: corr of the mean profile with time, with distance from midday (U-shape), monotone share per (stock, day)."""
    if t["minute"].n_unique() < 5:
        return float("nan"), float("nan"), float("nan")
    m = t.filter(pl.col("x").is_finite()).group_by("minute").agg(pl.col("x").mean().alias("v")).sort("minute")
    mn, v = m["minute"].to_numpy().astype(float), m["v"].to_numpy()
    if v.std() == 0:
        return 0.0, 0.0, float("nan")
    trend = float(np.corrcoef(mn, v)[0, 1])
    u = float(np.corrcoef(np.abs(mn - 735.0), v)[0, 1])
    mono = (t.filter(pl.col("x").is_finite()).sort("code", "day", "minute").group_by("code", "day")
            .agg((pl.col("x").diff().drop_nulls() >= 0).mean().alias("up")))["up"].mean()
    return trend, u, float(mono) if mono is not None else float("nan")


def memory(t: pl.DataFrame, dates: list[str]) -> tuple[float, float]:
    """Slope and correlation of pm_d on pm_{d-1} across cells (same stock and bar)."""
    xs, ys = [], []
    for d0, d1 in itertools.pairwise(dates):
        j = (t.filter(pl.col("day") == d0).select("code", "minute", pl.col("x").alias("x0"))
             .join(t.filter(pl.col("day") == d1).select("code", "minute", pl.col("x").alias("x1")), on=["code", "minute"])
             .filter(pl.col("x0").is_finite() & pl.col("x1").is_finite()))
        xs.append(j["x0"].to_numpy())
        ys.append(j["x1"].to_numpy())
    if not xs:
        return float("nan"), float("nan")
    x, y = np.concatenate(xs), np.concatenate(ys)
    if x.size < 50 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan")
    slope = float(np.cov(x, y)[0, 1] / x.var())
    return slope, float(np.corrcoef(x, y)[0, 1])


def guess(r: dict) -> str:
    """Rule-of-thumb type from the fingerprint."""
    tags = []
    if r["neg"] > 0.05 and r["max"] <= 1.0 + 1e-9 and r["min"] >= -1.0 - 1e-9:
        tags.append("imbalance[-1,1]")
    elif r["neg"] > 0.05:
        tags.append("signed")
    if r["int"] > 0.99 and r["min"] >= 0:
        tags.append("count" if r["q100"] < 0.9 else "shares(x100)")
    elif r["q100"] > 0.9:
        tags.append("shares(x100)")
    elif 0 < r["step"] <= 0.011 and r["q001"] > 0.9 and r["bp"] == r["bp"] and abs(r["bp"] - 1) < 0.25:
        tags.append("price")
    if r["bp"] == r["bp"]:
        if abs(r["bp"] - 1) < 0.25 and abs(r["ba"]) < 0.25:
            tags.append("~price-level")
        elif abs(r["ba"] - 1) < 0.3 and abs(r["bp"]) < 0.3:
            tags.append("~amount")
        elif abs(r["ba"] - 1) < 0.3 and abs(r["bp"] + 1) < 0.35:
            tags.append("~volume")
        elif abs(r["ba"]) < 0.2 and abs(r["bp"]) < 0.2:
            tags.append("~scale-free")
        elif 0.3 <= r["ba"] <= 0.75 and abs(r["bp"]) < 0.3:
            tags.append("~sqrt(amount)/count-like")
    if r["mono"] == r["mono"] and r["mono"] > 0.95:
        tags.append("cumulative")
    if r["ushape"] == r["ushape"] and r["ushape"] > 0.6:
        tags.append("U-profile")
    if r["mem_slope"] == r["mem_slope"] and r["mem_slope"] > 0.9 and r["mem_corr"] > 0.9:
        tags.append(f"long-memory(a~{r['mem_slope']:.2f})")
    return " ".join(tags) or "?"


def identities(t: pl.DataFrame, name: str) -> list[dict]:
    """Relations between columns of one file that hold cell by cell: equal / proportional pairs and three-column formulas."""
    wide = t.pivot(on="V", index=["code", "minute", "day"], values="x", aggregate_function="first")  # a file may carry several D rows per cell
    vs = [c for c in wide.columns if c not in ("code", "minute", "day")]
    cols = {v: wide[v].to_numpy().astype(float) for v in vs}
    out = []

    def share(lhs: np.ndarray, rhs: np.ndarray) -> float:
        ok = np.isfinite(lhs) & np.isfinite(rhs)
        return float(np.mean(np.abs(lhs[ok] - rhs[ok]) <= REL * np.abs(lhs[ok]) + 1e-9)) if ok.sum() >= 100 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        for a, b in itertools.combinations(vs, 2):
            x, y = cols[a], cols[b]
            if share(x, y) > 0.5:
                out.append({"file": name, "identity": f"{a} == {b}", "share": share(x, y)})
                continue
            ok = np.isfinite(x) & np.isfinite(y) & (y != 0)
            if ok.sum() >= 100:
                k = float(np.median(x[ok] / y[ok]))
                s = share(x, k * y)
                if s > 0.5 and abs(k - 1) > 1e-6:
                    out.append({"file": name, "identity": f"{a} == {k:.6g} * {b}", "share": s})
        for c in vs:
            for a, b in itertools.combinations([v for v in vs if v != c], 2):
                x, y, z = cols[a], cols[b], cols[c]
                for label, rhs in ((f"{a} + {b}", x + y), (f"{a} - {b}", x - y), (f"{b} - {a}", y - x), (f"{a} * {b}", x * y),
                                   (f"{a} / {b}", x / y), (f"{b} / {a}", y / x), (f"({a} - {b}) / ({a} + {b})", (x - y) / (x + y)),
                                   (f"({a} + {b}) / 2", (x + y) / 2)):
                    s = share(z, rhs)
                    if s > 0.5:
                        out.append({"file": name, "identity": f"{c} == {label}", "share": s})
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    a = ap.parse_args()
    dates = sorted(x for x in a.dates.split(",") if x)
    names = sorted({f.stem.rsplit("_", 1)[0] for f in a.exports.glob("*.parquet") if f.stem.rsplit("_", 1)[1] in dates})
    fams = {n: load_family(a.exports, n, dates) for n in names}
    price = amount = None
    if "ZZDS" in fams and fams["ZZDS"][0] is not None:
        price = fams["ZZDS"][0].filter(pl.col("V") == "c").group_by("code", "day").agg(pl.col("x").mean().alias("p"))
    if "ZZMS" in fams and fams["ZZMS"][0] is not None:
        amount = (fams["ZZMS"][0].filter((pl.col("V") == "n") & pl.col("x").is_finite()).sort("minute")
                  .group_by("code", "day").agg(pl.col("x").last().alias("a")))
    rows, ids, by_name = [], [], {}
    for name, (t, kind) in fams.items():
        if t is None:
            continue
        for V in sorted(t["V"].unique().to_list()):
            s = t.filter(pl.col("V") == V)
            v = s["x"].to_numpy().astype(float)
            fin = v[np.isfinite(v)]
            if fin.size == 0:
                continue
            bp, ba, r2 = elasticity(s, price, amount)
            trend, u, mono = profile(s) if kind == "time" else (float("nan"),) * 3
            ms, mc = memory(s, dates)
            r = {"file": name, "V": V, "kind": kind, "n": int(fin.size), "finite": float(np.isfinite(v).mean()), "min": float(fin.min()),
                 "p50": float(np.median(fin)), "max": float(fin.max()), "neg": float(np.mean(fin < 0)), "zero": float(np.mean(fin == 0)),
                 "int": quant_share(fin, 1.0), "q100": quant_share(fin, 100.0), "q001": quant_share(fin, 0.01), "step": step_of(fin),
                 "cv": float(fin.std() / abs(fin.mean())) if fin.mean() else float("nan"), "bp": bp, "ba": ba, "r2": r2,
                 "trend": trend, "ushape": u, "mono": mono, "mem_slope": ms, "mem_corr": mc}
            r["guess"] = guess(r)
            rows.append(r)
            by_name.setdefault(V, []).append((name, s.select("code", "minute", "day", "x")))
        ids += identities(t, name)
    # the same column name in two files: equal or proportional?
    for V, lst in by_name.items():
        for (n1, s1), (n2, s2) in itertools.combinations(lst, 2):
            j = s1.join(s2, on=["code", "minute", "day"], suffix="_2").filter(pl.col("x").is_finite() & pl.col("x_2").is_finite())
            if j.height < 100:
                continue
            x, y = j["x"].to_numpy(), j["x_2"].to_numpy()
            eq = float(np.mean(np.abs(x - y) <= REL * np.abs(x) + 1e-9))
            nz = y != 0
            k = float(np.median(x[nz] / y[nz])) if nz.sum() >= 100 else float("nan")
            pro = float(np.mean(np.abs(x - k * y) <= REL * np.abs(x) + 1e-9)) if k == k else 0.0
            if eq > 0.5 or pro > 0.5:
                ids.append({"file": f"{n1}|{n2}", "identity": f"{n1}.{V} == {n2}.{V}" if eq > 0.5 else f"{n1}.{V} == {k:.6g} * {n2}.{V}", "share": max(eq, pro)})
    a.out.mkdir(parents=True, exist_ok=True)
    fp = pl.DataFrame(rows)
    fp.write_csv(a.out / "fingerprint.csv")
    with pl.Config(tbl_rows=400, tbl_cols=-1, tbl_width_chars=250, float_precision=3, fmt_str_lengths=60):
        print(f"\n== fingerprints over {len(dates)} day(s): bp / ba = elasticity of the value on price level / daily amount; "
              f"mem = slope of pm_d on pm_prev; step = value grid")
        print(fp.select("file", "V", "n", "finite", "min", "p50", "max", "neg", "int", "q100", "step", "bp", "ba", "r2", "trend", "ushape", "mono",
                        "mem_slope", "mem_corr", "guess"))
        if ids:
            it = pl.DataFrame(ids).sort("file", "share", descending=[False, True])
            it.write_csv(a.out / "fingerprint_identities.csv")
            print("\n== identities between columns (share of cells where the relation holds to 1e-6)")
            print(it)
        else:
            print("\n== identities: none found")
    print(f"table: {a.out / 'fingerprint.csv'}")


if __name__ == "__main__":
    main()
