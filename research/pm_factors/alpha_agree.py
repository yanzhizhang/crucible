r"""How alike are our prediction and PM's, under every reading of "similar"? And do the differences sit where the feeds differ?

``ic_score --dump-pred`` rows (date, sid, mark, ours, pm; mark = trading-clock second) scored under many definitions, to
find which one a remark like "only ~20 % similar" could mean next to our per-mark rank correlation of ~0.6:

* cross-section per mark: rank / linear correlation (mean over marks), top / bottom decile overlap, same decile, same quintile;
* pooled: linear / rank correlation, r^2, 1 - var(pm - ours) / var(pm) (no fit), sign agreement, share within 10 % / 1 % of PM;
* per stock over the day: correlation of the two series (mean, median), the same after removing each mark's mean;
* changes from one mark to the next: per-mark and pooled correlation of the changes, sign agreement of the changes;
* stock means over the day: correlation across stocks; the pooled correlation after removing each stock's mean.

``--full / --sub`` (two catalogs; sub = a feed that lacks some of full's snapshots) flags every (stock, mark) with a snapshot
of ``full`` missing from ``sub`` in the W seconds before the mark and compares the size of the rank residual (PM's rank beyond
what ours explains) on flagged and unflagged rows, per prediction file; with two files also |ours A - ours B|.

    python research/pm_factors/alpha_agree.py --pred s224=data/reports/stage6_s224_1118/pred.parquet dysl2=data/reports/stage6_r99_1118/pred.parquet \
        --full /work/crucible_data/catalog_dysl2.duckdb --sub /work/crucible_data/catalog_s224.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

WINDOWS = (3.0, 30.0, 300.0)
SOD = "((time / 1e9 + 28800) % 86400)"
TC = f"CASE WHEN {SOD} <= 41400 THEN {SOD} - 34200 WHEN {SOD} < 46800 THEN 7200 ELSE {SOD} - 46800 + 7200 END"


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """Linear correlation, NaN when either side is constant or too short."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rank01(x: np.ndarray) -> np.ndarray:
    """Average ranks scaled to [0, 1]."""
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt))
    np.add.at(s, inv, r)
    return (s / cnt)[inv] / max(len(x) - 1, 1)


def agree(df: pl.DataFrame) -> list[tuple[str, float]]:
    """Every agreement number for one prediction file."""
    df = df.filter(pl.col("ours").is_finite() & pl.col("pm").is_finite()).sort("date", "mark", "sid")
    o, p = df["ours"].to_numpy(), df["pm"].to_numpy()
    out: list[tuple[str, float]] = [("rows", float(len(o))), ("stocks", float(df["sid"].n_unique())), ("marks", float(df.select("date", "mark").n_unique()))]
    xs = {k: [] for k in ("rank", "lin", "top", "bot", "dec", "qui")}
    ro, rp = np.empty(len(o)), np.empty(len(o))
    i = 0
    for g in df.partition_by(["date", "mark"], maintain_order=True):
        a, b = g["ours"].to_numpy(), g["pm"].to_numpy()
        ra, rb = rank01(a), rank01(b)
        ro[i:i + len(a)], rp[i:i + len(a)] = ra, rb
        i += len(a)
        if len(a) < 20:
            continue
        xs["rank"].append(corr(ra, rb))
        xs["lin"].append(corr(a, b))
        xs["top"].append(float(((ra >= 0.9) & (rb >= 0.9)).sum() / max((rb >= 0.9).sum(), 1)))
        xs["bot"].append(float(((ra <= 0.1) & (rb <= 0.1)).sum() / max((rb <= 0.1).sum(), 1)))
        xs["dec"].append(float((np.minimum(ra * 10, 9.999).astype(int) == np.minimum(rb * 10, 9.999).astype(int)).mean()))
        xs["qui"].append(float((np.minimum(ra * 5, 4.999).astype(int) == np.minimum(rb * 5, 4.999).astype(int)).mean()))
    m = lambda v: float(np.nanmean(v)) if len(v) else np.nan  # noqa: E731
    out += [("per mark: rank correlation (our 'similarity')", m(xs["rank"])), ("per mark: linear correlation", m(xs["lin"])),
            ("per mark: rank correlation squared", m(xs["rank"]) ** 2),
            ("per mark: PM's top decile also ours", m(xs["top"])), ("per mark: PM's bottom decile also ours", m(xs["bot"])),
            ("per mark: same decile", m(xs["dec"])), ("per mark: same quintile", m(xs["qui"]))]
    r = corr(o, p)
    out += [("pooled: linear correlation", r), ("pooled: r squared", r * r), ("pooled: rank correlation", corr(rank01(o), rank01(p))),
            ("pooled: 1 - var(pm - ours) / var(pm), no fit", float(1 - np.var(p - o) / np.var(p)) if np.var(p) > 0 else np.nan),
            ("pooled: same sign", float((np.sign(o) == np.sign(p)).mean())),
            ("pooled: ours within 10% of pm", float((np.abs(o - p) <= 0.1 * np.abs(p)).mean())),
            ("pooled: ours within 1% of pm", float((np.abs(o - p) <= 0.01 * np.abs(p)).mean())),
            ("pooled: std ours / std pm", float(np.std(o) / np.std(p)) if np.std(p) > 0 else np.nan)]
    d = df.with_columns(pl.Series("ro", ro), pl.Series("rp", rp)).with_columns(
        (pl.col("ours") - pl.col("ours").mean().over("date", "mark")).alias("oc"), (pl.col("pm") - pl.col("pm").mean().over("date", "mark")).alias("pc"))
    ts, tsc, dch = [], [], {"o": [], "p": []}
    for g in d.sort("date", "sid", "mark").partition_by(["date", "sid"], maintain_order=True):
        a, b = g["ours"].to_numpy(), g["pm"].to_numpy()
        ts.append(corr(a, b))
        tsc.append(corr(g["oc"].to_numpy(), g["pc"].to_numpy()))
        dch["o"].append(np.diff(a))
        dch["p"].append(np.diff(b))
    do, dp = np.concatenate(dch["o"]) if dch["o"] else np.array([]), np.concatenate(dch["p"]) if dch["p"] else np.array([])
    out += [("per stock over the day: correlation, mean", float(np.nanmean(ts))), ("per stock over the day: correlation, median", float(np.nanmedian(ts))),
            ("per stock, mark means removed: mean", float(np.nanmean(tsc))), ("per stock, mark means removed: median", float(np.nanmedian(tsc))),
            ("changes mark to mark: pooled correlation", corr(do, dp)),
            ("changes mark to mark: same sign", float((np.sign(do) == np.sign(dp)).mean()) if len(do) else np.nan)]
    sm = d.group_by("date", "sid").agg(pl.col("ours").mean().alias("om"), pl.col("pm").mean().alias("pm_"))
    out.append(("stock means over the day: correlation across stocks", corr(sm["om"].to_numpy(), sm["pm_"].to_numpy())))
    w = d.with_columns((pl.col("ours") - pl.col("ours").mean().over("date", "sid")).alias("ow"), (pl.col("pm") - pl.col("pm").mean().over("date", "sid")).alias("pw"))
    out.append(("pooled, stock means removed: correlation", corr(w["ow"].to_numpy(), w["pw"].to_numpy())))
    return out


def with_resid(df: pl.DataFrame) -> pl.DataFrame:
    """Add the rank residual: PM's rank within the mark minus the OLS fit on our rank."""
    df = df.filter(pl.col("ours").is_finite() & pl.col("pm").is_finite()).with_columns(
        ((pl.col("ours").rank().over("date", "mark") - 1) / (pl.len().over("date", "mark") - 1)).alias("ro"),
        ((pl.col("pm").rank().over("date", "mark") - 1) / (pl.len().over("date", "mark") - 1)).alias("rp"))
    a, b = df["ro"].to_numpy(), df["rp"].to_numpy()
    beta = np.cov(a, b)[0, 1] / np.var(a, ddof=1)
    return df.with_columns((pl.col("rp") - (b.mean() + beta * (pl.col("ro") - a.mean()))).alias("resid"))


def missing_snaps(full: Path, sub: Path, dates: list[int]) -> pl.DataFrame:
    """Snapshots of ``full`` with no (stock, exchange time) twin in ``sub``: date, sid, t (trading clock of the exchange stamp)."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"ATTACH '{full}' AS f (READ_ONLY)")
    con.execute(f"ATTACH '{sub}' AS s (READ_ONLY)")
    ds = ",".join(str(d) for d in dates)
    n = con.sql(f"SELECT date, count(*) FROM f.raw_quotation WHERE date IN ({ds}) GROUP BY 1 ORDER BY 1").fetchall()
    k = con.sql(f"SELECT date, count(*) FROM s.raw_quotation WHERE date IN ({ds}) GROUP BY 1 ORDER BY 1").fetchall()
    print(f"snapshots per day, full: {n}\nsnapshots per day, sub : {k}")
    return con.sql(f"""SELECT a.date::BIGINT AS date, a.symbol_id::BIGINT AS sid, {TC.replace('time', 'a.time')} AS t
        FROM f.raw_quotation a ANTI JOIN s.raw_quotation b ON a.date = b.date AND a.symbol_id = b.symbol_id AND a.time = b.time
        WHERE a.date IN ({ds})""").pl()


def flag(df: pl.DataFrame, miss: pl.DataFrame) -> pl.DataFrame:
    """Per row: the number of missing snapshots in the W seconds up to the mark (one column per W) and before the mark at all."""
    out = df.with_columns(pl.col("date").cast(pl.Int64), pl.col("sid").cast(pl.Int64))
    key = {(int(d), int(s)): np.sort(g["t"].to_numpy()) for (d, s), g in miss.group_by(["date", "sid"])} if not miss.is_empty() else {}
    dd, ss, mm = out["date"].to_numpy(), out["sid"].to_numpy(), out["mark"].to_numpy()
    cols = {f"m{int(w)}": np.zeros(len(mm), dtype=np.int64) for w in WINDOWS}
    cols["mday"] = np.zeros(len(mm), dtype=np.int64)
    for i in range(len(mm)):
        t = key.get((int(dd[i]), int(ss[i])))
        if t is None:
            continue
        hi = np.searchsorted(t, mm[i], side="right")
        cols["mday"][i] = hi
        for w in WINDOWS:
            cols[f"m{int(w)}"][i] = hi - np.searchsorted(t, mm[i] - w, side="right")
    return out.with_columns([pl.Series(k, v) for k, v in cols.items()])


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pred", nargs="+", required=True, help="label=pred.parquet, one or more")
    ap.add_argument("--full", type=Path, help="catalog of the feed with every snapshot")
    ap.add_argument("--sub", type=Path, help="catalog of the feed that lacks some")
    a = ap.parse_args()
    preds = {}
    for item in a.pred:
        lab, _, path = item.partition("=")
        if not Path(path).exists():
            print(f"{lab}: no {path}")
            continue
        preds[lab] = pl.read_parquet(path)
    if not preds:
        return
    tabs = {lab: agree(df) for lab, df in preds.items()}
    names = [n for n, _ in next(iter(tabs.values()))]
    print(f"{'definition':<56}" + "".join(f"{lab:>14}" for lab in tabs))
    for i, n in enumerate(names):
        print(f"{n:<56}" + "".join(f"{tabs[lab][i][1]:>14.4f}" for lab in tabs))
    near = sorted(((abs(v - 0.2), n, lab, v) for lab, t in tabs.items() for n, v in t[3:] if v == v))[:6]
    print("closest to 0.20: " + "; ".join(f"{n} [{lab}] {v:.3f}" for _, n, lab, v in near))
    if not (a.full and a.sub):
        return
    dates = sorted({int(d) for df in preds.values() for d in df["date"].unique().to_list()})
    miss = missing_snaps(a.full, a.sub, dates)
    per = miss.group_by("sid").len()
    print(f"\nmissing from sub: {miss.height} snapshots on {per.height} stocks; per stock median {per['len'].median() if per.height else 0}, max {per['len'].max() if per.height else 0}")
    if miss.height:
        h = miss.with_columns((pl.col("t") // 1800).cast(pl.Int64).alias("hh")).group_by("hh").len().sort("hh")
        print("  by half hour of the trading clock (index: count): " + " ".join(f"{r[0]}:{r[1]}" for r in h.iter_rows()))
    fl = {lab: flag(with_resid(df), miss) for lab, df in preds.items()}
    for lab, d in fl.items():
        print(f"\n[{lab}] rank residual |resid| on rows with / without a missing snapshot before the mark")
        for c in [f"m{int(w)}" for w in WINDOWS] + ["mday"]:
            y, n = d.filter(pl.col(c) > 0), d.filter(pl.col(c) == 0)
            f = lambda x: f"n {x.height:>7}  mean|resid| {x['resid'].abs().mean() or 0:.4f}  rms {((x['resid'] ** 2).mean() or 0) ** 0.5:.4f}"  # noqa: E731
            print(f"  {c:<5} with: {f(y)}   without: {f(n)}")
        st = d.group_by("sid").agg(pl.col("mday").max().alias("miss"), (pl.col("resid") ** 2).mean().sqrt().alias("rms"))
        print(f"  across stocks: correlation of (missing snapshots in the day, residual rms) = {corr(st['miss'].to_numpy().astype(float), st['rms'].to_numpy()):.3f}"
              f"; rms on stocks with none {st.filter(pl.col('miss') == 0)['rms'].mean() or 0:.4f}, with some {st.filter(pl.col('miss') > 0)['rms'].mean() or 0:.4f}")
    if len(fl) >= 2:
        (la, da), (lb, db) = list(fl.items())[:2]
        j = da.join(db.select("date", "sid", "mark", pl.col("ours").alias("ours_b")), on=["date", "sid", "mark"]).with_columns((pl.col("ours") - pl.col("ours_b")).abs().alias("gap"))
        print(f"\n|ours {la} - ours {lb}| on {j.height} shared rows: mean {j['gap'].mean():.6f} (std of ours {j['ours'].std():.6f}); correlation of the two {corr(j['ours'].to_numpy(), j['ours_b'].to_numpy()):.4f}")
        for c in [f"m{int(w)}" for w in WINDOWS] + ["mday"]:
            y, n = j.filter(pl.col(c) > 0), j.filter(pl.col(c) == 0)
            print(f"  {c:<5} with: n {y.height:>7} mean gap {y['gap'].mean() or 0:.6f}   without: n {n.height:>7} mean gap {n['gap'].mean() or 0:.6f}")


if __name__ == "__main__":
    main()
