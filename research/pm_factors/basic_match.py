r"""Stage 4: which basic tick quantity each PM sampler column is, up to a unit -- scored over every column of ``basics``.

For each PM export (``data/pm_export/<FILE>_<date>.parquet``) and column ``V``: joined to the basics table of its
granularity (1-minute / 5-minute / daily) on (stock, date, bar), at bar shifts 0 / +step / -step and against our
day D, D-1 and blends of earlier days (``blends``: means and EMAs -- a sampler state carries history), every basic column is scored as ``rank_candidates`` does: ``exact``
(equal to 1e-6), ``ratio`` (median PM / ours), ``exact_scaled`` (equal after dividing by the ratio), ``spearman``
and ``within`` (per-stock time-series correlation). Summed over the days given; the best rows per (file, V).

    python research/pm_factors/basic_match.py --dates 20260923,20260924 --out data/reports/stage4/basic_match.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import pm_frame, score

BASICS = Path("/work/crucible_data/store/pm_factors/basics")  # basics.OUT, without importing the DuckDB side


def blends(days: list[str], d: str) -> dict[str, list[tuple[str, float]]]:
    """Our days (with weights) a PM file of day d is tried against: the day, the day before, means and EMAs over earlier days.

    ``m3`` / ``m5``: equal-weight mean of the 3 / 5 days before d; ``m3i``: of d and the 2 before; ``e2`` / ``e5``: weights
    0.5 ** (days back / half-life) over the 10 days before d; ``e2i`` includes d (a sampler state that blends every day).
    """
    prev = [x for x in days if x < d]
    out: dict[str, list[tuple[str, float]]] = {}
    if d in days:
        out["d0"] = [(d, 1.0)]
    if prev:
        out["d-1"] = [(prev[-1], 1.0)]
    for n in (3, 5):
        if len(prev) >= n:
            out[f"m{n}"] = [(x, 1.0) for x in prev[-n:]]
    if d in days and len(prev) >= 2:
        out["m3i"] = [(x, 1.0) for x in [*prev[-2:], d]]
    for h in (2, 5):
        if len(prev) >= 3:
            out[f"e{h}"] = [(x, 0.5 ** ((k - 1) / h)) for k, x in enumerate(reversed(prev[-10:]), start=1)]
            if d in days:
                out[f"e{h}i"] = [(d, 1.0), *[(x, 0.5 ** (k / h)) for k, x in enumerate(reversed(prev[-10:]), start=1)]]
    return out


def blend_table(parts: list[tuple[pl.DataFrame, float]]) -> pl.DataFrame:
    """Weighted mean of the basics over days, per (code, minute); NaN / null cells drop out of their own mean."""
    if len(parts) == 1:
        return parts[0][0]
    t = pl.concat([df.with_columns(pl.lit(w).alias("_w")) for df, w in parts], how="diagonal_relaxed")
    cols = [c for c in t.columns if c not in ("code", "minute", "_w")]
    return t.group_by("code", "minute").agg(
        [(pl.col(c) * pl.col("_w")).filter(pl.col(c).is_not_null() & pl.col(c).is_not_nan()).sum()
         / pl.col("_w").filter(pl.col(c).is_not_null() & pl.col(c).is_not_nan()).sum() for c in cols])


def match_file(f: Path, d: str, days: list[str], basics_dir: Path) -> list[pl.DataFrame]:
    """Score one PM export: every (blend, shift, basic), the basics unpivoted once and one group_by per (blend, shift)."""
    cache: dict[tuple[str, str], pl.DataFrame | None] = {}

    def table_of(day: str, table: str) -> pl.DataFrame | None:
        if (day, table) not in cache:
            g = basics_dir / f"date={day}" / f"{table}.parquet"
            cache[(day, table)] = (pl.read_parquet(g).rename({"symbol_id": "code"})
                                   .with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))) if g.exists() else None
        return cache[(day, table)]

    rows: list[pl.DataFrame] = []
    pm, vals, kind = pm_frame(f, None, None, None, None, None)
    pm = pm.with_columns(pl.lit(int(d)).alias("date"))  # the folder's day, whatever the file's D coordinate says
    step = 1
    if kind == "time":
        step = int(pm.select(pl.col("minute").unique().sort().diff().drop_nulls().min()).item() or 1)
        table = "m5" if step >= 5 else "m1"
    elif kind == "daily":
        table = "d1"
    else:
        print(f"{f.name}: I kind {kind} not handled")
        return rows
    name = f.stem.rsplit("_", 1)[0]
    for blend, parts in blends(days, d).items():
        frames = [(t, w) for day, w in parts if (t := table_of(day, table)) is not None]
        if len(frames) < len(parts):
            continue
        ours = blend_table(frames).with_columns(pl.lit(int(d)).alias("date"))
        basics = [c for c in ours.columns if c not in ("code", "minute", "date")]
        long = ours.unpivot(index=["code", "minute", "date"], on=basics, variable_name="basic", value_name="ours")
        for sh in ((0, step, -step) if kind == "time" else (0,)):
            p = pm.with_columns(pl.col("minute") + sh) if kind == "time" else pm
            keys = ["code", "date", "minute"] if kind == "time" else ["code", "date"]
            j = p.join(long if kind != "daily" else long.drop("minute"), on=keys)
            if j.height == 0:
                print(f"  {f.stem}: nothing joined for blend {blend} shift {sh} (pm codes {pm['code'].min()}..{pm['code'].max()}, "
                      f"ours {ours['code'].min()}..{ours['code'].max()}, pm minutes {pm['minute'].min()}..{pm['minute'].max()})")
                continue
            for c in vals:
                r = score(j, c, ["V", "basic"], within=kind == "time")
                rows.append(r.with_columns(pl.lit(name).alias("file"), pl.lit(c).alias("value_col"), pl.lit(sh).alias("shift"),
                                           pl.lit(blend).alias("blend"), pl.lit(d).alias("day")))
    print(f"{d} {name}: {kind}, {len(vals)} value columns, {pm['V'].n_unique()} V, blends {list(blends(days, d))}", flush=True)
    return rows


def _job(args: tuple) -> list[pl.DataFrame]:
    return match_file(*args)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1, help="(day, file) pairs scored in parallel")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4/basic_match.csv"))
    a = ap.parse_args()
    dates = [x for x in a.dates.split(",") if x]
    days = sorted(p.name.split("=")[1] for p in a.basics.glob("date=*"))
    jobs = [(f, d, days, a.basics) for d in dates for f in sorted(a.exports.glob(f"*_{d}.parquet"))]
    rows: list[pl.DataFrame] = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_job, jobs):
                rows += r
    else:
        for job in jobs:
            rows += match_file(*job)
    if not rows:
        print("nothing scored")
        return
    t = pl.concat(rows, how="diagonal_relaxed")
    agg = (t.group_by("file", "value_col", "V", "basic", "shift", "blend").agg(
                pl.len().alias("days"), pl.col("n").sum(), pl.col("exact").mean(), pl.col("exact_scaled").mean(),
                pl.col("ratio").median(), pl.col("spearman").mean(), pl.col("within").mean())
           .with_columns(pl.max_horizontal(pl.col("spearman").abs(), pl.col("within").abs().fill_null(0)).alias("score"))
           .sort("file", "V", "exact_scaled", "score", descending=[False, False, True, True]))
    agg.write_csv(a.out)
    best = agg.group_by("file", "value_col", "V", maintain_order=True).head(a.top)
    with pl.Config(tbl_rows=400, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(f"\n== PM column -> basic quantity, best {a.top} per column over {len(dates)} day(s) (exact_scaled 1.0 = that basic up to a unit)")
        print(best.select("file", "V", "basic", "shift", "blend", "days", "n", "exact_scaled", "exact", "ratio", "spearman", "within"))
    print(f"table: {a.out}")


if __name__ == "__main__":
    main()
