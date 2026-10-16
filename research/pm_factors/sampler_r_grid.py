r"""Stage 3 search: which (window, quantile method, rounding, session) reproduces PM's ZZCR samplerR.

What is settled (docs/PM_REPRO.md, stage 3): the quantity is per-order filled totals (median
ratio ~1, rank correlation ~0.95), not per-trade volume. What is not: the values. PM's are
continuous and half are half-integers, ours sit on round lots, and ZZAC's five-minute series
does not depend on the time of day, so the window is longer than one day. This grid scores every
combination against the PM file:

* window  -- the last N trading days (ending at ``--date``) pooled, N in ``--windows``;
* method  -- numpy's quantile methods, plus ``log_linear`` (linear interpolation on log size);
* round   -- none, or to the nearest 0.5 (PM's values are multiples of 0.5);
* session -- every trade, or continuous session only.

Score per combination over the 16 columns: ``exact`` (|PM - ours| <= 0.25 after rounding, i.e.
equal on PM's 0.5 grid), ``w2`` / ``w10`` (within 2 % / 10 %), median ratio.

    python research/pm_factors/sampler_r_grid.py --date 20260429 \
        --pm /data/prod/hstats/sod/HS300/ZZCR/20260430/samplerR.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampler_r as R  # catalog, universe, codes

QS = (0.5, 0.9, 0.95, 0.98)
METHODS = ("linear", "lower", "higher", "nearest", "midpoint", "hazen", "weibull", "median_unbiased",
           "normal_unbiased", "inverted_cdf", "averaged_inverted_cdf", "closest_observation",
           "interpolated_inverted_cdf", "log_linear")


def order_totals(con: duckdb.DuckDBPyConnection, dates: list[str], cont: bool) -> pl.DataFrame:
    """Per (date, stock, role, order) filled total for the four roles, over ``dates``."""
    session = R._CONTINUOUS if cont else "TRUE"
    dl = ", ".join(str(int(d)) for d in dates)
    return con.sql(f"""
      WITH t AS (
        SELECT r.date, r.symbol_id, r.dir, r.volume, r.buy_seq_id, r.sell_seq_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date IN ({dl}) AND r.trade_type = 1 AND r.dir IN ({R.BUY}, {R.SELL}) AND {session})
      SELECT date, symbol_id, CASE dir WHEN {R.BUY} THEN 'ab' ELSE 'as' END AS role,
             CASE dir WHEN {R.BUY} THEN buy_seq_id ELSE sell_seq_id END AS oid, sum(volume) AS v
      FROM t GROUP BY ALL
      UNION ALL
      SELECT date, symbol_id, CASE dir WHEN {R.BUY} THEN 'ps' ELSE 'pb' END,
             CASE dir WHEN {R.BUY} THEN sell_seq_id ELSE buy_seq_id END, sum(volume)
      FROM t GROUP BY ALL""").pl()


def quantiles(v: np.ndarray, method: str) -> np.ndarray:
    """Return the four PM quantiles of one stock's sizes under ``method``."""
    if method == "log_linear":
        return np.exp(np.quantile(np.log(v), QS, method="linear"))
    return np.quantile(v, QS, method=method)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="last trade day of the window (PM sod/<D+1> = day D)")
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--windows", default="1,2,3,5,10")
    ap.add_argument("--index", default="000300")
    ap.add_argument("--memory", default="200GB")
    a = ap.parse_args()
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    days = [str(d) for d in con.sql(
        f"SELECT DISTINCT date FROM raw_transaction WHERE date <= {int(a.date)} ORDER BY date DESC").pl()["date"]]
    print(f"decoded trade days up to {a.date}: {days[:12]}")
    uni = R.ced_universe(a.index, a.date)
    con.register("uni", uni if uni is not None else R.universe(a.index))
    pm = pl.read_parquet(a.pm)
    roles = ("as", "ps", "ab", "pb")
    cols = [f"{r}.q{int(q * 100)}" for r in roles for q in QS]
    pm_long = pm.unpivot(index="sid", on=cols, variable_name="column", value_name="pm")
    windows = [n for n in (int(x) for x in a.windows.split(",")) if n <= len(days)]
    rows, keep = [], {}
    for cont in (False, True):
        tot = order_totals(con, days[:max(windows)], cont)
        for n in windows:
            win = set(days[:n])
            sub = tot.filter(pl.col("date").cast(pl.String).is_in(win))
            groups = {k: g["v"].to_numpy().astype(np.float64)
                      for k, g in sub.group_by("symbol_id", "role")}
            for method in METHODS:
                recs = []
                for (sym, role), v in groups.items():
                    qv = quantiles(v, method)
                    recs += [(float(sym), f"{role}.q{int(q * 100)}", float(x)) for q, x in zip(QS, qv, strict=True)]
                ours = pl.DataFrame(recs, schema={"sid": pl.Float64, "column": pl.String, "ours": pl.Float64}, orient="row")
                j = pm_long.join(ours, on=["sid", "column"]).filter((pl.col("pm") > 0) & (pl.col("ours") > 0))
                for rnd in ("none", "half"):
                    o = pl.col("ours") if rnd == "none" else (pl.col("ours") * 2).round(0) / 2
                    rows.append(j.select(
                        pl.lit("cont" if cont else "all").alias("session"), pl.lit(n).alias("days"),
                        pl.lit(method).alias("method"), pl.lit(rnd).alias("round"), pl.len().alias("cells"),
                        ((pl.col("pm") - o).abs() <= 0.25).mean().alias("exact"),
                        ((pl.col("pm") / o - 1).abs() <= 0.02).mean().alias("w2"),
                        ((pl.col("pm") / o - 1).abs() <= 0.10).mean().alias("w10"),
                        (pl.col("pm") / o).median().alias("ratio")))
                    keep[(cont, n, method, rnd)] = j.with_columns(o.alias("ours"))
    res = pl.concat(rows).sort("exact", "w2", "w10", descending=True)
    out = Path("data/reports/stage3")
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / f"sampler_r_grid_{a.date}.csv")
    with pl.Config(tbl_rows=30, tbl_width_chars=200, float_precision=4):
        print("== best 25 combinations")
        print(res.head(25))
        print("== best per window (exact, then w2)")
        print(res.group_by("days").head(1).sort("days"))
        b = res.row(0, named=True)
        j = keep[(b["session"] == "cont", b["days"], b["method"], b["round"])]
        print(f"== per column at the best ({b['session']}, {b['days']} days, {b['method']}, round {b['round']})")
        print(j.group_by("column").agg(((pl.col("pm") - pl.col("ours")).abs() <= 0.25).mean().alias("exact"),
                                       ((pl.col("pm") / pl.col("ours") - 1).abs() <= 0.02).mean().alias("w2"),
                                       (pl.col("pm") / pl.col("ours")).median().alias("ratio")).sort("column"))
        print("== 000001 at the best")
        print(j.filter(pl.col("sid") == 1.0).select("column", "pm", "ours").sort("column"))
    print(f"full grid: {out / f'sampler_r_grid_{a.date}.csv'}")


if __name__ == "__main__":
    main()
