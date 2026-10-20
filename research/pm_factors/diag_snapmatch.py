r"""Stage 4 test: is a PM samplerS value the book value of one (random) snapshot in its bar.

``diag_subsets`` found ZZAL's ``an`` / ``bn`` track the CNY value of the resting ask / bid book
(ratio ~1, rank 0.93-0.95) but match neither the bar's last snapshot, its mean, nor a running
mean. If PM samples snapshots, every PM value should equal the book value of *some* snapshot near
its bar. This searches, per PM cell, all snapshots of that stock in a window around the bar for
an exact match (relative 1e-6, or within 0.5 CNY) of each candidate expression, and reports:

* ``matched``  share of PM cells with at least one exactly equal snapshot;
* where the matched snapshot sits relative to the PM label (seconds; PM labels look like bar
  starts, so a bar is [label, label + 300 s)) -- spread across the bar = sampling, always at one
  end = a deterministic rule we have mislabelled.

    python research/pm_factors/diag_snapmatch.py --date 20260429 --family ZZAL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampler_r as R  # catalog, universe

NS = 1_000_000_000
LOCAL = 8 * 3600
EXPR = {  # candidate snapshot expressions per PM column
    "an": {"atot*wavg": "total_ask_volume * weighted_avg_ask_price", "atot": "total_ask_volume::DOUBLE",
           "ask10": "list_sum(list_transform(list_zip(ask_px, ask_vol), z -> z[1] * z[2]))"},
    "bn": {"btot*wavg": "total_bid_volume * weighted_avg_bid_price", "btot": "total_bid_volume::DOUBLE",
           "bid10": "list_sum(list_transform(list_zip(bid_px, bid_vol), z -> z[1] * z[2]))"},
}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--family", default="ZZAL")
    ap.add_argument("--pm-dir", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--before", type=int, default=600, help="seconds before the PM label to search")
    ap.add_argument("--after", type=int, default=900, help="seconds after the PM label to search")
    ap.add_argument("--index", default="000300")
    ap.add_argument("--memory", default="200GB")
    a = ap.parse_args()
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    uni = R.ced_universe(a.index, a.date)
    con.register("uni", uni if uni is not None else R.universe(a.index))
    pm = pl.read_parquet(a.pm_dir / f"{a.family}_{a.date}.parquet")
    hm = pm["I"].str.extract_groups(r"^(\d{1,2}):(\d{2})")
    pm = pm.with_columns(pl.col("S").cast(pl.Int64).alias("sid"),
                         ((hm.struct.field("1").cast(pl.Int64) * 60 + hm.struct.field("2").cast(pl.Int64)) * 60).alias("lab_s"))
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=4):
        for v, exprs in EXPR.items():
            cells = pm.filter((pl.col("V") == v) & pl.col("x").is_finite()).select("sid", "lab_s", pl.col("x").alias("pm"))
            if cells.is_empty():
                continue
            con.register("cells", cells)
            for name, e in exprs.items():
                m = con.sql(f"""
                  WITH s AS (
                    SELECT q.symbol_id AS sid, ((q.time // {NS}) + {LOCAL}) % 86400 AS t_s,
                           (q.time % {NS}) / 1e9 AS frac, {e} AS val
                    FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {int(a.date)})
                  SELECT c.sid, c.lab_s, c.pm, min(s.t_s + s.frac - c.lab_s) AS first_dt,
                         max(s.t_s + s.frac - c.lab_s) AS last_dt, count(*) AS hits
                  FROM cells c JOIN s ON s.sid = c.sid
                   AND s.t_s BETWEEN c.lab_s - {a.before} AND c.lab_s + {a.after}
                   AND abs(s.val - c.pm) <= greatest(0.5, 1e-6 * abs(c.pm))
                  GROUP BY ALL""").pl()
                share = m.height / cells.height
                print(f"== {a.family}.{v} = {name}: {m.height} of {cells.height} PM cells have an exactly equal snapshot "
                      f"({share:.4f}) within [-{a.before}s, +{a.after}s] of the label")
                if m.height:
                    print("   seconds from PM label to the (first) matching snapshot:")
                    print(m.select(pl.col("first_dt").quantile(q).alias(f"p{int(q * 100)}") for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)))
                    print(m.with_columns((pl.col("first_dt") // 60).alias("minute_after_label"))
                          .group_by("minute_after_label").len().sort("minute_after_label").head(25))
                    print(m.sort("sid", "lab_s").head(6))
            con.unregister("cells")


if __name__ == "__main__":
    main()
