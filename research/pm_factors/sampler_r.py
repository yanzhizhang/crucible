"""Stage 3: reproduce PM's ``samplerR`` -- per-stock order-size thresholds from one day of trades.

PM layout (``hstats/<sod|.>/HS300/<ZZAC|ZZCR>/<date>/samplerR.parquet``): one row per CSI 300
constituent, ``sid`` = the 6-digit code as a number (``000001`` -> 1.0), and 16 columns
``{as, ps, ab, pb} x {q50, q90, q95, q98}``. The ``sod/<D>`` copy is built from day ``D-1``.

Working hypothesis (the evidence is in docs/PM_REPRO.md, stage 3):

* ``as`` / ``ab`` -- **active** sell / buy: the aggressor order of each trade (trade ``dir`` =
  SELL / BUY), with its fills **summed per order** (an aggressive order sweeps several resting
  orders in one go).
* ``ps`` / ``pb`` -- **passive** sell / buy: the resting order a trade hit, summed per order over
  the whole day (it can be hit many times).
* quantiles q50/q90/q95/q98 of those per-order share totals, per stock.

Per-trade volume cannot be it: ``as`` and ``pb`` would then be the same trades and equal, while
PM's sample has them clearly different (000001: as.q50 799.5 vs pb.q50 557). Candidate variants
are all written so the intranet comparison can pick the one that matches:

* ``order_all``   -- per-order totals, every trade with a direction (default hypothesis)
* ``order_cont``  -- per-order totals, continuous session only (auction trades carry a
  vendor-assigned direction but have no real aggressor)
* ``trade_all``   -- per-trade volume (the rejected-by-construction baseline, kept as a control)

Output: ``/work/crucible_data/store/pm_factors/sampler_r/variant=<v>/date=<D>/samplerR.parquet``
in PM's exact column layout (``sid`` float64), where ``D`` is the *trade* day; PM's
``sod/<next trading day>`` should be compared against it.

Runs in DuckDB over the catalog views (out-of-core, memory-capped); trades are restricted to the
universe *before* the per-order grouping, which keeps it to a few million groups.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench.resmon import ResourceMonitor, append_history

REPO = Path(__file__).resolve().parents[2]
DATA = Path("/work/crucible_data")
CATALOG = DATA / "catalog.duckdb"
OUT = DATA / "store" / "pm_factors" / "sampler_r"
PUBLIC = REPO / "data" / "public_cache"
QS = (0.5, 0.9, 0.95, 0.98)
ROLES = ("as", "ps", "ab", "pb")  # PM's column order
BUY, SELL = 1063, 1550
_CONTINUOUS = (
    "strftime(make_timestamp((time // 1000) + 28800000000::BIGINT), '%H:%M:%S') "
    "BETWEEN '09:30:00' AND '14:56:59.999'"
)
VARIANTS = {
    "order_all": ("order", "TRUE"),
    "order_cont": ("order", _CONTINUOUS),
    "trade_all": ("trade", "TRUE"),
}


def universe(index_code: str) -> pl.DataFrame:
    """Latest cached constituents of ``index_code`` as (symbol_id, market_id)."""
    files = sorted((PUBLIC / "index_weights" / f"index={index_code}").glob("asof=*.parquet"))
    if not files:
        raise SystemExit(f"no cached constituents for {index_code}: run research/astock/fetch.py")
    w = pl.read_parquet(files[-1])
    print(f"universe {index_code}: {w.height} names as of {w['date'][0]} ({files[-1].name})")
    return w.select(
        pl.col("code").cast(pl.Int32).alias("symbol_id"),
        pl.when(pl.col("exchange") == "SH")
        .then(3553)
        .otherwise(3554)
        .cast(pl.Int16)
        .alias("market_id"),
    )


def sampler_r(con: duckdb.DuckDBPyConnection, date: str, variant: str) -> pl.DataFrame:
    """PM-layout samplerR for one trade day under one hypothesis variant."""
    unit, session = VARIANTS[variant]
    qlist = "[" + ", ".join(str(q) for q in QS) + "]"
    base = f"""
      WITH t AS (
        SELECT r.symbol_id, r.market_id, r.dir, r.volume, r.buy_seq_id, r.sell_seq_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {int(date)} AND r.trade_type = 1 AND r.dir IN ({BUY}, {SELL}) AND {session}
      )"""
    if unit == "order":
        sizes = f"""{base},
      act AS (SELECT symbol_id, CASE dir WHEN {BUY} THEN 'ab' ELSE 'as' END AS role,
                     CASE dir WHEN {BUY} THEN buy_seq_id ELSE sell_seq_id END AS oid, sum(volume) AS v
              FROM t GROUP BY ALL),
      pas AS (SELECT symbol_id, CASE dir WHEN {BUY} THEN 'ps' ELSE 'pb' END AS role,
                     CASE dir WHEN {BUY} THEN sell_seq_id ELSE buy_seq_id END AS oid, sum(volume) AS v
              FROM t GROUP BY ALL),
      s AS (SELECT symbol_id, role, v FROM act UNION ALL SELECT symbol_id, role, v FROM pas)"""
    else:
        sizes = f"""{base},
      s AS (SELECT symbol_id, CASE dir WHEN {BUY} THEN 'ab' ELSE 'as' END AS role, volume AS v FROM t
            UNION ALL
            SELECT symbol_id, CASE dir WHEN {BUY} THEN 'ps' ELSE 'pb' END AS role, volume AS v FROM t)"""
    long = con.sql(
        f"""{sizes}
        SELECT symbol_id, role, quantile_cont(v, {qlist}) AS q, count(*) AS n
        FROM s GROUP BY ALL"""
    ).pl()
    wide = long.with_columns(
        [pl.col("q").list.get(i).alias(f"q{int(q * 100)}") for i, q in enumerate(QS)]
    ).drop("q")
    cols = [pl.col("symbol_id").cast(pl.Float64).alias("sid")]
    frames = []
    for role in ROLES:
        part = wide.filter(pl.col("role") == role).select(
            "symbol_id",
            *[pl.col(f"q{int(q * 100)}").alias(f"{role}.q{int(q * 100)}") for q in QS],
            pl.col("n").alias(f"{role}.n"),
        )
        frames.append(part)
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="symbol_id", how="full", coalesce=True)
    return out.select(
        *cols,
        *[f"{r}.q{int(q * 100)}" for r in ROLES for q in QS],
        *[f"{r}.n" for r in ROLES],
    ).sort("sid")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="20260615,20260805")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--index", default="000300")
    ap.add_argument("--memory", default="6GB")
    a = ap.parse_args()
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    con.execute(f"SET temp_directory='{DATA / 'duckdb_tmp'}'")
    con.register("uni", universe(a.index))
    for d in filter(None, a.dates.split(",")):
        for v in filter(None, a.variants.split(",")):
            with ResourceMonitor("W3.sampler_r", params={"date": d, "variant": v}) as mon:
                df = sampler_r(con, d, v)
                mon.rows = df.height
            dst = OUT / f"variant={v}" / f"date={d}"
            dst.mkdir(parents=True, exist_ok=True)
            df.write_parquet(dst / "samplerR.parquet")
            assert mon.stats is not None
            append_history(mon.stats, DATA / "bench" / "history.parquet")
            s1 = df.filter(pl.col("sid") == 1.0)
            ex = s1.select("as.q50", "as.q90", "as.q95", "as.q98").row(0) if s1.height else None
            print(
                f"{d} {v:10s}: {df.height} stocks, {mon.stats.wall_s:.1f}s, "
                f"rss {mon.stats.rss_peak_mb:.0f}MB; 000001 as.q* = {ex}"
            )


if __name__ == "__main__":
    main()
