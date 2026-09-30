"""Build ``catalog.duckdb``: one database exposing every crucible dataset as a SQL view.

Views read the Parquet files in place (nothing is copied), so the catalog is a few KB and
always current. Summary tables are materialised once here -- per-minute activity and latency
for every decoded day -- so the overview notebook renders instantly instead of re-scanning
~2e9 raw records on every page load.

Views
-----
raw_order / raw_transaction / raw_quotation / raw_index   decoded feitu v3 ticks (+ ``date``)
raw_manifest            one row per source dump file (minute, rows, bytes, time range)
bars_1min               right-labelled 1-minute stock bars (+ ``date``)
quality_checks          every quality check result (+ ``date``, ``source``)
bench_history           every performance-monitored run
pub_tdx_daily / pub_index_weights / pub_trading_calendar / pub_sw_industry / pub_st_list
                        public reference data fetched through the a-stock-data skill

Summary tables
--------------
summary_minute          date, kind, exchange, minute (local HH:MM), n, latency p50/p99/max (ms)
summary_inventory       dataset, date, files, rows, bytes on disk

Usage (WSL)::

    python research/catalog.py                # views + summaries for days not yet summarised
    python research/catalog.py --rebuild      # recompute all summaries
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resmon import ResourceMonitor, append_history

REPO = Path(__file__).resolve().parents[1]
DATA = Path("/work/crucible_data")
PUBLIC = REPO / "data" / "public_cache"
CATALOG = DATA / "catalog.duckdb"

VIEWS = {
    **{
        f"raw_{k}": f"read_parquet('{DATA}/feitu_raw/kind={k}/date=*/part-*.parquet', hive_partitioning=true)"
        for k in ("order", "transaction", "quotation", "index")
    },
    "raw_manifest": (
        f"read_parquet('{DATA}/feitu_raw/kind=*/date=*/_manifest.parquet', hive_partitioning=true, "
        "union_by_name=true)"
    ),
    "bars_1min": f"read_parquet('{DATA}/store/bars/1min/date=*/part.parquet', hive_partitioning=true)",
    "quality_checks": (
        f"read_parquet('{DATA}/quality/date=*/source=*/checks.parquet', hive_partitioning=true)"
    ),
    "bench_history": f"read_parquet('{DATA}/bench/history.parquet', union_by_name=true)",
    "pub_tdx_daily": f"read_parquet('{PUBLIC}/tdx_daily/date=*/part.parquet', hive_partitioning=true)",
    "pub_index_weights": (
        f"read_parquet('{PUBLIC}/index_weights/index=*/asof=*.parquet', hive_partitioning=true)"
    ),
    "pub_trading_calendar": f"read_parquet('{PUBLIC}/trading_calendar/*.parquet')",
    "pub_sw_industry": f"read_parquet('{PUBLIC}/sw_industry/*.parquet')",
    "pub_st_list": f"read_parquet('{PUBLIC}/st_list/*.parquet', hive_partitioning=true)",
    "pm_sampler_r": (
        f"read_parquet('{DATA}/store/pm_factors/sampler_r/variant=*/date=*/samplerR.parquet', "
        "hive_partitioning=true)"
    ),
    "bt_summary": f"read_parquet('{DATA}/store/bt/date=*/summary.parquet', union_by_name=true)",
    "bt_sweep": f"read_parquet('{DATA}/store/bt/date=*/sweep.parquet', union_by_name=true)",
    "bt_fills": f"read_parquet('{DATA}/store/bt/date=*/fills.parquet', union_by_name=true)",
}

# market events only: vendor status records are not latency
_EVENT_FILTER = {
    "order": "update_type IN (1, 2)",
    "transaction": "trade_type IN (1, 2)",
    "quotation": "TRUE",
    "index": "TRUE",
}


def _local_minute(col: str) -> str:
    return f"strftime(make_timestamp(({col} // 1000) + 28800000000::BIGINT), '%H:%M')"


def build_views(con: duckdb.DuckDBPyConnection) -> list[str]:
    """(Re)create every view whose files exist; return the names created."""
    made = []
    for name, src in VIEWS.items():
        try:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM {src}")
            con.execute(f"SELECT * FROM {name} LIMIT 0")
            made.append(name)
        except duckdb.Error as exc:
            con.execute(f"DROP VIEW IF EXISTS {name}")
            print(f"  skip view {name}: {str(exc).splitlines()[0]}")
    return made


def build_summaries(con: duckdb.DuckDBPyConnection, *, rebuild: bool) -> None:
    """Per-minute activity/latency per (date, kind, exchange); only for new days unless rebuild."""
    con.execute(
        """CREATE TABLE IF NOT EXISTS summary_minute (
             date VARCHAR, kind VARCHAR, exchange VARCHAR, minute VARCHAR, n BIGINT,
             lat_p50_ms DOUBLE, lat_p99_ms DOUBLE, lat_max_ms DOUBLE)"""
    )
    if rebuild:
        con.execute("DELETE FROM summary_minute")
    done = {
        (d, k) for d, k in con.execute("SELECT DISTINCT date, kind FROM summary_minute").fetchall()
    }
    for kind, cond in _EVENT_FILTER.items():
        days = [
            str(r[0])
            for r in con.execute(f"SELECT DISTINCT date FROM raw_{kind} ORDER BY 1").fetchall()
        ]
        for day in days:
            if (day, kind) in done:
                continue
            t0 = time.perf_counter()
            con.execute(
                f"""INSERT INTO summary_minute
                SELECT '{day}', '{kind}',
                       CASE market_id WHEN 3553 THEN 'XSHG' WHEN 3554 THEN 'XSHE' ELSE 'other' END,
                       {_local_minute("time")} AS minute,
                       count(*),
                       approx_quantile((spider_ts - time) / 1e6, 0.5),
                       approx_quantile((spider_ts - time) / 1e6, 0.99),
                       max((spider_ts - time) / 1e6)
                FROM raw_{kind}
                WHERE date = '{day}' AND {cond}
                GROUP BY ALL"""
            )
            print(f"  summary {kind} {day}: {time.perf_counter() - t0:.1f}s")

    con.execute("DROP TABLE IF EXISTS summary_inventory")
    con.execute(
        f"""CREATE TABLE summary_inventory AS
        SELECT 'feitu_raw/' || kind AS dataset, CAST(date AS VARCHAR) AS date,
               count(*) AS files, sum(rows) AS rows, sum(bytes) AS source_bytes
        FROM read_parquet('{DATA}/feitu_raw/kind=*/date=*/_manifest.parquet',
                          hive_partitioning=true, union_by_name=true)
        GROUP BY ALL
        UNION ALL
        SELECT 'store/bars_1min', CAST(date AS VARCHAR), 1, count(*), NULL
        FROM bars_1min GROUP BY ALL
        UNION ALL
        SELECT 'public/tdx_daily', CAST(date AS VARCHAR), 1, count(*), NULL
        FROM pub_tdx_daily GROUP BY ALL
        ORDER BY 1, 2"""
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="recompute all summaries")
    ap.add_argument("--memory", default="5GB", help="DuckDB memory_limit")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "duckdb_tmp").mkdir(exist_ok=True)
    with ResourceMonitor("catalog.build") as mon:
        con = duckdb.connect(str(CATALOG))
        con.execute(f"SET memory_limit='{a.memory}'")
        con.execute(f"SET temp_directory='{DATA / 'duckdb_tmp'}'")
        views = build_views(con)
        print(f"views: {', '.join(views)}")
        build_summaries(con, rebuild=a.rebuild)
        mon.rows = con.execute("SELECT count(*) FROM summary_minute").fetchone()[0]
        con.close()
    assert mon.stats is not None
    print(mon.stats.line())
    append_history(mon.stats, DATA / "bench" / "history.parquet")
    print(f"catalog: {CATALOG}")


if __name__ == "__main__":
    main()
