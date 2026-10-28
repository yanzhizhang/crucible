"""Compare one day converted from the XTP TDF dump (``decode_tdf_day.py``) with the same day decoded from feitu.

Per exchange and kind: record counts and volume of both sources, how many symbols each has, the per-symbol trade count
agreement (an XTP day with losses shows symbols with fewer trades than feitu), and, on trades matched by
``(symbol_id, seq_id)``, the receiver clock offset ``xtp spider_ts - feitu spider_ts`` (median / p90). The same offset
is reported on snapshots matched by ``(symbol_id, time)``.

    python research/compare_xtp_feitu.py --date 20261009 --xtp /work/crucible_data/xtp_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl

FEITU = Path("/work/crucible_data/feitu_raw")


def _src(root: Path, kind: str, date: str) -> str:
    return f"read_parquet('{root}/kind={kind}/date={date}/part-*.parquet')"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--xtp", type=Path, default=Path("/work/crucible_data/xtp_raw"))
    ap.add_argument("--feitu", type=Path, default=FEITU)
    args = ap.parse_args()
    con = duckdb.connect()
    d = args.date
    with pl.Config(tbl_rows=40, tbl_width_chars=170, float_precision=4):
        for kind in ("order", "transaction", "quotation"):
            x, f = _src(args.xtp, kind, d), _src(args.feitu, kind, d)
            if not list((args.xtp / f"kind={kind}" / f"date={d}").glob("part-*.parquet")) or \
               not list((args.feitu / f"kind={kind}" / f"date={d}").glob("part-*.parquet")):
                print(f"== {kind}: one side has no parts for {d}, skipped")
                continue
            vol = "sum(volume)" if kind != "quotation" else "max(total_volume)"
            filt = "WHERE trade_type = 1" if kind == "transaction" else ""
            print(f"\n== {kind} {d}: totals per exchange (feitu symbols restricted to the XTP universe)")
            print(con.sql(f"""
                WITH u AS (SELECT DISTINCT symbol_id FROM {x}),
                xs AS (SELECT market_id, 'xtp' src, count(*) records, {vol} volume, count(DISTINCT symbol_id) symbols FROM {x} {filt} GROUP BY 1),
                fs AS (SELECT market_id, 'feitu' src, count(*) records, {vol} volume, count(DISTINCT symbol_id) symbols
                       FROM {f} {filt} {"AND" if filt else "WHERE"} symbol_id IN (SELECT symbol_id FROM u) GROUP BY 1)
                SELECT * FROM xs UNION ALL SELECT * FROM fs ORDER BY market_id, src""").pl())
            if kind == "transaction":
                print("-- per-symbol trade count, xtp / feitu: 1 = complete; symbols below 0.999 listed")
                r = con.sql(f"""
                    WITH a AS (SELECT market_id, symbol_id, count(*) n FROM {x} WHERE trade_type = 1 GROUP BY 1, 2),
                    b AS (SELECT symbol_id, count(*) n FROM {f} WHERE trade_type = 1 GROUP BY 1)
                    SELECT a.market_id, a.symbol_id, a.n xtp_n, b.n feitu_n, a.n * 1.0 / b.n ratio
                    FROM a JOIN b USING (symbol_id)""").pl()
                print(r.group_by("market_id").agg(pl.len().alias("symbols"), pl.col("ratio").median().alias("ratio_med"),
                                                  pl.col("ratio").min().alias("ratio_min"), (pl.col("ratio") < 0.999).sum().alias("n_short"),
                                                  (pl.col("ratio") > 1.001).sum().alias("n_long")).sort("market_id"))
                print(r.filter(pl.col("ratio") < 0.999).sort("ratio").head(15))
                print("-- receiver clock: xtp spider_ts - feitu spider_ts on trades matched by (symbol_id, seq_id), ms")
                print(con.sql(f"""
                    SELECT a.market_id, count(*) n_matched,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.5) dt_ms_p50,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.9) dt_ms_p90,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.1) dt_ms_p10,
                           sum(CASE WHEN a.price = b.price AND a.volume = b.volume THEN 1 ELSE 0 END) * 1.0 / count(*) same_px_vol
                    FROM {x} a JOIN {f} b USING (symbol_id, seq_id)
                    WHERE a.trade_type = 1 AND b.trade_type = 1 GROUP BY 1 ORDER BY 1""").pl())
            if kind == "quotation":
                print("-- snapshots matched by (symbol_id, time): clock offset and field agreement")
                print(con.sql(f"""
                    SELECT a.market_id, count(*) n_matched,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.5) dt_ms_p50,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.9) dt_ms_p90,
                           avg(CASE WHEN a.bid_px[1] = b.bid_px[1] AND a.ask_px[1] = b.ask_px[1] THEN 1 ELSE 0 END) same_top,
                           avg(CASE WHEN a.total_volume = b.total_volume THEN 1 ELSE 0 END) same_cum_vol,
                           avg(CASE WHEN a.bid_vol[1] = b.bid_vol[1] AND a.ask_vol[1] = b.ask_vol[1] THEN 1 ELSE 0 END) same_top_vol
                    FROM {x} a JOIN {f} b USING (symbol_id, time) GROUP BY 1 ORDER BY 1""").pl())


if __name__ == "__main__":
    main()
