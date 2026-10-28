"""Compare one day converted from the XTP TDF dump (``decode_tdf_day.py``) with the same day decoded from feitu.

Per exchange and kind: record counts and volume of both sources, how many symbols each has, the per-symbol trade count
agreement (an XTP day with losses shows symbols with fewer trades than feitu), and, on trades matched by
``(symbol_id, seq_id)``, the receiver clock offset ``xtp spider_ts - feitu spider_ts`` (median / p90). The same offset
is reported on snapshots matched by ``(symbol_id, time)``. Prices are also compared to the cent, since feitu's raw
prices are the vendor's floats (``build_tick_panel._tick_price`` rounds them) while TDF's are decimal strings; feitu rows
missing from XTP are broken down by session phase.

    python research/compare_xtp_feitu.py --date 20261009 --xtp /work/crucible_data/xtp_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl

FEITU = Path("/work/crucible_data/feitu_raw")
#: session phase of an epoch-ns UTC column, by CST second of day.
PHASE = """CASE WHEN ({t} // 1000000000 + 28800) % 86400 < 33900 THEN '1 <0925'
                 WHEN ({t} // 1000000000 + 28800) % 86400 < 34200 THEN '2 0925-0930'
                 WHEN ({t} // 1000000000 + 28800) % 86400 < 41400 THEN '3 am'
                 WHEN ({t} // 1000000000 + 28800) % 86400 < 46800 THEN '4 lunch'
                 WHEN ({t} // 1000000000 + 28800) % 86400 < 53820 THEN '5 pm'
                 WHEN ({t} // 1000000000 + 28800) % 86400 < 54000 THEN '6 close auction'
                 ELSE '7 >1500' END"""


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
    with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
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
                           avg(CASE WHEN a.price = b.price AND a.volume = b.volume THEN 1 ELSE 0 END) same_px_vol,
                           avg(CASE WHEN round(a.price * 100) = round(b.price * 100) THEN 1 ELSE 0 END) same_px_cent,
                           avg(CASE WHEN a.volume = b.volume THEN 1 ELSE 0 END) same_vol,
                           max(abs(a.price - b.price)) max_px_diff
                    FROM {x} a JOIN {f} b USING (symbol_id, seq_id)
                    WHERE a.trade_type = 1 AND b.trade_type = 1 GROUP BY 1 ORDER BY 1""").pl())
            if kind == "order":
                print("-- feitu orders with no xtp (symbol_id, seq_id): per exchange, update_type and session phase")
                print(con.sql(f"""
                    SELECT b.market_id, b.update_type, {PHASE.format(t="b.time")} phase, count(*) n, count(DISTINCT b.symbol_id) symbols
                    FROM {f} b ANTI JOIN {x} a USING (symbol_id, seq_id)
                    WHERE b.symbol_id IN (SELECT DISTINCT symbol_id FROM {x}) GROUP BY ALL ORDER BY n DESC LIMIT 15""").pl())
            if kind == "quotation":
                print("-- snapshots matched by (symbol_id, time): clock offset and field agreement")
                print(con.sql(f"""
                    SELECT a.market_id, count(*) n_matched,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.5) dt_ms_p50,
                           quantile_cont((a.spider_ts - b.spider_ts) / 1e6, 0.9) dt_ms_p90,
                           avg(CASE WHEN a.bid_px[1] = b.bid_px[1] AND a.ask_px[1] = b.ask_px[1] THEN 1 ELSE 0 END) same_top,
                           avg(CASE WHEN a.total_volume = b.total_volume THEN 1 ELSE 0 END) same_cum_vol,
                           avg(CASE WHEN a.bid_vol[1] = b.bid_vol[1] AND a.ask_vol[1] = b.ask_vol[1] THEN 1 ELSE 0 END) same_top_vol,
                           avg(CASE WHEN round(a.bid_px[1] * 100) = round(b.bid_px[1] * 100)
                                     AND round(a.ask_px[1] * 100) = round(b.ask_px[1] * 100) THEN 1 ELSE 0 END) same_top_cent,
                           avg(CASE WHEN round(a.price * 100) = round(b.price * 100) THEN 1 ELSE 0 END) same_last_cent,
                           avg(CASE WHEN a.total_no = b.total_no THEN 1 ELSE 0 END) same_num_trades,
                           median(b.bid_vol[1] * 1.0 / a.bid_vol[1]) FILTER (WHERE a.bid_vol[1] > 0) bv1_ratio_f_x,
                           avg(CASE WHEN a.bid_vol[1] = b.bid_vol[2] THEN 1 ELSE 0 END) FILTER (WHERE a.bid_vol[1] > 0) x_bv1_eq_f_bv2
                    FROM {x} a JOIN {f} b USING (symbol_id, time) GROUP BY 1 ORDER BY 1""").pl())
                print("-- five matched snapshots whose level-1 bid volume differs (a_ = xtp, b_ = feitu)")
                print(con.sql(f"""
                    SELECT a.market_id, a.symbol_id, {PHASE.format(t="a.time")} phase, a.bid_px[1] a_bid1, b.bid_px[1] b_bid1,
                           a.bid_vol[1] a_bv1, b.bid_vol[1] b_bv1, a.bid_vol[2] a_bv2, b.bid_vol[2] b_bv2, a.ask_vol[1] a_av1, b.ask_vol[1] b_av1
                    FROM {x} a JOIN {f} b USING (symbol_id, time)
                    WHERE a.bid_vol[1] > 0 AND a.bid_vol[1] <> b.bid_vol[1] AND a.symbol_id IN (600000, 1, 600519, 300750, 2594)
                    LIMIT 5""").pl())
                print("-- feitu snapshots with no xtp (symbol_id, time): session phase, and whether nothing traded since the previous one")
                print(con.sql(f"""
                    WITH fb AS (SELECT symbol_id, market_id, time, total_volume,
                                       lag(total_volume) OVER (PARTITION BY symbol_id ORDER BY time) prev_vol,
                                       (time - lag(time) OVER (PARTITION BY symbol_id ORDER BY time)) / 1e9 gap_s
                                FROM {f} WHERE symbol_id IN (SELECT DISTINCT symbol_id FROM {x}))
                    SELECT fb.market_id, {PHASE.format(t="fb.time")} phase, count(*) n, count(DISTINCT fb.symbol_id) symbols,
                           avg(CASE WHEN total_volume = prev_vol THEN 1 ELSE 0 END) no_trade_since_prev, median(gap_s) gap_s_med
                    FROM fb ANTI JOIN {x} a USING (symbol_id, time) GROUP BY ALL ORDER BY 1, n DESC""").pl())


if __name__ == "__main__":
    main()
