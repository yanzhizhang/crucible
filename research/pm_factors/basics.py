r"""Stage 4: the basic tick quantities per (stock, minute) -- one wide table any PM sampler column can be matched against.

Not family-specific candidates: every plain quantity the streams give, so ``basic_match.py`` can ask of each PM
column "which basic is it, up to a unit?" before anyone guesses a formula. Per 1-minute bar (right-labelled like
``build_bars_from_l2``: bar t covers (t-1min, t], auctions folded into 09:31 / 13:01 / 15:00):

* trades: volume / amount / count, by aggressor side (``b_`` buy, ``s_`` sell); distinct aggressor orders and
  their mean size (``b_osz``), mean trade size (``b_tsz``), the aggressor-order size quantiles q50 / q90 of the
  orders first seen in the bar, distinct passive orders and their mean fill; vwap, high, low, last;
* new orders (``ordb_`` / ``orda_``: bid / ask side): count, volume, amount, mean size; cancels (``cxb_`` / ``cxa_``):
  count, volume (SSE from the order stream, SZSE from the cancel transactions);
* snapshots: last of the bar -- ``bv1 av1 bv5 av5 bv10 av10 btot atot bno ano bp1 ap1 mid spread`` and the
  bar means ``m_bv1 m_av1 m_bv5 m_av5 m_btot m_atot``; ``*_amt`` = volume x price for the totals, ``*_wamt`` = total x the feed's weighted average price (``*_wamt_sp`` the same snapshot picked on the trade clock, ``m_*_wamt`` its bar mean); ``ord*_vol_l`` limit orders only; ``ord*_vol_r`` SZSE net of the order's own continuous fills as aggressor (what rests, the SSE
  feed's convention), ``ord*_vol_n`` orders with no such fill (SSE: no fill as the BS-flag side), ``ord*_vol_x`` net of the order's own aggressive fills on both
  exchanges, ``ord*_vol_c`` / ``_k`` the same without orders placed before 09:30 / 09:25; ``ord*_vol_rc`` the ``_r``
  convention from 09:30, ``_dc`` / ``_ic`` / ``_jc`` that again without SSE repeats of an order id / SSE orders cancelled
  within 1 ms / 1 s; ``*_wamt_pre`` / ``cvol_pre`` / ``camt_pre`` the last snapshot strictly before the bar end;
* running totals since the open: ``cum_vol cum_amt cum_n cum_bvol cum_svol``; ``cum_vol_oa`` / ``cum_amt_oa`` plus the
  opening auction's fills (``oa_vol`` / ``oa_amt``, booked at 09:31); ``cvol_snap`` / ``camt_snap`` the feed's own day totals
  at the last snapshot of the bar; ``snap_vol`` / ``snap_amt`` the same flows from consecutive snapshots (PM's 1min_src way)
  and ``snap_b*`` / ``snap_s*`` split by the interval's price against the previous mid (no BS flag);
* for every bid / ask or buy / sell pair: ``<x>_sum``, ``<x>_mean``, ``<x>_imb``.

The same table aggregated to 5-minute bars (``m5``) and the day (``d1``): flows summed, states taken last, means
averaged. Output: ``store/pm_factors/basics/date=<D>/{m1,m5,d1}.parquet`` (``symbol_id``, ``minute``, columns).

    python research/pm_factors/basics.py --dates 20261009 --thr-file /data/prod/hstats/HS300/ZZCR/20261008/samplerR.parquet

``--only orders,pre,cum,snap`` recomputes just those columns (order / cancel flows, the strictly-before snapshot, day totals,
snapshot flows) and patches them in;
each finished day prints seconds per stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import threads  # noqa: E402
from candidates import (
    AM_CLOSE,
    AM_OPEN,
    BUY,
    CATALOG,
    CLOSE,
    CONT_END,
    DATA,
    NS_DAY,
    NS_LOCAL,
    NS_MIN,
    PM_OPEN,
    SELL,
    SSE,
    SZSE,
    _CEIL,
    _continuous,
    _label,
    ced_universe,
    universe,
)

OUT = DATA / "store" / "pm_factors" / "basics"
OUT_PM = DATA / "store" / "pm_factors" / "basics_pm"
SSE_LAG_NS = 25_000_000  # PM's SSE clock = our receive time - 25 ms (bar_variants.py: SSE exact 0.57 -> 0.97)
FLOW = ("vol", "amt", "n", "ord", "pas", "b_vol", "s_vol", "b_amt", "s_amt", "b_n", "s_n", "b_ord", "s_ord",
        "ordb_n", "orda_n", "ordb_vol", "orda_vol", "ordb_amt", "orda_amt", "cxb_n", "cxa_n", "cxb_vol", "cxa_vol", "ordb_vol_l", "orda_vol_l",
        "ordb_vol_r", "orda_vol_r", "ordb_vol_n", "orda_vol_n", "ordb_vol_x", "orda_vol_x",
        "ordb_vol_c", "orda_vol_c", "ordb_vol_k", "orda_vol_k", "ordb_vol_rc", "orda_vol_rc", "ordb_vol_dc", "orda_vol_dc",
        "ordb_vol_ic", "orda_vol_ic", "ordb_vol_jc", "orda_vol_jc", "oa_vol", "oa_amt", "ca_vol", "ca_amt",
        "snap_vol", "snap_amt", "snap_bvol", "snap_svol", "snap_bamt", "snap_samt")
PAIRS = {"vol": ("b_vol", "s_vol"), "amt": ("b_amt", "s_amt"), "n": ("b_n", "s_n"), "ord": ("b_ord", "s_ord"),
         "osz": ("b_osz", "s_osz"), "tsz": ("b_tsz", "s_tsz"), "q50": ("b_q50", "s_q50"), "q90": ("b_q90", "s_q90"),
         "ordn": ("ordb_n", "orda_n"), "ordvol": ("ordb_vol", "orda_vol"), "ordamt": ("ordb_amt", "orda_amt"),
         "ordmsz": ("ordb_msz", "orda_msz"), "cxn": ("cxb_n", "cxa_n"), "cxvol": ("cxb_vol", "cxa_vol"),
         "v1": ("bv1", "av1"), "v5": ("bv5", "av5"), "v10": ("bv10", "av10"), "tot": ("btot", "atot"), "no": ("bno", "ano"),
         "mv1": ("m_bv1", "m_av1"), "mv5": ("m_bv5", "m_av5"), "mtot": ("m_btot", "m_atot"), "totamt": ("btot_amt", "atot_amt"),
         "totwamt": ("btot_wamt", "atot_wamt"), "totwamtsp": ("btot_wamt_sp", "atot_wamt_sp"), "mtotwamt": ("m_btot_wamt", "m_atot_wamt"),
         "ordvoll": ("ordb_vol_l", "orda_vol_l"), "ordvolr": ("ordb_vol_r", "orda_vol_r"), "ordvoln": ("ordb_vol_n", "orda_vol_n"), "ordvolx": ("ordb_vol_x", "orda_vol_x"),
         "ordvolc": ("ordb_vol_c", "orda_vol_c"), "ordvolk": ("ordb_vol_k", "orda_vol_k"), "totwamtpre": ("btot_wamt_pre", "atot_wamt_pre"),
         "ordvolrc": ("ordb_vol_rc", "orda_vol_rc"), "ordvoldc": ("ordb_vol_dc", "orda_vol_dc"), "ordvolic": ("ordb_vol_ic", "orda_vol_ic"),
         "ordvoljc": ("ordb_vol_jc", "orda_vol_jc"), "snapv": ("snap_bvol", "snap_svol"), "snapa": ("snap_bamt", "snap_samt")}


def _clocks(pm: bool):
    """SQL builders for the tick clock, its local minute and the bar label (PM: SSE on receive time - 25 ms)."""

    def t(a: str) -> str:  # the clock a tick is bucketed by
        return f"(CASE WHEN {a}.market_id = {SSE} THEN {a}.spider_ts - {SSE_LAG_NS} ELSE {a}.time END)" if pm else f"{a}.time"

    def mins(a: str) -> str:  # fractional local minute of day on that clock
        return f"((({t(a)} + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE)"

    def lab(a: str) -> str:  # bar label; on the receive clock an 11:29:59.9 fill lands just past 11:30 and stays in the morning
        return f"(CASE WHEN {mins(a)} > {AM_CLOSE} AND {mins(a)} < {AM_CLOSE + 10} THEN {AM_CLOSE} ELSE {_label(t(a))} END)" if pm else _label(t(a))

    return t, mins, lab


def order_flows(con: duckdb.DuckDBPyConnection, date: str, pm: bool = False) -> pl.DataFrame:
    """New orders and cancels per (stock, minute): the ``ord*`` / ``cx*`` columns."""
    d, b, s = int(date), BUY, SELL
    _, mins, lab = _clocks(pm)
    return con.sql(f"""
        WITH ag AS (  -- continuous fills per aggressor order: SZSE the later (larger) sequence number, SSE the BS flag's side (order_id)
          SELECT r.symbol_id, r.market_id,
                 CASE WHEN r.market_id = {SZSE} THEN greatest(r.buy_seq_id, r.sell_seq_id) WHEN r.dir = {b} THEN r.buy_seq_id ELSE r.sell_seq_id END AS aid,
                 sum(r.volume)::DOUBLE AS fv
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
          WHERE r.date = {d} AND r.trade_type = 1 AND (r.market_id = {SZSE} OR r.dir IN ({b}, {s}))
            AND (((r.time + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE) >= {AM_OPEN}
          GROUP BY ALL),
        cx AS (  -- SSE: first cancel time per order
          SELECT c.symbol_id, c.market_id, c.order_id AS aid, min(c.time) AS ctime
          FROM raw_order c JOIN uni u USING (symbol_id, market_id)
          WHERE c.date = {d} AND c.market_id = {SSE} AND c.update_type = 2
          GROUP BY ALL),
        oo AS (
          SELECT o.*, CASE WHEN o.market_id = {SSE} THEN o.order_id ELSE o.seq_id END AS okey,  -- the key trades quote
                 row_number() OVER (PARTITION BY o.symbol_id, o.market_id, CASE WHEN o.market_id = {SSE} THEN o.order_id ELSE o.seq_id END
                                    ORDER BY o.time, o.seq_id) AS rn
          FROM raw_order o JOIN uni u USING (symbol_id, market_id)
          WHERE o.date = {d} AND o.update_type = 1 AND o.dir IN ({b}, {s})),
        e AS (
          SELECT o.symbol_id, {lab('o')} AS minute, 0 AS k, o.dir, o.volume::DOUBLE AS volume, o.volume * o.price AS amt,
                 (o.market_id = {SSE} OR o.order_type = 2) AS lim,
                 CASE WHEN o.market_id = {SZSE} THEN greatest(o.volume - coalesce(ag.fv, 0), 0) ELSE o.volume END::DOUBLE AS rvol,
                 ag.fv IS NULL AS nomkt,
                 greatest(o.volume - coalesce(ag.fv, 0), 0)::DOUBLE AS xvol, {mins('o')} AS om,
                 o.rn = 1 AS first, coalesce(cx.ctime - o.time <= 1000000, false) AS ioc, coalesce(cx.ctime - o.time <= 1000000000, false) AS fast
          FROM oo o
          LEFT JOIN ag ON ag.symbol_id = o.symbol_id AND ag.market_id = o.market_id AND ag.aid = o.okey  -- equalities only: a one-sided term turns this into a nested-loop join
          LEFT JOIN cx ON cx.symbol_id = o.symbol_id AND cx.market_id = o.market_id AND cx.aid = o.okey
          UNION ALL
          SELECT o.symbol_id, {lab('o')}, 1, o.dir, o.volume::DOUBLE, o.volume * o.price, false, NULL::DOUBLE, false, NULL::DOUBLE, NULL::DOUBLE, false, false, false
          FROM raw_order o JOIN uni u USING (symbol_id, market_id) WHERE o.date = {d} AND o.market_id = {SSE} AND o.update_type = 2
          UNION ALL
          SELECT r.symbol_id, {lab('r')}, 1, CASE WHEN r.buy_seq_id > 0 THEN {b} ELSE {s} END, r.volume::DOUBLE, r.volume * r.price, false, NULL::DOUBLE, false, NULL::DOUBLE, NULL::DOUBLE, false, false, false
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id) WHERE r.date = {d} AND r.market_id = {SZSE} AND r.trade_type = 2)
        SELECT symbol_id, minute,
               count(*) FILTER (k = 0 AND dir = {b}) AS ordb_n, count(*) FILTER (k = 0 AND dir = {s}) AS orda_n,
               sum(volume) FILTER (k = 0 AND dir = {b}) AS ordb_vol, sum(volume) FILTER (k = 0 AND dir = {s}) AS orda_vol,
               sum(amt) FILTER (k = 0 AND dir = {b}) AS ordb_amt, sum(amt) FILTER (k = 0 AND dir = {s}) AS orda_amt,
               count(*) FILTER (k = 1 AND dir = {b}) AS cxb_n, count(*) FILTER (k = 1 AND dir = {s}) AS cxa_n,
               sum(volume) FILTER (k = 1 AND dir = {b}) AS cxb_vol, sum(volume) FILTER (k = 1 AND dir = {s}) AS cxa_vol,
               sum(volume) FILTER (k = 0 AND lim AND dir = {b}) AS ordb_vol_l, sum(volume) FILTER (k = 0 AND lim AND dir = {s}) AS orda_vol_l,
               sum(rvol) FILTER (k = 0 AND dir = {b}) AS ordb_vol_r, sum(rvol) FILTER (k = 0 AND dir = {s}) AS orda_vol_r,
               sum(volume) FILTER (k = 0 AND nomkt AND dir = {b}) AS ordb_vol_n, sum(volume) FILTER (k = 0 AND nomkt AND dir = {s}) AS orda_vol_n,
               sum(xvol) FILTER (k = 0 AND dir = {b}) AS ordb_vol_x, sum(xvol) FILTER (k = 0 AND dir = {s}) AS orda_vol_x,
               sum(xvol) FILTER (k = 0 AND om >= {AM_OPEN} AND dir = {b}) AS ordb_vol_c, sum(xvol) FILTER (k = 0 AND om >= {AM_OPEN} AND dir = {s}) AS orda_vol_c,
               sum(xvol) FILTER (k = 0 AND om >= {AM_OPEN - 5} AND dir = {b}) AS ordb_vol_k, sum(xvol) FILTER (k = 0 AND om >= {AM_OPEN - 5} AND dir = {s}) AS orda_vol_k,
               sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND dir = {b}) AS ordb_vol_rc, sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND dir = {s}) AS orda_vol_rc,
               sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND first AND dir = {b}) AS ordb_vol_dc, sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND first AND dir = {s}) AS orda_vol_dc,
               sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND NOT ioc AND dir = {b}) AS ordb_vol_ic, sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND NOT ioc AND dir = {s}) AS orda_vol_ic,
               sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND NOT fast AND dir = {b}) AS ordb_vol_jc, sum(rvol) FILTER (k = 0 AND om >= {AM_OPEN} AND NOT fast AND dir = {s}) AS orda_vol_jc
        FROM e GROUP BY ALL""").pl()


def snapshots_pre(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """Resting totals x weighted average price of the last snapshot strictly before the bar end (exchange time)."""
    sh = "(q.time + 1)"  # bucket [t - 1 min, t) instead of (t - 1 min, t]
    return con.sql(f"""
        SELECT q.symbol_id, {_label(sh)} AS minute,
               arg_max(q.total_bid_volume::DOUBLE, q.time) AS btot_pre, arg_max(q.total_ask_volume::DOUBLE, q.time) AS atot_pre,
               arg_max(q.total_bid_volume * q.weighted_avg_bid_price, q.time) AS btot_wamt_pre,
               arg_max(q.total_ask_volume * q.weighted_avg_ask_price, q.time) AS atot_wamt_pre,
               arg_max(q.total_volume::DOUBLE, q.time) AS cvol_pre, arg_max(q.total_amount, q.time) AS camt_pre
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {int(date)} AND {_continuous(sh)}
        GROUP BY ALL""").pl()


def _pm_label(col: str) -> str:
    """PM's 1min_src bar label: right-inclusive minute, pre-open into 09:31, 11:30 reads up to 13:00:00 (callers stop at 14:57:00)."""
    c = _CEIL.format(t=col)
    return f"CASE WHEN {c} <= {AM_OPEN} THEN {AM_OPEN + 1} WHEN {c} > {AM_CLOSE} AND {c} <= {PM_OPEN} THEN {AM_CLOSE} ELSE {c} END"


def snap_flows(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """Flows from consecutive snapshots (PM's 1min_src way): ``snap_vol`` / ``snap_amt`` = growth of the day totals, the first
    continuous interval counted from the day total minus the opening auction; ``snap_b*`` / ``snap_s*`` the same split by the interval's average price
    against the previous snapshot's mid (tick rule on a tie or a one-sided book) -- direction without the BS flag."""
    d = int(date)
    c = _CEIL.format(t="q.time")
    mx = f"((({{t}} + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE)"
    return con.sql(f"""
        WITH oa AS (
          SELECT r.symbol_id, sum(r.volume)::DOUBLE AS v, sum(r.volume * r.price) AS a
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
          WHERE r.date = {d} AND r.trade_type = 1 AND {mx.format(t='r.time')} < {AM_OPEN}
          GROUP BY 1),
        s AS (
          SELECT q.symbol_id, q.time, {_pm_label('q.time')} AS minute, {c} AS c, q.price, q.total_volume::DOUBLE AS cv, q.total_amount AS ca,
                 q.total_volume::DOUBLE - lag(q.total_volume::DOUBLE) OVER w AS dv, q.total_amount - lag(q.total_amount) OVER w AS da,
                 lag(q.time) OVER w AS ptime, lag(q.price) OVER w AS pprice,
                 lag(CASE WHEN q.bid_px[1] > 0 AND q.ask_px[1] > 0 THEN (q.bid_px[1] + q.ask_px[1]) / 2 END) OVER w AS pmid
          FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d}
          WINDOW w AS (PARTITION BY q.symbol_id ORDER BY q.time)),
        x AS (
          SELECT s.symbol_id, s.minute,
                 CASE WHEN {mx.format(t='s.ptime')} < {AM_OPEN} THEN greatest(s.cv - coalesce(oa.v, 0), 0) ELSE s.dv END AS dv,
                 CASE WHEN {mx.format(t='s.ptime')} < {AM_OPEN} THEN greatest(s.ca - coalesce(oa.a, 0), 0) ELSE s.da END AS da,
                 s.price, s.pprice, s.pmid
          FROM s LEFT JOIN oa USING (symbol_id)
          WHERE s.ptime IS NOT NULL AND {mx.format(t='s.time')} >= {AM_OPEN} AND s.c <= {CONT_END}),
        y AS (
          SELECT *, CASE WHEN dv <= 0 THEN 0
                         WHEN pmid IS NOT NULL AND da / dv > pmid THEN 1 WHEN pmid IS NOT NULL AND da / dv < pmid THEN -1
                         ELSE sign(price - pprice) END AS side
          FROM x)
        SELECT symbol_id, minute, sum(dv) AS snap_vol, sum(da) AS snap_amt,
               sum(dv) FILTER (side = 1) AS snap_bvol, sum(dv) FILTER (side = -1) AS snap_svol,
               sum(da) FILTER (side = 1) AS snap_bamt, sum(da) FILTER (side = -1) AS snap_samt
        FROM y GROUP BY ALL""").pl()


def snap_cum(con: duckdb.DuckDBPyConnection, date: str, pm: bool = False) -> pl.DataFrame:
    """Day-cumulative volume / amount of the last snapshot at the bar end, and the opening auction's fills (at 09:31)."""
    d = int(date)
    _, mins, _ = _clocks(pm)
    sn = con.sql(f"""
        SELECT q.symbol_id, {_label('q.time')} AS minute,
               arg_max(q.total_volume::DOUBLE, q.time) AS cvol_snap, arg_max(q.total_amount, q.time) AS camt_snap
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d} AND {_continuous('q.time')}
        GROUP BY ALL""").pl()
    oa = con.sql(f"""
        SELECT r.symbol_id, {AM_OPEN + 1} AS minute, sum(r.volume)::DOUBLE AS oa_vol, sum(r.volume * r.price) AS oa_amt
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {d} AND r.trade_type = 1 AND {mins('r')} < {AM_OPEN}
        GROUP BY ALL""").pl()  # exactly the fills the pm trade flows leave out before the open
    flow = f"(r.dir IN ({BUY}, {SELL}) AND {mins('r')} >= {AM_OPEN})" if pm else f"r.dir IN ({BUY}, {SELL})"
    ca = con.sql(f"""
        SELECT r.symbol_id, {CLOSE} AS minute, sum(r.volume)::DOUBLE AS ca_vol, sum(r.volume * r.price) AS ca_amt
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {d} AND r.trade_type = 1 AND (((r.time + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE) >= {CONT_END} AND NOT {flow}
        GROUP BY ALL""").pl()  # closing-auction fills the flows leave out (exchange time from 14:57)
    out = sn.join(oa, on=["symbol_id", "minute"], how="full", coalesce=True).join(ca, on=["symbol_id", "minute"], how="full", coalesce=True)
    for tag, col in (("x13", "q.time"), ("rx", "q.spider_ts")):  # PM's bar rules: label inclusive, 11:30 reads up to 13:00:00, stop at 14:57:00
        c = _CEIL.format(t=col)
        lab_ = _pm_label(col)
        x = con.sql(f"""
            SELECT q.symbol_id, {lab_} AS minute,
                   arg_max(q.total_volume::DOUBLE, {col}) AS cvol_{tag}, arg_max(q.total_amount, {col}) AS camt_{tag}
            FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d} AND {c} <= {CONT_END}
            GROUP BY ALL""").pl()  # x13 by exchange time (PM's 1min_src), rx by receive time (PM's zzfb profile)
        out = out.join(x, on=["symbol_id", "minute"], how="full", coalesce=True)
    return out


def minute_table(con: duckdb.DuckDBPyConnection, date: str, pm: bool = False, clock: dict | None = None) -> pl.DataFrame:
    """Build the wide 1-minute table for one day (universe registered as ``uni``).

    ``pm``: PM's convention (``bar_variants.py``) -- SSE ticks timed by receive time minus ``SSE_LAG_NS``, the opening
    auction left out of the trade flows, high / low / last over every fill up to 14:57 (opening auction in, closing out).
    ``clock`` collects seconds per stage (trades / orders / snapshots / derive).
    """
    d, b, s = int(date), BUY, SELL
    t, mins, lab = _clocks(pm)
    clock = {} if clock is None else clock
    t0 = time.perf_counter()
    m = mins("r")
    flow = f"r.dir IN ({b}, {s})" + (f" AND {m} >= {AM_OPEN}" if pm else "")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tp AS
        SELECT r.symbol_id, r.market_id, r.dir, r.volume::DOUBLE AS volume, r.price, r.volume * r.price AS amt,
               {lab('r')} AS minute, {t('r')} AS time, r.time AS xtime, r.seq_id, {m} AS m,
               (((r.time + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE) AS mx, {flow} AS flow,
               CASE r.dir WHEN {b} THEN r.buy_seq_id ELSE r.sell_seq_id END AS agg_id,
               CASE r.dir WHEN {b} THEN r.sell_seq_id ELSE r.buy_seq_id END AS pas_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {d} AND r.trade_type = 1{"" if pm else f" AND r.dir IN ({b}, {s})"}""")
    con.execute("CREATE OR REPLACE TEMP TABLE tb AS SELECT * FROM tp WHERE flow")
    px = con.sql(f"""
        SELECT symbol_id, minute, max(price) AS high, min(price) AS low, last(price ORDER BY xtime, seq_id) AS last
        FROM tp WHERE {f"mx <= {CONT_END}" if pm else "flow"} GROUP BY ALL""").pl()  # prices stay on exchange time (ZZDS h / l / c exact)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE ob AS
        SELECT symbol_id, dir, agg_id, sum(volume) AS osize, min(minute) AS minute FROM tb GROUP BY ALL""")
    tr = con.sql(f"""
        SELECT symbol_id, minute, sum(volume) AS vol, sum(amt) AS amt, count(*) AS n,
               count(DISTINCT agg_id) AS ord, count(DISTINCT pas_id) AS pas,
               sum(volume) FILTER (dir = {b}) AS b_vol, sum(volume) FILTER (dir = {s}) AS s_vol,
               sum(amt) FILTER (dir = {b}) AS b_amt, sum(amt) FILTER (dir = {s}) AS s_amt,
               count(*) FILTER (dir = {b}) AS b_n, count(*) FILTER (dir = {s}) AS s_n,
               count(DISTINCT agg_id) FILTER (dir = {b}) AS b_ord, count(DISTINCT agg_id) FILTER (dir = {s}) AS s_ord,
               sum(amt) / sum(volume) AS vwap
        FROM tb GROUP BY ALL""").pl().join(px, on=["symbol_id", "minute"], how="full", coalesce=True)
    oq = con.sql(f"""
        SELECT symbol_id, minute,
               quantile_cont(osize, 0.5) FILTER (dir = {b}) AS b_q50, quantile_cont(osize, 0.9) FILTER (dir = {b}) AS b_q90,
               quantile_cont(osize, 0.5) FILTER (dir = {s}) AS s_q50, quantile_cont(osize, 0.9) FILTER (dir = {s}) AS s_q90
        FROM ob GROUP BY ALL""").pl()
    clock["trades"] = time.perf_counter() - t0; t0 = time.perf_counter()
    ev = order_flows(con, date, pm)
    clock["orders"] = time.perf_counter() - t0; t0 = time.perf_counter()
    sn = con.sql(f"""
        WITH q AS (
          SELECT q.symbol_id, q.market_id, q.time, {_label('q.time')} AS minute,
                 q.bid_px[1] AS bp1, q.ask_px[1] AS ap1, q.bid_vol[1]::DOUBLE AS bv1, q.ask_vol[1]::DOUBLE AS av1,
                 list_sum(q.bid_vol[1:5])::DOUBLE AS bv5, list_sum(q.ask_vol[1:5])::DOUBLE AS av5,
                 list_sum(q.bid_vol)::DOUBLE AS bv10, list_sum(q.ask_vol)::DOUBLE AS av10,
                 q.total_bid_volume::DOUBLE AS btot, q.total_ask_volume::DOUBLE AS atot,
                 q.total_bid_volume * q.weighted_avg_bid_price AS btot_wamt, q.total_ask_volume * q.weighted_avg_ask_price AS atot_wamt,
                 CASE WHEN q.market_id = {SSE} THEN q.total_buy_no END AS bno, CASE WHEN q.market_id = {SSE} THEN q.total_sell_no END AS ano
          FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d} AND {_continuous('q.time')})
        SELECT symbol_id, minute,
               arg_max(bv1, time) AS bv1, arg_max(av1, time) AS av1, arg_max(bv5, time) AS bv5, arg_max(av5, time) AS av5,
               arg_max(bv10, time) AS bv10, arg_max(av10, time) AS av10, arg_max(btot, time) AS btot, arg_max(atot, time) AS atot,
               arg_max(btot_wamt, time) AS btot_wamt, arg_max(atot_wamt, time) AS atot_wamt,
               arg_max(bno, time) AS bno, arg_max(ano, time) AS ano, arg_max(bp1, time) AS bp1, arg_max(ap1, time) AS ap1,
               avg(bv1) AS m_bv1, avg(av1) AS m_av1, avg(bv5) AS m_bv5, avg(av5) AS m_av5, avg(btot) AS m_btot, avg(atot) AS m_atot,
               avg(btot_wamt) AS m_btot_wamt, avg(atot_wamt) AS m_atot_wamt, count(*) AS n_snap
        FROM q GROUP BY ALL""").pl()
    sp = con.sql(f"""
        SELECT q.symbol_id, {lab('q')} AS minute,
               arg_max(q.total_bid_volume * q.weighted_avg_bid_price, {t('q')}) AS btot_wamt_sp,
               arg_max(q.total_ask_volume * q.weighted_avg_ask_price, {t('q')}) AS atot_wamt_sp
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d} AND {_continuous(t('q'))}
        GROUP BY ALL""").pl()  # the same snapshot on the trade clock (SSE receive time)
    sn = sn.join(sp, on=["symbol_id", "minute"], how="full", coalesce=True).join(snapshots_pre(con, date), on=["symbol_id", "minute"], how="full", coalesce=True)
    sn = sn.join(snap_cum(con, date, pm), on=["symbol_id", "minute"], how="full", coalesce=True)
    sn = sn.join(snap_flows(con, date), on=["symbol_id", "minute"], how="full", coalesce=True)
    w = tr.join(oq, on=["symbol_id", "minute"], how="full", coalesce=True).join(ev, on=["symbol_id", "minute"], how="full", coalesce=True)
    w = w.join(sn, on=["symbol_id", "minute"], how="full", coalesce=True).sort("symbol_id", "minute")
    clock["snapshots"] = time.perf_counter() - t0; t0 = time.perf_counter()
    w = derive(w)
    clock["derive"] = time.perf_counter() - t0
    return w


DERIVED_ONE = ("b_osz", "s_osz", "b_tsz", "s_tsz", "pas_fill", "ordb_msz", "orda_msz", "mid", "spread", "btot_amt", "atot_amt")


def is_derived(c: str) -> bool:
    """Columns ``derive`` computes from the base ones (dropped before a patch rebuilds them)."""
    return c in DERIVED_ONE or c.startswith("cum_") or (c.endswith(("_sum", "_mean", "_imb")) and c.rsplit("_", 1)[0] in PAIRS)


def derive(w: pl.DataFrame) -> pl.DataFrame:
    """Float casts, zero-filled flows, ratios, running totals and the bid / ask pair columns (``w`` sorted by stock, minute)."""
    w = w.with_columns(pl.col(c).cast(pl.Float64) for c in w.columns if c not in ("symbol_id", "minute"))
    w = w.with_columns(pl.col(c).fill_null(0.0) for c in FLOW if c in w.columns)
    f = pl.col
    w = w.with_columns(
        (f("b_vol") / f("b_ord")).alias("b_osz"), (f("s_vol") / f("s_ord")).alias("s_osz"),
        (f("b_vol") / f("b_n")).alias("b_tsz"), (f("s_vol") / f("s_n")).alias("s_tsz"), (f("vol") / f("pas")).alias("pas_fill"),
        (f("ordb_vol") / f("ordb_n")).alias("ordb_msz"), (f("orda_vol") / f("orda_n")).alias("orda_msz"),
        ((f("bp1") + f("ap1")) / 2).alias("mid"), (f("ap1") - f("bp1")).alias("spread"),
        (f("btot") * f("bp1")).alias("btot_amt"), (f("atot") * f("ap1")).alias("atot_amt"),
        *[f(c).cum_sum().over("symbol_id").alias(f"cum_{n}") for c, n in (("vol", "vol"), ("amt", "amt"), ("n", "n"), ("b_vol", "bvol"), ("s_vol", "svol"))],
    )
    exprs = [(f(f"cum_{n}") + f(f"oa_{n}").sum().over("symbol_id")).alias(f"cum_{n}_oa") for n in ("vol", "amt") if f"oa_{n}" in w.columns]
    exprs += [(f(f"cum_{n}") + f(f"oa_{n}").sum().over("symbol_id") + f(f"ca_{n}").cum_sum().over("symbol_id")).alias(f"cum_{n}_oac")
              for n in ("vol", "amt") if f"oa_{n}" in w.columns and f"ca_{n}" in w.columns]
    for name, (x, y) in PAIRS.items():
        if x in w.columns and y in w.columns:
            exprs += [(f(x) + f(y)).alias(f"{name}_sum"), ((f(x) + f(y)) / 2).alias(f"{name}_mean"),
                      pl.when(f(x) + f(y) != 0).then((f(x) - f(y)) / (f(x) + f(y))).alias(f"{name}_imb")]
    return w.with_columns(exprs)


def patch(old: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    """``old`` m1 with ``new``'s base columns replaced (or added) and every derived column rebuilt."""
    keys = ["symbol_id", "minute"]
    keep = [c for c in old.columns if c in keys or (not is_derived(c) and c not in new.columns)]
    new = new.with_columns(pl.col(k).cast(old.schema[k]) for k in keys)
    return derive(old.select(keep).join(new, on=keys, how="full", coalesce=True).sort(keys))


def aggregate(m1: pl.DataFrame, step: int | None) -> pl.DataFrame:
    """5-minute bars (step 5; the bar ending at t holds minutes t-4..t) or the day (None): flows summed, states last, means averaged."""
    key = (((pl.col("minute") - 571) // step) * step + 575) if step else pl.lit(-1)
    state = {"last", "bv1", "av1", "bv5", "av5", "bv10", "av10", "btot", "atot", "bno", "ano", "bp1", "ap1", "mid", "spread", "btot_amt", "atot_amt", "btot_wamt", "atot_wamt", "btot_wamt_sp", "atot_wamt_sp", "btot_pre", "atot_pre", "btot_wamt_pre", "atot_wamt_pre", "cvol_pre", "camt_pre", "cvol_snap", "camt_snap", "cvol_x13", "camt_x13", "cvol_rx", "camt_rx"}
    flow_pairs = {p for p, (x, _) in PAIRS.items() if x in FLOW}
    agg = []
    for c in [c for c in m1.columns if c not in ("symbol_id", "minute")]:
        stem = c.rsplit("_", 1)[0] if c.endswith(("_sum", "_mean", "_imb")) else None
        if c in FLOW or c == "n_snap" or (stem in flow_pairs and c.endswith(("_sum", "_mean"))):
            agg.append(pl.col(c).sum().alias(c))
        elif c in state or c.startswith("cum_") or (stem is not None and stem not in flow_pairs and not c.endswith("_imb") and PAIRS[stem][0] in state):
            agg.append(pl.col(c).drop_nulls().last().alias(c))
        elif c == "high":
            agg.append(pl.col(c).max().alias(c))
        elif c == "low":
            agg.append(pl.col(c).min().alias(c))
        else:
            agg.append(pl.col(c).mean().alias(c))
    out = m1.with_columns(key.alias("bar")).group_by("symbol_id", "bar").agg(agg).rename({"bar": "minute"}).sort("symbol_id", "minute")
    return out.with_columns(pl.when(pl.col("minute") > 900).then(900).otherwise(pl.col("minute")).alias("minute")) if step else out


def build_day(d: str, index: str, thr_file: Path | None, memory: str, out: Path, threads: int, pm: bool = False, only: str = "") -> str:
    """Build one day's basics tables (its own catalog connection, so days can run in parallel).

    ``only="orders"``: recompute just the order / cancel columns and patch them into the day's existing tables.
    """
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{memory}'")
    tmp = DATA / "tmp" / f"basics_{os.getpid()}"  # parallel workers must not share DuckDB spill files
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute(f"SET threads={threads}")
    uni = ced_universe(index, d)
    if uni is None and thr_file:
        thr = pl.read_parquet(thr_file)
        uni = thr.select(pl.col("sid").cast(pl.Int32).alias("symbol_id"),
                         pl.when(pl.col("sid") >= 600000).then(3553).otherwise(3554).cast(pl.Int16).alias("market_id"))
    dst = out / f"date={d}"
    clock: dict[str, float] = {}
    if only:
        old = pl.read_parquet(dst / "m1.parquet")
        uni = old.select(pl.col("symbol_id").cast(pl.Int32).unique()).with_columns(  # patch exactly the stocks already there
            pl.when(pl.col("symbol_id") >= 600000).then(SSE).otherwise(SZSE).cast(pl.Int16).alias("market_id"))
        con.register("uni", uni)
        parts = {"orders": lambda: order_flows(con, d, pm), "pre": lambda: snapshots_pre(con, d), "cum": lambda: snap_cum(con, d, pm), "snap": lambda: snap_flows(con, d)}
        new = None
        for part in only.split(","):
            t0 = time.perf_counter()
            x = parts[part]()
            clock[part] = time.perf_counter() - t0
            new = x if new is None else new.join(x, on=["symbol_id", "minute"], how="full", coalesce=True)
        m1 = patch(old, new)
    else:
        con.register("uni", uni if uni is not None else universe(index))
        m1 = minute_table(con, d, pm, clock)
    dst.mkdir(parents=True, exist_ok=True)
    m1.write_parquet(dst / "m1.parquet")
    aggregate(m1, 5).write_parquet(dst / "m5.parquet")
    aggregate(m1, None).write_parquet(dst / "d1.parquet")
    con.close()
    secs = " ".join(f"{k} {v:.0f}s" for k, v in clock.items())
    return f"{d}: {m1.height:,} stock-minutes, {m1.width - 2} basics -> {dst} ({secs})"


def _job(args: tuple) -> str:
    try:
        return build_day(*args)
    except Exception as e:  # one bad day must not stop the others
        return f"{args[0]}: FAILED {type(e).__name__}: {str(e)[:300]}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--index", default="000300")
    ap.add_argument("--thr-file", type=Path, help="PM's samplerR: its sid column is the universe when CED lacks the day (one file for every date)")
    ap.add_argument("--thr-dir", type=Path, help="PM's ZZCR folder: each date takes the samplerR of the latest earlier PM day")
    ap.add_argument("--memory", default="8GB", help="per worker")
    ap.add_argument("--workers", type=int, default=1, help="days built in parallel")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB (and every other library pool) threads per worker")
    ap.add_argument("--convention", choices=("feitu", "pm"), default="feitu", help="pm: PM's clock / auction / price rules, default --out basics_pm")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", default="", help="comma list of orders / pre / cum / snap: patch just those columns into existing tables")
    a = ap.parse_args()
    dates = [d for d in a.dates.split(",") if d]
    pm_days = sorted(p.name for p in a.thr_dir.iterdir() if p.name.isdigit()) if a.thr_dir else []

    def thr_of(d: str) -> Path | None:
        if a.thr_dir:
            prev = [x for x in pm_days if x < d] or pm_days[:1]  # before PM's first day: its first universe (HS300 rebalances twice a year)
            return a.thr_dir / prev[-1] / "samplerR.parquet" if prev else None
        return a.thr_file

    if a.workers > 1:
        threads.cap(a.threads)  # the workers' polars / BLAS pools would otherwise each take the whole machine
    pm = a.convention == "pm"
    bad = set(filter(None, a.only.split(","))) - {"orders", "pre", "cum", "snap"}
    if bad:
        sys.exit(f"--only: unknown part(s) {sorted(bad)}, choose from orders, pre, cum, snap")
    out = a.out or (OUT_PM if pm else OUT)
    jobs = [(d, a.index, thr_of(d), a.memory, out, a.threads, pm, a.only) for d in dates]
    failed = 0
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            msgs = pool.imap_unordered(_job, jobs)
            for msg in msgs:
                failed += "FAILED" in msg
                print(msg, flush=True)
    else:
        for job in jobs:
            msg = _job(job)
            failed += "FAILED" in msg
            print(msg, flush=True)
    if failed:
        sys.exit(f"{failed} of {len(jobs)} days FAILED")


if __name__ == "__main__":
    main()
