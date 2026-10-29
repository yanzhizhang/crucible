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
  bar means ``m_bv1 m_av1 m_bv5 m_av5 m_btot m_atot``; ``*_amt`` = volume x price for the totals;
* running totals since the open: ``cum_vol cum_amt cum_n cum_bvol cum_svol``;
* for every bid / ask or buy / sell pair: ``<x>_sum``, ``<x>_mean``, ``<x>_imb``.

The same table aggregated to 5-minute bars (``m5``) and the day (``d1``): flows summed, states taken last, means
averaged. Output: ``store/pm_factors/basics/date=<D>/{m1,m5,d1}.parquet`` (``symbol_id``, ``minute``, columns).

    python research/pm_factors/basics.py --dates 20261009 --thr-file /data/prod/hstats/HS300/ZZCR/20261008/samplerR.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidates import (
    BUY,
    CATALOG,
    DATA,
    SELL,
    SSE,
    SZSE,
    _continuous,
    _label,
    ced_universe,
    universe,
)

OUT = DATA / "store" / "pm_factors" / "basics"
FLOW = ("vol", "amt", "n", "ord", "pas", "b_vol", "s_vol", "b_amt", "s_amt", "b_n", "s_n", "b_ord", "s_ord",
        "ordb_n", "orda_n", "ordb_vol", "orda_vol", "ordb_amt", "orda_amt", "cxb_n", "cxa_n", "cxb_vol", "cxa_vol")
PAIRS = {"vol": ("b_vol", "s_vol"), "amt": ("b_amt", "s_amt"), "n": ("b_n", "s_n"), "ord": ("b_ord", "s_ord"),
         "osz": ("b_osz", "s_osz"), "tsz": ("b_tsz", "s_tsz"), "q50": ("b_q50", "s_q50"), "q90": ("b_q90", "s_q90"),
         "ordn": ("ordb_n", "orda_n"), "ordvol": ("ordb_vol", "orda_vol"), "ordamt": ("ordb_amt", "orda_amt"),
         "ordmsz": ("ordb_msz", "orda_msz"), "cxn": ("cxb_n", "cxa_n"), "cxvol": ("cxb_vol", "cxa_vol"),
         "v1": ("bv1", "av1"), "v5": ("bv5", "av5"), "v10": ("bv10", "av10"), "tot": ("btot", "atot"), "no": ("bno", "ano"),
         "mv1": ("m_bv1", "m_av1"), "mv5": ("m_bv5", "m_av5"), "mtot": ("m_btot", "m_atot"), "totamt": ("btot_amt", "atot_amt")}


def minute_table(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """Build the wide 1-minute table for one day (universe registered as ``uni``)."""
    d, b, s = int(date), BUY, SELL
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tb AS
        SELECT r.symbol_id, r.market_id, r.dir, r.volume::DOUBLE AS volume, r.price, r.volume * r.price AS amt,
               {_label('r.time')} AS minute, r.time,
               CASE r.dir WHEN {b} THEN r.buy_seq_id ELSE r.sell_seq_id END AS agg_id,
               CASE r.dir WHEN {b} THEN r.sell_seq_id ELSE r.buy_seq_id END AS pas_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {d} AND r.trade_type = 1 AND r.dir IN ({b}, {s})""")
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
               sum(amt) / sum(volume) AS vwap, max(price) AS high, min(price) AS low, arg_max(price, time) AS last
        FROM tb GROUP BY ALL""").pl()
    oq = con.sql(f"""
        SELECT symbol_id, minute,
               quantile_cont(osize, 0.5) FILTER (dir = {b}) AS b_q50, quantile_cont(osize, 0.9) FILTER (dir = {b}) AS b_q90,
               quantile_cont(osize, 0.5) FILTER (dir = {s}) AS s_q50, quantile_cont(osize, 0.9) FILTER (dir = {s}) AS s_q90
        FROM ob GROUP BY ALL""").pl()
    ev = con.sql(f"""
        WITH e AS (
          SELECT o.symbol_id, {_label('o.time')} AS minute, 0 AS k, o.dir, o.volume::DOUBLE AS volume, o.volume * o.price AS amt
          FROM raw_order o JOIN uni u USING (symbol_id, market_id) WHERE o.date = {d} AND o.update_type = 1 AND o.dir IN ({b}, {s})
          UNION ALL
          SELECT o.symbol_id, {_label('o.time')}, 1, o.dir, o.volume::DOUBLE, o.volume * o.price
          FROM raw_order o JOIN uni u USING (symbol_id, market_id) WHERE o.date = {d} AND o.market_id = {SSE} AND o.update_type = 2
          UNION ALL
          SELECT r.symbol_id, {_label('r.time')}, 1, CASE WHEN r.buy_seq_id > 0 THEN {b} ELSE {s} END, r.volume::DOUBLE, r.volume * r.price
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id) WHERE r.date = {d} AND r.market_id = {SZSE} AND r.trade_type = 2)
        SELECT symbol_id, minute,
               count(*) FILTER (k = 0 AND dir = {b}) AS ordb_n, count(*) FILTER (k = 0 AND dir = {s}) AS orda_n,
               sum(volume) FILTER (k = 0 AND dir = {b}) AS ordb_vol, sum(volume) FILTER (k = 0 AND dir = {s}) AS orda_vol,
               sum(amt) FILTER (k = 0 AND dir = {b}) AS ordb_amt, sum(amt) FILTER (k = 0 AND dir = {s}) AS orda_amt,
               count(*) FILTER (k = 1 AND dir = {b}) AS cxb_n, count(*) FILTER (k = 1 AND dir = {s}) AS cxa_n,
               sum(volume) FILTER (k = 1 AND dir = {b}) AS cxb_vol, sum(volume) FILTER (k = 1 AND dir = {s}) AS cxa_vol
        FROM e GROUP BY ALL""").pl()
    sn = con.sql(f"""
        WITH q AS (
          SELECT q.symbol_id, q.market_id, q.time, {_label('q.time')} AS minute,
                 q.bid_px[1] AS bp1, q.ask_px[1] AS ap1, q.bid_vol[1]::DOUBLE AS bv1, q.ask_vol[1]::DOUBLE AS av1,
                 list_sum(q.bid_vol[1:5])::DOUBLE AS bv5, list_sum(q.ask_vol[1:5])::DOUBLE AS av5,
                 list_sum(q.bid_vol)::DOUBLE AS bv10, list_sum(q.ask_vol)::DOUBLE AS av10,
                 q.total_bid_volume::DOUBLE AS btot, q.total_ask_volume::DOUBLE AS atot,
                 CASE WHEN q.market_id = {SSE} THEN q.total_buy_no END AS bno, CASE WHEN q.market_id = {SSE} THEN q.total_sell_no END AS ano
          FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d} AND {_continuous('q.time')})
        SELECT symbol_id, minute,
               arg_max(bv1, time) AS bv1, arg_max(av1, time) AS av1, arg_max(bv5, time) AS bv5, arg_max(av5, time) AS av5,
               arg_max(bv10, time) AS bv10, arg_max(av10, time) AS av10, arg_max(btot, time) AS btot, arg_max(atot, time) AS atot,
               arg_max(bno, time) AS bno, arg_max(ano, time) AS ano, arg_max(bp1, time) AS bp1, arg_max(ap1, time) AS ap1,
               avg(bv1) AS m_bv1, avg(av1) AS m_av1, avg(bv5) AS m_bv5, avg(av5) AS m_av5, avg(btot) AS m_btot, avg(atot) AS m_atot,
               count(*) AS n_snap
        FROM q GROUP BY ALL""").pl()
    w = tr.join(oq, on=["symbol_id", "minute"], how="full", coalesce=True).join(ev, on=["symbol_id", "minute"], how="full", coalesce=True)
    w = w.join(sn, on=["symbol_id", "minute"], how="full", coalesce=True).sort("symbol_id", "minute")
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
    exprs = []
    for name, (x, y) in PAIRS.items():
        if x in w.columns and y in w.columns:
            exprs += [(f(x) + f(y)).alias(f"{name}_sum"), ((f(x) + f(y)) / 2).alias(f"{name}_mean"),
                      pl.when(f(x) + f(y) != 0).then((f(x) - f(y)) / (f(x) + f(y))).alias(f"{name}_imb")]
    return w.with_columns(exprs)


def aggregate(m1: pl.DataFrame, step: int | None) -> pl.DataFrame:
    """5-minute bars (step 5; the bar ending at t holds minutes t-4..t) or the day (None): flows summed, states last, means averaged."""
    key = (((pl.col("minute") - 571) // step) * step + 575) if step else pl.lit(-1)
    state = {"last", "bv1", "av1", "bv5", "av5", "bv10", "av10", "btot", "atot", "bno", "ano", "bp1", "ap1", "mid", "spread", "btot_amt", "atot_amt"}
    flow_pairs = {p for p, (x, _) in PAIRS.items() if x in FLOW}
    agg = []
    for c in [c for c in m1.columns if c not in ("symbol_id", "minute")]:
        stem = c.rsplit("_", 1)[0] if c.endswith(("_sum", "_mean", "_imb")) else None
        if c in FLOW or c == "n_snap" or (stem in flow_pairs and c.endswith(("_sum", "_mean"))):
            agg.append(pl.col(c).sum().alias(c))
        elif c in state or c.startswith("cum_") or (stem is not None and stem not in flow_pairs and not c.endswith("_imb") and PAIRS[stem][0] in state):
            agg.append(pl.col(c).last().alias(c))
        elif c == "high":
            agg.append(pl.col(c).max().alias(c))
        elif c == "low":
            agg.append(pl.col(c).min().alias(c))
        else:
            agg.append(pl.col(c).mean().alias(c))
    out = m1.with_columns(key.alias("bar")).group_by("symbol_id", "bar").agg(agg).rename({"bar": "minute"}).sort("symbol_id", "minute")
    return out.with_columns(pl.when(pl.col("minute") > 900).then(900).otherwise(pl.col("minute")).alias("minute")) if step else out


def build_day(d: str, index: str, thr_file: Path | None, memory: str, out: Path, threads: int) -> str:
    """Build one day's basics tables (its own catalog connection, so days can run in parallel)."""
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{memory}'")
    con.execute(f"SET threads={threads}")
    uni = ced_universe(index, d)
    if uni is None and thr_file:
        thr = pl.read_parquet(thr_file)
        uni = thr.select(pl.col("sid").cast(pl.Int32).alias("symbol_id"),
                         pl.when(pl.col("sid") >= 600000).then(3553).otherwise(3554).cast(pl.Int16).alias("market_id"))
    con.register("uni", uni if uni is not None else universe(index))
    m1 = minute_table(con, d)
    dst = out / f"date={d}"
    dst.mkdir(parents=True, exist_ok=True)
    m1.write_parquet(dst / "m1.parquet")
    aggregate(m1, 5).write_parquet(dst / "m5.parquet")
    aggregate(m1, None).write_parquet(dst / "d1.parquet")
    con.close()
    return f"{d}: {m1.height:,} stock-minutes, {m1.width - 2} basics -> {dst}"


def _job(args: tuple) -> str:
    return build_day(*args)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--index", default="000300")
    ap.add_argument("--thr-file", type=Path, help="PM's samplerR: its sid column is the universe when CED lacks the day (one file for every date)")
    ap.add_argument("--thr-dir", type=Path, help="PM's ZZCR folder: each date takes the samplerR of the latest earlier PM day")
    ap.add_argument("--memory", default="8GB", help="per worker")
    ap.add_argument("--workers", type=int, default=1, help="days built in parallel")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads per worker")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    dates = [d for d in a.dates.split(",") if d]
    pm_days = sorted(p.name for p in a.thr_dir.iterdir() if p.name.isdigit()) if a.thr_dir else []

    def thr_of(d: str) -> Path | None:
        if a.thr_dir:
            prev = [x for x in pm_days if x < d]
            return a.thr_dir / prev[-1] / "samplerR.parquet" if prev else None
        return a.thr_file

    jobs = [(d, a.index, thr_of(d), a.memory, a.out, a.threads) for d in dates]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for msg in pool.imap_unordered(_job, jobs):
                print(msg, flush=True)
    else:
        for job in jobs:
            print(build_day(*job), flush=True)


if __name__ == "__main__":
    main()
