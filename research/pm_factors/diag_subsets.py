r"""Stage 4 search: which basic per-5-minute quantity equals each PM samplerS column.

The first ranking showed the five model-input families (ZZWA, ZZAL, ZZQO, ZZXC, ZZTS) are not
our first-guess formulas, and the PM values say what kind of thing they are: ZZXC's ``n / q`` is
the stock's price (a CNY amount and a volume of the same trades, ~77 % of all volume), ZZAL's
``an`` / ``bn`` look like CNY value of the resting book, ZZQO / ZZTS like CNY amounts of a subset,
ZZWA like running ratios. So this builds a broad library of per-bar quantities for one day and
scores every PM column against every item:

* trades: volume / amount / count -- all, aggressor buy / sell, and size tiers of the aggressor
  order (filled day total) and of the passive order against PM's own samplerR thresholds
  (``--thr``, ZZCR layout: ``as/ab/ps/pb.q50..q98``);
* order stream: new orders and cancels, volume / amount / count per side;
* snapshots (last in the bar): total ask / bid volume, their CNY value (x weighted average
  price), weighted average prices and their distance from the mid;
* each per-bar item also cumulated from the open (``cum:``); with ``--pairs``, ratios of cumulated
  items (``a/b``) for the PM columns that look like ratios (ZZWA).

Bars are right-labelled five-minute windows on exchange time, with the opening auction as its own
bar (09:30) and the close auction inside 15:00. PM's bar labels are tried as end labels (offset 0)
and start labels (offset +5). Per PM column the best items are printed with ``exact`` (equal to 1e-6
relative, or within 0.5 for values >= 1000), ``w1`` (within 1 %), median ratio and Spearman correlation.

    python research/pm_factors/diag_subsets.py --date 20260429 \
        --thr /data/prod/hstats/HS300/ZZCR/20260429/samplerR.parquet --pairs ZZWA
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampler_r as R  # catalog, universe, side codes

FAMILIES = ("ZZWA", "ZZAL", "ZZQO", "ZZXC", "ZZTS")
NS = 1_000_000_000
LOCAL = 8 * 3600
TIERS = (50, 90, 95, 98)


def _m5(col: str) -> str:
    """Right-labelled five-minute bar (local minutes); opening auction -> 570."""
    m = f"((({col} // {NS}) + {LOCAL}) % 86400) / 60.0"
    return f"greatest(570, ceil(({m}) / 5) * 5)::INTEGER"


def library(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """Long frame (sid, minute, item, value) of per-bar quantities for ``date``."""
    d = int(date)
    con.execute(f"""
      CREATE OR REPLACE TEMP TABLE t AS
      SELECT r.symbol_id, r.market_id, r.dir, r.volume, r.price, {_m5('r.time')} AS m5,
             CASE r.dir WHEN {R.BUY} THEN r.buy_seq_id ELSE r.sell_seq_id END AS agg,
             CASE r.dir WHEN {R.BUY} THEN r.sell_seq_id ELSE r.buy_seq_id END AS pas
      FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
      WHERE r.date = {d} AND r.trade_type = 1""")
    con.execute("""
      CREATE OR REPLACE TEMP TABLE t2 AS
      SELECT t.*, sum(volume) OVER (PARTITION BY symbol_id, market_id, agg) AS asz,
             sum(volume) OVER (PARTITION BY symbol_id, market_id, pas) AS psz FROM t""")
    tier = []
    for q in TIERS:
        tier += [f"(dir = {R.SELL} AND asz >= th.\"as.q{q}\") OR (dir = {R.BUY} AND asz >= th.\"ab.q{q}\")",
                 f"(dir = {R.SELL} AND psz >= th.\"pb.q{q}\") OR (dir = {R.BUY} AND psz >= th.\"ps.q{q}\")"]
    subsets = {"all": "TRUE", "buy": f"dir = {R.BUY}", "sell": f"dir = {R.SELL}", "cont": "m5 > 570 AND m5 < 900"}
    for q, (a, p) in zip(TIERS, itertools.batched(tier, 2), strict=True):
        subsets[f"agg>=q{q}"] = a
        subsets[f"agg<q{q}"] = f"NOT ({a})"
        subsets[f"pas>=q{q}"] = p
        subsets[f"pas<q{q}"] = f"NOT ({p})"
        subsets[f"buy_agg>=q{q}"] = f"dir = {R.BUY} AND asz >= th.\"ab.q{q}\""
        subsets[f"sell_agg>=q{q}"] = f"dir = {R.SELL} AND asz >= th.\"as.q{q}\""
    sel = []
    for name, cond in subsets.items():
        sel += [f"sum(volume) FILTER ({cond})::DOUBLE AS \"tr_vol:{name}\"",
                f"sum(volume * price) FILTER ({cond}) AS \"tr_amt:{name}\"",
                f"count(*) FILTER ({cond})::DOUBLE AS \"tr_cnt:{name}\"",
                f"count(DISTINCT agg) FILTER ({cond})::DOUBLE AS \"tr_ord:{name}\""]
    trades = con.sql(f"""SELECT symbol_id AS sid, m5 AS minute, {', '.join(sel)}
                         FROM t2 JOIN thr th ON th.sid = t2.symbol_id GROUP BY ALL""").pl()
    orders = con.sql(f"""
      WITH o AS (
        SELECT o.symbol_id, o.dir, o.volume, o.price, {_m5('o.time')} AS m5,
               CASE WHEN o.market_id = 3553 AND o.update_type = 2 THEN 1 ELSE 0 END AS cxl
        FROM raw_order o JOIN uni u USING (symbol_id, market_id) WHERE o.date = {d}
        UNION ALL
        SELECT r.symbol_id, CASE WHEN r.buy_seq_id > 0 THEN {R.BUY} ELSE {R.SELL} END, r.volume, NULL,
               {_m5('r.time')}, 1
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {d} AND r.market_id = 3554 AND r.trade_type = 2)
      SELECT symbol_id AS sid, m5 AS minute,
        sum(volume) FILTER (cxl = 0 AND dir = {R.SELL})::DOUBLE AS "add_vol:ask", sum(volume) FILTER (cxl = 0 AND dir = {R.BUY})::DOUBLE AS "add_vol:bid",
        sum(volume * price) FILTER (cxl = 0 AND dir = {R.SELL}) AS "add_amt:ask", sum(volume * price) FILTER (cxl = 0 AND dir = {R.BUY}) AS "add_amt:bid",
        count(*) FILTER (cxl = 0 AND dir = {R.SELL})::DOUBLE AS "add_cnt:ask", count(*) FILTER (cxl = 0 AND dir = {R.BUY})::DOUBLE AS "add_cnt:bid",
        sum(volume) FILTER (cxl = 1 AND dir = {R.SELL})::DOUBLE AS "cxl_vol:ask", sum(volume) FILTER (cxl = 1 AND dir = {R.BUY})::DOUBLE AS "cxl_vol:bid",
        count(*) FILTER (cxl = 1 AND dir = {R.SELL})::DOUBLE AS "cxl_cnt:ask", count(*) FILTER (cxl = 1 AND dir = {R.BUY})::DOUBLE AS "cxl_cnt:bid"
      FROM o GROUP BY ALL""").pl()
    snaps = con.sql(f"""
      WITH q AS (
        SELECT q.symbol_id, {_m5('q.time')} AS m5, q.time, q.total_ask_volume AS av, q.total_bid_volume AS bv,
               q.weighted_avg_ask_price AS wa, q.weighted_avg_bid_price AS wb, q.ask_px[1] AS a1, q.bid_px[1] AS b1,
               q.price AS px
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id) WHERE q.date = {d})
      SELECT symbol_id AS sid, m5 AS minute,
        arg_max(av, time)::DOUBLE AS "book_vol:ask", arg_max(bv, time)::DOUBLE AS "book_vol:bid",
        arg_max(av * wa, time) AS "book_amt:ask", arg_max(bv * wb, time) AS "book_amt:bid",
        arg_max(wa, time) AS "book_wpx:ask", arg_max(wb, time) AS "book_wpx:bid",
        arg_max(ln(nullif(wa, 0) / nullif((a1 + b1) / 2, 0)), time) AS "book_wdist:ask",
        arg_max(ln(nullif(wb, 0) / nullif((a1 + b1) / 2, 0)), time) AS "book_wdist:bid",
        arg_max(ln(nullif(wa, 0) / nullif(px, 0)), time) AS "book_wdist_px:ask", arg_max(ln(nullif(wb, 0) / nullif(px, 0)), time) AS "book_wdist_px:bid"
      FROM q GROUP BY ALL""").pl()
    wide = trades.join(orders, on=["sid", "minute"], how="full", coalesce=True).join(
        snaps, on=["sid", "minute"], how="full", coalesce=True)
    items = [c for c in wide.columns if c not in ("sid", "minute")]
    flow = [c for c in items if not c.startswith("book_")]
    wide = wide.sort("sid", "minute").with_columns(
        [pl.col(c).fill_null(0).cum_sum().over("sid").alias(f"cum:{c}") for c in flow])
    return wide.with_columns(pl.col("sid").cast(pl.Float64))


def score(pm: pl.DataFrame, lib: pl.DataFrame, items: list[str]) -> pl.DataFrame:
    """Score one PM column (sid, minute, pm) against every library item, offsets 0 and +5."""
    rows = []
    for off in (0, 5):
        j = pm.with_columns(pl.col("minute") + off).join(lib, on=["sid", "minute"], how="inner")
        a = pl.col("pm")
        for it in items:
            b = pl.col(it)
            ok = j.filter(b.is_not_null() & b.is_finite() & a.is_finite())
            if ok.height < 50:
                continue
            rows.append(ok.select(
                pl.lit(it).alias("item"), pl.lit(off).alias("offset"), pl.len().alias("n"),
                # relative 1e-6; the 0.5 grid allowance only for share / CNY sized values
                ((a - b).abs() <= pl.max_horizontal(1e-6 * a.abs(), pl.when(a.abs() >= 1000).then(0.5).otherwise(1e-12)))
                .mean().alias("exact"),
                ((a / b - 1).abs() <= 0.01).mean().alias("w1"),
                (a / b).filter(b != 0).median().alias("ratio"),
                pl.corr(a, b, method="spearman").alias("spearman")))
    return pl.concat(rows) if rows else pl.DataFrame()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--thr", type=Path, required=True, help="samplerR thresholds (PM's ZZCR file)")
    ap.add_argument("--pm-dir", type=Path, default=Path("data/pm_export"), help="<F>_<date>.parquet exports")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--pairs", default="ZZWA", help="families whose columns are also tried against cum-ratios")
    ap.add_argument("--index", default="000300")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--memory", default="200GB")
    a = ap.parse_args()
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    uni = R.ced_universe(a.index, a.date)
    con.register("uni", uni if uni is not None else R.universe(a.index))
    con.register("thr", pl.read_parquet(a.thr))
    lib = library(con, a.date)
    items = [c for c in lib.columns if c not in ("sid", "minute")]
    print(f"library: {len(items)} items, {lib['sid'].n_unique()} stocks, bars {sorted(lib['minute'].unique().to_list())[:3]} ..")
    pair_fams = set(a.pairs.split(",")) if a.pairs else set()
    base = ("tr_vol:all", "tr_vol:buy", "tr_vol:sell", "tr_amt:all", "tr_amt:buy", "tr_amt:sell", "tr_cnt:all",
            "tr_cnt:buy", "tr_cnt:sell", "add_vol:ask", "add_vol:bid", "add_cnt:ask", "add_cnt:bid", "cxl_vol:ask",
            "cxl_vol:bid", "cxl_cnt:ask", "cxl_cnt:bid", "add_amt:ask", "add_amt:bid")
    cums = [f"cum:{b}" for b in base if f"cum:{b}" in items]  # ratio pairs only over these (~340 pairs)
    out = []
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4, fmt_str_lengths=60):
        for fam in a.families.split(","):
            f = a.pm_dir / f"{fam}_{a.date}.parquet"
            if not f.exists():
                print(f"== {fam}: {f} missing")
                continue
            pm = pl.read_parquet(f)
            hm = pm["I"].str.extract_groups(r"^(\d{1,2}):(\d{2})")
            pm = pm.with_columns(pl.col("S").cast(pl.Int64).cast(pl.Float64).alias("sid"),
                                 (hm.struct.field("1").cast(pl.Int64) * 60 + hm.struct.field("2").cast(pl.Int64)).alias("minute"))
            for v in sorted(pm["V"].unique().to_list()):
                p = pm.filter(pl.col("V") == v).select("sid", "minute", pl.col("x").alias("pm"))
                res = score(p, lib, items)
                if fam in pair_fams:
                    rat = lib.select("sid", "minute", *[(pl.col(x) / pl.col(y)).alias(f"{x}/{y}")
                                                       for x, y in itertools.permutations(cums, 2)])
                    res = pl.concat([res, score(p, rat, [c for c in rat.columns if c not in ("sid", "minute")])])
                res = res.with_columns(pl.lit(fam).alias("family"), pl.lit(v).alias("V"),
                                       pl.col("spearman").abs().alias("abs_rho"))
                out.append(res)
                print(f"== {fam}.{v}")
                print(res.sort("exact", "w1", "abs_rho", descending=True).head(a.top)
                      .select("item", "offset", "n", "exact", "w1", "ratio", "spearman"))
    dst = Path("data/reports/stage4")
    dst.mkdir(parents=True, exist_ok=True)
    pl.concat(out).write_parquet(dst / f"subsets_{a.date}.parquet")
    print(f"full table: {dst / f'subsets_{a.date}.parquet'}")


if __name__ == "__main__":
    main()
