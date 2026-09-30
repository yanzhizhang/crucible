"""Stage 4: candidate formulas for PM factor families ZZUG, ZZDS, ZZQI, ZZVC (+ the rest).

The PM's ``samplerS.nc`` only gives each family's column names (``V``) and slot count (``I``);
the formulas are unknown. This module computes **several candidate formulas per column** on the
same days, so the intranet comparison (:mod:`research.pm_factors.rank_candidates`) can pick the
one whose values line up with the PM's. Nothing here is believed to be *the* formula.

Output: one long frame per family and day,
``/work/crucible_data/store/pm_factors/candidates/family=<F>/date=<D>/part.parquet`` with
columns ``date`` (int yyyymmdd), ``sid`` (float, 6-digit code as a number -- PM's samplerR key),
``minute`` (minute of day of a right-labelled 1-minute bar, e.g. 571 = 09:31; -1 for a daily
family), ``cand`` (candidate name), ``V`` (``<pm column>@<variant>``, e.g. ``ab@imb``) and
``value``.

Bar labels follow ``research/build_bars_from_l2.py``: bar ``t`` covers ``(t-1min, t]``
exchange time; the open auction folds into 09:31, the lunch break into 13:01, the close auction
into the 15:00 bar. Snapshot families (QI) use continuous-session snapshots only.

Pair columns (``ab``, ``cd``, ``ef``, ``qab``, ``nab``) are written in three forms, since the
combination rule is unknown too: ``@imb`` = (x-y)/(x+y), ``@diff`` = x-y, ``@share`` = x/(x+y).

Candidates
----------

ZZVC (1 min; ab a b cd c d) -- a/b = active buy / sell volume in every candidate, and c/d:

* ``cnt``   c/d = active buy / sell trade count
* ``amt``   c/d = active buy / sell amount (CNY)
* ``ord``   c/d = number of distinct aggressor orders, buy / sell
* ``size``  a/b = active volume of *small* aggressor orders (day total < samplerR q90),
  c/d = of large ones (>= q90)
* ``flow``  a/b = new limit-order volume (bid / ask side), c/d = cancelled volume

ZZUG (1 min; ab a b cd c d ef e f) -- three size tiers of aggressor orders, classified by the
order's size against that stock's samplerR thresholds (``ab.*`` for buys, ``as.*`` for sells);
a/b = tier 1 buy / sell active volume, c/d = tier 2, e/f = tier 3:

* ``t50_90``     <q50 | q50..q90 | >=q90
* ``t90_98``     <q90 | q90..q98 | >=q98
* ``t90_95_98``  q90..q95 | q95..q98 | >=q98 (large orders only)
* ``c90_95_98``  >=q90 | >=q95 | >=q98 (cumulative)

"Order size" follows the samplerR variant (``--thr-variant``): an order's day total for
``order_*``, the trade's volume for ``trade_all``.

ZZQI (1 min; ab a b cd c d s m) -- order book from snapshots; a/b = bid1 / ask1 volume:

* ``last5``   c/d = bid / ask volume over levels 1-5, state at the bar's last snapshot
* ``mean5``   same, averaged over the bar's snapshots
* ``last10``  c/d over levels 1-10
* ``lasttot`` c/d = total resting bid / ask volume (snapshot totals)

and ``s`` / ``m`` in several units: ``s@abs`` (CNY), ``s@rel`` (/mid), ``s@ticks`` (/0.01),
``m@mid`` (mid price), ``m@ret`` (log mid change over the bar), ``m@micro``
((microprice - mid) / mid).

ZZDS (daily; qab qa qb nab na nb) -- resting book, qa/qb = ask / bid volume, na/nb = ask / bid
order count:

* ``snapclose``  snapshot totals at the last continuous snapshot (na/nb: SSE only -- SZSE
  snapshots publish no order counts, and the per-level counts are empty on both venues)
* ``snapmean``   the same fields averaged over the continuous session
* ``lvl10close`` qa/qb = sum of the 10 published levels; na/nb = number of price levels (SSE)
* ``mboclose``   rebuilt from the order and trade streams: volume and count of orders still
  resting at 14:57 (approximate: SZSE market orders are ignored)

The other ten families (HL MS GW XC TS QO WA AL SQ CR) live in :mod:`candidates_more`; this
CLI runs them too.

Thresholds come from ``store/pm_factors/sampler_r/variant=<v>/date=<prev>/`` -- the PM builds
``sod/<D>`` from day ``D-1``. When no earlier day exists (the local days are not consecutive)
the same day is used and a warning is printed; the intranet run has consecutive days.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench.resmon import ResourceMonitor, append_history
from sampler_r import BUY, CATALOG, DATA, SELL, universe

OUT = DATA / "store" / "pm_factors" / "candidates"
THR = DATA / "store" / "pm_factors" / "sampler_r"
SSE, SZSE = 3553, 3554
NS_MIN = 60_000_000_000
NS_DAY = 86_400_000_000_000
NS_LOCAL = 8 * 3600 * 1_000_000_000
AM_OPEN, AM_CLOSE, PM_OPEN, CONT_END, CLOSE = 570, 690, 780, 897, 900  # minutes of day

# right-labelled minute of day of an exchange ns timestamp, auctions / lunch folded like the bars
_CEIL = f"CAST(ceil((({{t}} + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE) AS INTEGER)"


def _label(col: str = "time") -> str:
    c = _CEIL.format(t=col)
    return (f"CASE WHEN {c} <= {AM_OPEN} THEN {AM_OPEN + 1} "
            f"WHEN {c} > {AM_CLOSE} AND {c} <= {PM_OPEN} THEN {PM_OPEN + 1} "
            f"WHEN {c} > {CONT_END} THEN {CLOSE} ELSE {c} END")


def _continuous(col: str = "time") -> str:
    """Exchange time strictly inside continuous trading (for snapshots).

    Open at 14:57:00 itself: SZSE's 14:57:00 snapshot is already in close-auction mode (its
    resting-volume totals read 0).
    """
    m = f"((({col} + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE)"
    return f"(({m} > {AM_OPEN} AND {m} <= {AM_CLOSE}) OR ({m} > {PM_OPEN} AND {m} < {CONT_END}))"


def _pairs(df: pl.DataFrame, pairs: dict[str, tuple[str, str]]) -> pl.DataFrame:
    """Add ``<pair>@imb|diff|share`` columns for each (x, y) pair."""
    # DuckDB sums arrive as Decimal, where x/0 raises even inside a guarded branch
    df = df.with_columns(pl.col(c).cast(pl.Float64) for c in {v for p in pairs.values() for v in p})
    exprs = []
    for name, (x, y) in pairs.items():
        s = pl.col(x) + pl.col(y)
        exprs += [
            pl.when(s != 0).then((pl.col(x) - pl.col(y)) / s).alias(f"{name}@imb"),
            (pl.col(x) - pl.col(y)).alias(f"{name}@diff"),
            pl.when(s != 0).then(pl.col(x) / s).alias(f"{name}@share"),
        ]
    return df.with_columns(exprs)


def _long(df: pl.DataFrame, cand: str, date: str, keep: list[str]) -> pl.DataFrame:
    """Wide (sid, minute, V...) -> the long output layout."""
    vals = [c for c in df.columns if c not in ("symbol_id", "minute")]
    ren = {c: c if "@" in c else f"{c}@raw" for c in vals if c.split("@")[0] in keep}
    return (
        df.select("symbol_id", "minute", *ren)
        .rename(ren)
        .unpivot(index=["symbol_id", "minute"], variable_name="V", value_name="value")
        .select(
            pl.lit(int(date)).alias("date"),
            pl.col("symbol_id").cast(pl.Float64).alias("sid"),
            pl.col("minute").cast(pl.Int16),
            pl.lit(cand).alias("cand"),
            "V",
            pl.col("value").cast(pl.Float64),
        )
    )


# --------------------------------------------------------------------------- trades (VC, UG)
def _trades_view(con: duckdb.DuckDBPyConnection, date: str, thr_variant: str) -> None:
    """``tr``: universe trades with minute label, aggressor order id and that order's size."""
    size = ("sum(volume) OVER (PARTITION BY symbol_id, market_id, agg_id)"
            if thr_variant.startswith("order") else "volume")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE tr AS
        WITH t AS (
          SELECT r.symbol_id, r.market_id, r.dir, r.volume, r.price, r.time, r.seq_id,
                 CASE r.dir WHEN {BUY} THEN r.buy_seq_id ELSE r.sell_seq_id END AS agg_id
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
          WHERE r.date = {int(date)} AND r.trade_type = 1 AND r.dir IN ({BUY}, {SELL}))
        SELECT symbol_id, market_id, dir, volume, price, agg_id, {_label()} AS minute,
               {size} AS osize, time, seq_id
        FROM t""")


def zzvc(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZVC candidates (see module docstring)."""
    b, s = BUY, SELL
    base = con.sql(f"""
        SELECT symbol_id, minute,
          sum(volume) FILTER (dir = {b}) AS a, sum(volume) FILTER (dir = {s}) AS b,
          count(*) FILTER (dir = {b}) AS c_cnt, count(*) FILTER (dir = {s}) AS d_cnt,
          sum(volume * price) FILTER (dir = {b}) AS c_amt, sum(volume * price) FILTER (dir = {s}) AS d_amt,
          count(DISTINCT agg_id) FILTER (dir = {b}) AS c_ord, count(DISTINCT agg_id) FILTER (dir = {s}) AS d_ord,
          sum(volume) FILTER (dir = {b} AND NOT big) AS a_small, sum(volume) FILTER (dir = {s} AND NOT big) AS b_small,
          sum(volume) FILTER (dir = {b} AND big) AS c_big, sum(volume) FILTER (dir = {s} AND big) AS d_big
        FROM (SELECT tr.*, osize >= CASE dir WHEN {b} THEN th."ab.q90" ELSE th."as.q90" END AS big
              FROM tr JOIN thr th ON th.sid = tr.symbol_id)
        GROUP BY ALL""").pl().fill_null(0)
    flow = con.sql(f"""
        WITH adds AS (
          SELECT o.symbol_id, {_label()} AS minute, o.dir, o.volume
          FROM raw_order o JOIN uni u USING (symbol_id, market_id)
          WHERE o.date = {int(date)} AND o.update_type = 1 AND o.dir IN ({b}, {s})),
        cx_sh AS (
          SELECT o.symbol_id, {_label()} AS minute, o.dir, o.volume
          FROM raw_order o JOIN uni u USING (symbol_id, market_id)
          WHERE o.date = {int(date)} AND o.market_id = {SSE} AND o.update_type = 2),
        cx_sz AS (
          SELECT r.symbol_id, {_label()} AS minute,
                 CASE WHEN r.buy_seq_id > 0 THEN {b} ELSE {s} END AS dir, r.volume
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
          WHERE r.date = {int(date)} AND r.market_id = {SZSE} AND r.trade_type = 2)
        SELECT symbol_id, minute,
          sum(volume) FILTER (k = 0 AND dir = {b}) AS a, sum(volume) FILTER (k = 0 AND dir = {s}) AS b,
          sum(volume) FILTER (k = 1 AND dir = {b}) AS c, sum(volume) FILTER (k = 1 AND dir = {s}) AS d
        FROM (SELECT *, 0 AS k FROM adds UNION ALL SELECT *, 1 FROM cx_sh UNION ALL SELECT *, 1 FROM cx_sz)
        GROUP BY ALL""").pl().fill_null(0)
    keep = ["a", "b", "c", "d", "ab", "cd"]
    out = []
    for cand, (c, d) in {"cnt": ("c_cnt", "d_cnt"), "amt": ("c_amt", "d_amt"),
                         "ord": ("c_ord", "d_ord")}.items():
        w = base.select("symbol_id", "minute", "a", "b", pl.col(c).alias("c"), pl.col(d).alias("d"))
        out.append(_long(_pairs(w, {"ab": ("a", "b"), "cd": ("c", "d")}), cand, date, keep))
    w = base.select("symbol_id", "minute", pl.col("a_small").alias("a"), pl.col("b_small").alias("b"),
                    pl.col("c_big").alias("c"), pl.col("d_big").alias("d"))
    out.append(_long(_pairs(w, {"ab": ("a", "b"), "cd": ("c", "d")}), "size", date, keep))
    out.append(_long(_pairs(flow, {"ab": ("a", "b"), "cd": ("c", "d")}), "flow", date, keep))
    return pl.concat(out)


UG_TIERS = {  # tier -> (lower quantile or None, upper quantile or None); None = unbounded
    "t50_90": (("", "q50"), ("q50", "q90"), ("q90", "")),
    "t90_98": (("", "q90"), ("q90", "q98"), ("q98", "")),
    "t90_95_98": (("q90", "q95"), ("q95", "q98"), ("q98", "")),
    "c90_95_98": (("q90", ""), ("q95", ""), ("q98", "")),
}


def zzug(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZUG candidates (see module docstring)."""
    out = []
    for cand, tiers in UG_TIERS.items():
        cols = []
        for (lo, hi), (x, y) in zip(tiers, (("a", "b"), ("c", "d"), ("e", "f")), strict=True):
            for side, name, role in ((BUY, x, "ab"), (SELL, y, "as")):
                cond = [f"dir = {side}"]
                if lo:
                    cond.append(f'osize >= th."{role}.{lo}"')
                if hi:
                    cond.append(f'osize < th."{role}.{hi}"')
                cols.append(f"sum(volume) FILTER ({' AND '.join(cond)}) AS {name}")
        w = con.sql(f"""SELECT symbol_id, minute, {', '.join(cols)}
                        FROM tr JOIN thr th ON th.sid = tr.symbol_id GROUP BY ALL""").pl().fill_null(0)
        w = _pairs(w, {"ab": ("a", "b"), "cd": ("c", "d"), "ef": ("e", "f")})
        out.append(_long(w, cand, date, ["a", "b", "c", "d", "e", "f", "ab", "cd", "ef"]))
    return pl.concat(out)


# --------------------------------------------------------------------------- snapshots (QI, DS)
def _snap_view(con: duckdb.DuckDBPyConnection, date: str) -> None:
    """``sn``: continuous-session snapshots of the universe with the book fields we need."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE sn AS
        SELECT q.symbol_id, q.market_id, q.time, {_label()} AS minute,
               q.bid_px[1] AS bp1, q.ask_px[1] AS ap1, q.bid_vol[1] AS bv1, q.ask_vol[1] AS av1,
               list_sum(q.bid_vol[1:5]) AS bv5, list_sum(q.ask_vol[1:5]) AS av5,
               list_sum(q.bid_vol) AS bv10, list_sum(q.ask_vol) AS av10,
               q.total_bid_volume AS btot, q.total_ask_volume AS atot,
               CASE WHEN q.market_id = {SSE} THEN q.total_buy_no END AS bno,
               CASE WHEN q.market_id = {SSE} THEN q.total_sell_no END AS ano,
               CASE WHEN q.market_id = {SSE} THEN q.num_buy_levels END AS blv,
               CASE WHEN q.market_id = {SSE} THEN q.num_sell_levels END AS alv
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id)
        WHERE q.date = {int(date)} AND {_continuous('q.time')}""")


def zzqi(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZQI candidates (see module docstring)."""
    agg = con.sql("""
        SELECT symbol_id, minute,
          arg_max(bv1, time) AS a_last, arg_max(av1, time) AS b_last,
          avg(bv1) AS a_mean, avg(av1) AS b_mean,
          arg_max(bv5, time) AS c5_last, arg_max(av5, time) AS d5_last,
          avg(bv5) AS c5_mean, avg(av5) AS d5_mean,
          arg_max(bv10, time) AS c10_last, arg_max(av10, time) AS d10_last,
          arg_max(btot, time) AS ctot_last, arg_max(atot, time) AS dtot_last,
          arg_max(bp1, time) AS bp, arg_max(ap1, time) AS ap,
          avg(ap1 - bp1) FILTER (bp1 > 0 AND ap1 > 0) AS spr_mean
        FROM sn GROUP BY ALL""").pl()
    agg = agg.sort("symbol_id", "minute").with_columns(
        pl.when((pl.col("bp") > 0) & (pl.col("ap") > 0)).then((pl.col("bp") + pl.col("ap")) / 2).alias("mid"))
    agg = agg.with_columns(
        (pl.col("ap") - pl.col("bp")).alias("s@abs"),
        ((pl.col("ap") - pl.col("bp")) / pl.col("mid")).alias("s@rel"),
        ((pl.col("ap") - pl.col("bp")) / 0.01).round(6).alias("s@ticks"),
        pl.col("spr_mean").alias("s@absmean"),
        pl.col("mid").alias("m@mid"),
        (pl.col("mid").log() - pl.col("mid").log().shift(1).over("symbol_id")).alias("m@ret"),
        (((pl.col("bp") * pl.col("b_last").cast(pl.Float64) + pl.col("ap") * pl.col("a_last").cast(pl.Float64))
          / (pl.col("a_last") + pl.col("b_last")).cast(pl.Float64) - pl.col("mid")) / pl.col("mid")).alias("m@micro"),
    )
    sm = ["s@abs", "s@rel", "s@ticks", "s@absmean", "m@mid", "m@ret", "m@micro"]
    specs = {
        "last5": ("a_last", "b_last", "c5_last", "d5_last"),
        "mean5": ("a_mean", "b_mean", "c5_mean", "d5_mean"),
        "last10": ("a_last", "b_last", "c10_last", "d10_last"),
        "lasttot": ("a_last", "b_last", "ctot_last", "dtot_last"),
    }
    out = []
    for cand, (a, b, c, d) in specs.items():
        w = agg.select("symbol_id", "minute", pl.col(a).alias("a"), pl.col(b).alias("b"),
                       pl.col(c).alias("c"), pl.col(d).alias("d"), *sm)
        w = _pairs(w, {"ab": ("a", "b"), "cd": ("c", "d")})
        out.append(_long(w, cand, date, ["a", "b", "c", "d", "ab", "cd", "s", "m"]))
    return pl.concat(out)


def zzds(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZDS candidates (see module docstring)."""
    snap = con.sql("""
        SELECT symbol_id,
          arg_max(atot, time) AS qa_c, arg_max(btot, time) AS qb_c,
          arg_max(ano, time) AS na_c, arg_max(bno, time) AS nb_c,
          avg(atot) AS qa_m, avg(btot) AS qb_m, avg(ano) AS na_m, avg(bno) AS nb_m,
          arg_max(av10, time) AS qa_l, arg_max(bv10, time) AS qb_l,
          arg_max(alv, time) AS na_l, arg_max(blv, time) AS nb_l
        FROM sn GROUP BY ALL""").pl()
    cutoff = ((pl.datetime(int(date[:4]), int(date[4:6]), int(date[6:]), 14, 57, 0))
              .dt.epoch("ns") - NS_LOCAL)
    t57 = pl.select(cutoff).item()
    mbo = con.sql(f"""
        WITH adds AS (
          SELECT o.symbol_id, o.market_id, o.dir, o.volume,
                 CASE WHEN o.market_id = {SSE} THEN o.order_id ELSE o.seq_id END AS oid
          FROM raw_order o JOIN uni u USING (symbol_id, market_id)
          WHERE o.date = {int(date)} AND o.update_type = 1 AND o.dir IN ({BUY}, {SELL})
            AND o.time < {t57} AND NOT (o.market_id = {SZSE} AND o.order_type = 1)),
        fills AS (  -- SSE adds are post-match remainders: only passive fills reduce them
          SELECT r.symbol_id, r.market_id, x.oid, sum(r.volume) AS v
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id),
               LATERAL (SELECT unnest(
                 CASE WHEN r.market_id = {SSE} AND r.dir = {BUY} THEN [r.sell_seq_id]
                      WHEN r.market_id = {SSE} AND r.dir = {SELL} THEN [r.buy_seq_id]
                      ELSE [r.buy_seq_id, r.sell_seq_id] END) AS oid) x  -- auction: both legs
          WHERE r.date = {int(date)} AND r.trade_type = 1 AND r.time < {t57}
          GROUP BY ALL),
        cx AS (
          SELECT o.symbol_id, o.market_id, o.order_id AS oid, sum(o.volume) AS v
          FROM raw_order o JOIN uni u USING (symbol_id, market_id)
          WHERE o.date = {int(date)} AND o.market_id = {SSE} AND o.update_type = 2 AND o.time < {t57}
          GROUP BY ALL
          UNION ALL
          SELECT r.symbol_id, r.market_id, greatest(r.buy_seq_id, r.sell_seq_id), sum(r.volume)
          FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
          WHERE r.date = {int(date)} AND r.market_id = {SZSE} AND r.trade_type = 2 AND r.time < {t57}
          GROUP BY ALL),
        rest AS (
          SELECT a.symbol_id, a.dir, a.volume - coalesce(f.v, 0) - coalesce(c.v, 0) AS remain
          FROM adds a
          LEFT JOIN fills f USING (symbol_id, market_id, oid)
          LEFT JOIN cx c USING (symbol_id, market_id, oid))
        SELECT symbol_id,
          sum(remain) FILTER (dir = {SELL} AND remain > 0) AS qa_r, sum(remain) FILTER (dir = {BUY} AND remain > 0) AS qb_r,
          count(*) FILTER (dir = {SELL} AND remain > 0) AS na_r, count(*) FILTER (dir = {BUY} AND remain > 0) AS nb_r
        FROM rest GROUP BY ALL""").pl()
    specs = {"snapclose": "c", "snapmean": "m", "lvl10close": "l"}
    out = []
    keep = ["qa", "qb", "na", "nb", "qab", "nab"]
    for cand, k in specs.items():
        w = snap.select("symbol_id", pl.lit(-1).alias("minute"),
                        *[pl.col(f"{v}_{k}").cast(pl.Float64).alias(v) for v in ("qa", "qb", "na", "nb")])
        out.append(_long(_pairs(w, {"qab": ("qa", "qb"), "nab": ("na", "nb")}), cand, date, keep))
    w = mbo.select("symbol_id", pl.lit(-1).alias("minute"),
                   *[pl.col(f"{v}_r").cast(pl.Float64).alias(v) for v in ("qa", "qb", "na", "nb")])
    out.append(_long(_pairs(w, {"qab": ("qa", "qb"), "nab": ("na", "nb")}), "mboclose", date, keep))
    return pl.concat(out)


FAMILIES = {"ZZVC": zzvc, "ZZUG": zzug, "ZZQI": zzqi, "ZZDS": zzds}


def thresholds(date: str, variant: str) -> tuple[pl.DataFrame, str]:
    """SamplerR of the latest day before ``date`` (same day as a flagged fallback)."""
    root = THR / f"variant={variant}"
    days = sorted(p.name.split("=")[1] for p in root.glob("date=*") if (p / "samplerR.parquet").exists())
    prev = [d for d in days if d < date]
    use = prev[-1] if prev else (date if date in days else None)
    if use is None:
        raise SystemExit(f"no samplerR ({variant}) for or before {date}: run research/pm_factors/sampler_r.py")
    if use == date:
        print(f"  WARNING {date}: no earlier samplerR, using the same day's thresholds (look-ahead; "
              "fine for testing the pipeline, not for the PM comparison)")
    return pl.read_parquet(root / f"date={use}" / "samplerR.parquet"), use


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", default="20260615,20260805")
    ap.add_argument("--families", default="all", help="comma-separated, or 'all'")
    ap.add_argument("--index", default="000300")
    ap.add_argument("--thr-variant", default="order_all", help="samplerR variant for size tiers")
    ap.add_argument("--memory", default="6GB")
    a = ap.parse_args()
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    con.execute(f"SET temp_directory='{DATA / 'duckdb_tmp'}'")
    con.register("uni", universe(a.index))
    from candidates_more import FAMILIES_MORE, NEEDS, _events_view

    table = {**FAMILIES, **FAMILIES_MORE}
    needs = {"ZZVC": {"tr", "ev"}, "ZZUG": {"tr"}, "ZZQI": {"sn"}, "ZZDS": {"sn"}, **NEEDS}
    fams = list(table) if a.families == "all" else [f.strip().upper() for f in a.families.split(",") if f.strip()]
    want = set().union(*(needs[f] for f in fams))
    for d in filter(None, a.dates.split(",")):
        thr, thr_day = thresholds(d, a.thr_variant)
        con.register("thr", thr)
        if "tr" in want:
            _trades_view(con, d, a.thr_variant)
        if "sn" in want:
            _snap_view(con, d)
        if "ev" in want:
            _events_view(con, d)
        for fam in fams:
            with ResourceMonitor("W3.pm_candidates", params={"date": d, "family": fam}) as mon:
                df = table[fam](con, d).with_columns(pl.lit(int(thr_day)).alias("thr_date"))
                mon.rows = df.height
            dst = OUT / f"family={fam}" / f"date={d}"
            dst.mkdir(parents=True, exist_ok=True)
            df.write_parquet(dst / "part.parquet")
            assert mon.stats is not None
            append_history(mon.stats, DATA / "bench" / "history.parquet")
            print(f"{d} {fam}: {df.height:,} rows, {df['sid'].n_unique()} stocks, "
                  f"{df['cand'].n_unique()} candidates x {df['V'].n_unique()} columns, "
                  f"{mon.stats.wall_s:.1f}s rss {mon.stats.rss_peak_mb:.0f}MB")


if __name__ == "__main__":
    main()
