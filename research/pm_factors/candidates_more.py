"""Stage 4, continued: candidate formulas for the other ten PM factor families.

Same output layout, naming (``<pm column>@<variant>``), pair forms and threshold rules as
:mod:`candidates` (which also runs these: ``candidates.py --families ZZHL,...``). Families and
the PM's column names (``V``) / slots (``I``):

=====  =====  ============================  ===============================================
family I      V                             candidates (``cand``)
=====  =====  ============================  ===============================================
ZZHL   236    q n bs bm bl                  q = volume, n = trade count; bs/bm/bl = active
                                            buy volume in small / mid / large tiers as a share
                                            of active buy volume (``sh_t50_90``,
                                            ``sh_t90_98``), of all volume (``tot_t50_90``), or
                                            raw (``raw_t50_90``)
ZZMS   241    ab a b cd c d s m             a/b = large (>= q90) active buy / sell, c/d = small
                                            (< q90 or < q50); s / m = small / main net flow as
                                            ``@net`` or ``@ratio`` (of bar volume); volume or
                                            amount (``v90``, ``v50_90``, ``amt90``)
ZZGW   236    n                             counts per bar: new orders (``adds``), cancels
                                            (``cxl``), trades (``trades``), aggressor orders
                                            (``aggord``)
ZZXC   48     q n                           q/n = volume / trades (``vol_cnt``), amount /
                                            trades (``amt_cnt``), volume / aggressor orders
                                            (``vol_ord``)
ZZTS   48     q                             average trade size (``avgtrade``), average
                                            aggressor order (``avgord``), 5-minute log return
                                            (``ret``), realised vol of 1-minute returns
                                            (``rv``), (high - low) / open (``hl``)
ZZQO   48     q                             level-1 order-flow imbalance, Cont et al.
                                            (``ofi``), active imbalance (``actimb``), mean
                                            level-1 queue imbalance (``qimb``), mid return
                                            (``midret``)
ZZWA   48     aq bq an bn                   ask / bid side volume (q) and count (n) of new
                                            orders (``adds``), cancels (``cxl``), active trades
                                            (``act``)
ZZAL   48     an bn da db                   an/bn = cancel (``cxl_*``) or new-order
                                            (``add_*``) counts, ask / bid; da/db = change over
                                            the bar of ask / bid depth: levels 1-5 (``*_d5``),
                                            1-10 (``*_d10``), whole book (``*_dtot``)
ZZSQ   1      ab a b                        daily buy / sell volume of new orders (``adds``,
                                            ``adds_cont`` = continuous only), cancels
                                            (``cxl``), active trades (``act``)
ZZCR   1      o h l c ms msr mb vr          OHLC as price (``@px``), vs previous close
                                            (``@rel``), log (``@log``); ms/mb = large active
                                            sell / buy (q90 or q95, volume or amount), msr =
                                            (mb - ms) / total; vr = day volume (``@vol``) and
                                            vs the previous session (``@prev``, when its bars
                                            exist)
=====  =====  ============================  ===============================================

5-minute bars are right-labelled like the 1-minute ones (09:35 ... 11:30, 13:05 ... 15:00; the
15:00 bar holds 14:55-14:57 plus the close auction).
"""

from __future__ import annotations

import duckdb
import polars as pl
from candidates import BUY, SELL, SSE, SZSE, _label, _long, _pairs

M5 = "((minute + 4) // 5) * 5"  # right-labelled 5-minute bar of a 1-minute label


def _events_view(con: duckdb.DuckDBPyConnection, date: str) -> None:
    """``ev``: universe order flow -- new orders (k 0) and cancels (k 1) with side and minute."""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ev AS
        SELECT o.symbol_id, {_label('o.time')} AS minute, 0 AS k, o.dir, o.volume,
               ({_label('o.time')} BETWEEN 571 AND 897) AS cont
        FROM raw_order o JOIN uni u USING (symbol_id, market_id)
        WHERE o.date = {int(date)} AND o.update_type = 1 AND o.dir IN ({BUY}, {SELL})
        UNION ALL
        SELECT o.symbol_id, {_label('o.time')}, 1, o.dir, o.volume, TRUE
        FROM raw_order o JOIN uni u USING (symbol_id, market_id)
        WHERE o.date = {int(date)} AND o.market_id = {SSE} AND o.update_type = 2
        UNION ALL
        SELECT r.symbol_id, {_label('r.time')}, 1, CASE WHEN r.buy_seq_id > 0 THEN {BUY} ELSE {SELL} END,
               r.volume, TRUE
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date = {int(date)} AND r.market_id = {SZSE} AND r.trade_type = 2""")


def _q(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    return con.sql(sql).pl().fill_null(0)


def _shares(w: pl.DataFrame, cols: list[str], total: str) -> pl.DataFrame:
    return w.with_columns(
        pl.when(pl.col(total) > 0).then(pl.col(c).cast(pl.Float64) / pl.col(total)).alias(c) for c in cols)


# --------------------------------------------------------------------------- 1-minute families
def zzhl(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZHL candidates (module docstring)."""
    out = []
    for tiers, (lo, hi) in {"t50_90": ("q50", "q90"), "t90_98": ("q90", "q98")}.items():
        w = _q(con, f"""
            SELECT symbol_id, minute, sum(volume) AS q, count(*) AS n,
              sum(volume) FILTER (dir = {BUY}) AS buy,
              sum(volume) FILTER (dir = {BUY} AND osize < th."ab.{lo}") AS bs,
              sum(volume) FILTER (dir = {BUY} AND osize >= th."ab.{lo}" AND osize < th."ab.{hi}") AS bm,
              sum(volume) FILTER (dir = {BUY} AND osize >= th."ab.{hi}") AS bl
            FROM tr JOIN thr th ON th.sid = tr.symbol_id GROUP BY ALL""")
        keep = ["q", "n", "bs", "bm", "bl"]
        out.append(_long(_shares(w, ["bs", "bm", "bl"], "buy"), f"sh_{tiers}", date, keep))
        if tiers == "t50_90":
            out.append(_long(_shares(w, ["bs", "bm", "bl"], "q"), "tot_t50_90", date, keep))
            out.append(_long(w, "raw_t50_90", date, keep))
    return pl.concat(out)


def zzms(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZMS candidates (module docstring)."""
    out = []
    for cand, (small_q, val) in {"v90": ("q90", "volume"), "v50_90": ("q50", "volume"),
                                 "amt90": ("q90", "volume * price")}.items():
        w = _q(con, f"""
            SELECT symbol_id, minute, sum({val}) AS tot,
              sum({val}) FILTER (dir = {BUY} AND osize >= th."ab.q90") AS a,
              sum({val}) FILTER (dir = {SELL} AND osize >= th."as.q90") AS b,
              sum({val}) FILTER (dir = {BUY} AND osize < th."ab.{small_q}") AS c,
              sum({val}) FILTER (dir = {SELL} AND osize < th."as.{small_q}") AS d
            FROM tr JOIN thr th ON th.sid = tr.symbol_id GROUP BY ALL""")
        w = _pairs(w.with_columns(pl.col("tot").cast(pl.Float64)), {"ab": ("a", "b"), "cd": ("c", "d")})
        w = w.with_columns(
            (pl.col("c") - pl.col("d")).alias("s@net"), (pl.col("a") - pl.col("b")).alias("m@net"),
            pl.when(pl.col("tot") > 0).then((pl.col("c") - pl.col("d")) / pl.col("tot")).alias("s@ratio"),
            pl.when(pl.col("tot") > 0).then((pl.col("a") - pl.col("b")) / pl.col("tot")).alias("m@ratio"))
        out.append(_long(w, cand, date, ["a", "b", "c", "d", "ab", "cd", "s", "m"]))
    return pl.concat(out)


def zzgw(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZGW candidates (module docstring)."""
    fl = _q(con, "SELECT symbol_id, minute, count(*) FILTER (k = 0) AS adds, count(*) FILTER (k = 1) AS cxl "
                 "FROM ev GROUP BY ALL")
    tr = _q(con, "SELECT symbol_id, minute, count(*) AS trades, count(DISTINCT agg_id) AS aggord "
                 "FROM tr GROUP BY ALL")
    w = fl.join(tr, on=["symbol_id", "minute"], how="full", coalesce=True).fill_null(0)
    return pl.concat([_long(w.select("symbol_id", "minute", pl.col(c).alias("n")), c, date, ["n"])
                      for c in ("adds", "cxl", "trades", "aggord")])


# --------------------------------------------------------------------------- 5-minute families
def zzxc(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZXC candidates (module docstring)."""
    w = _q(con, f"""SELECT symbol_id, {M5} AS minute, sum(volume) AS vol, sum(volume * price) AS amt,
                          count(*) AS cnt, count(DISTINCT agg_id) AS ord
                   FROM tr GROUP BY ALL""")
    spec = {"vol_cnt": ("vol", "cnt"), "amt_cnt": ("amt", "cnt"), "vol_ord": ("vol", "ord")}
    return pl.concat([_long(w.select("symbol_id", "minute", pl.col(q).alias("q"), pl.col(n).alias("n")),
                            cand, date, ["q", "n"]) for cand, (q, n) in spec.items()])


def _bars1(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """1-minute open/high/low/close from the trades (volume-free bars skipped)."""
    return con.sql("""
        SELECT symbol_id, minute, arg_min(price, rn) AS o, max(price) AS h, min(price) AS l,
               arg_max(price, rn) AS c
        FROM (SELECT *, row_number() OVER (PARTITION BY symbol_id ORDER BY time, seq_id) AS rn FROM tr)
        GROUP BY ALL""").pl()


def zzts(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZTS candidates (module docstring)."""
    agg = _q(con, f"""SELECT symbol_id, {M5} AS minute, sum(volume)::DOUBLE / count(*) AS avgtrade,
                             sum(volume)::DOUBLE / count(DISTINCT agg_id) AS avgord
                      FROM tr GROUP BY ALL""")
    b1 = _bars1(con).sort("symbol_id", "minute").with_columns(
        (pl.col("c").log() - pl.col("c").log().shift(1).over("symbol_id")).alias("r1"),
        ((pl.col("minute") + 4) // 5 * 5).alias("m5"))
    b5 = b1.group_by("symbol_id", "m5").agg(
        pl.col("o").first(), pl.col("h").max(), pl.col("l").min(), pl.col("c").last(),
        (pl.col("r1").drop_nulls() ** 2).sum().sqrt().alias("rv")).rename({"m5": "minute"}).sort("symbol_id", "minute")
    b5 = b5.with_columns((pl.col("c").log() - pl.col("c").log().shift(1).over("symbol_id")).alias("ret"),
                         ((pl.col("h") - pl.col("l")) / pl.col("o")).alias("hl"))
    w = agg.join(b5, on=["symbol_id", "minute"], how="full", coalesce=True)
    return pl.concat([_long(w.select("symbol_id", "minute", pl.col(c).alias("q")), c, date, ["q"])
                      for c in ("avgtrade", "avgord", "ret", "rv", "hl")])


def zzqo(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZQO candidates (module docstring)."""
    s = con.sql("SELECT symbol_id, time, minute, bp1, ap1, bv1, av1 FROM sn ORDER BY symbol_id, time").pl()
    s = s.with_columns(
        pl.col("bp1").shift(1).over("symbol_id").alias("bp0"), pl.col("ap1").shift(1).over("symbol_id").alias("ap0"),
        pl.col("bv1").shift(1).over("symbol_id").alias("bv0"), pl.col("av1").shift(1).over("symbol_id").alias("av0"),
        ((pl.col("minute") + 4) // 5 * 5).alias("m5"))
    e = (pl.when(pl.col("bp1") >= pl.col("bp0")).then(pl.col("bv1")).otherwise(0)
         - pl.when(pl.col("bp1") <= pl.col("bp0")).then(pl.col("bv0")).otherwise(0)
         - pl.when(pl.col("ap1") <= pl.col("ap0")).then(pl.col("av1")).otherwise(0)
         + pl.when(pl.col("ap1") >= pl.col("ap0")).then(pl.col("av0")).otherwise(0))
    mid = pl.when((pl.col("bp1") > 0) & (pl.col("ap1") > 0)).then((pl.col("bp1") + pl.col("ap1")) / 2)
    q5 = (s.with_columns(e.cast(pl.Float64).alias("e"), mid.alias("mid"),
                         ((pl.col("bv1") - pl.col("av1")) / (pl.col("bv1") + pl.col("av1"))).alias("qi"))
          .group_by("symbol_id", "m5").agg(pl.col("e").sum().alias("ofi"), pl.col("qi").mean().alias("qimb"),
                                           pl.col("mid").drop_nulls().last().alias("mid"))
          .rename({"m5": "minute"}).sort("symbol_id", "minute")
          .with_columns((pl.col("mid").log() - pl.col("mid").log().shift(1).over("symbol_id")).alias("midret")))
    act = _q(con, f"""SELECT symbol_id, {M5} AS minute,
                        (sum(volume) FILTER (dir = {BUY}) - sum(volume) FILTER (dir = {SELL}))::DOUBLE
                          / sum(volume) AS actimb
                      FROM tr GROUP BY ALL""")
    w = q5.join(act, on=["symbol_id", "minute"], how="full", coalesce=True)
    return pl.concat([_long(w.select("symbol_id", "minute", pl.col(c).alias("q")), c, date, ["q"])
                      for c in ("ofi", "actimb", "qimb", "midret")])


def zzwa(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZWA candidates (module docstring)."""
    def side(src: str, k: str) -> str:
        return (f"""SELECT symbol_id, {M5} AS minute,
                  sum(volume) FILTER (dir = {SELL}) AS aq, sum(volume) FILTER (dir = {BUY}) AS bq,
                  count(*) FILTER (dir = {SELL}) AS an, count(*) FILTER (dir = {BUY}) AS bn
                FROM {src} {k} GROUP BY ALL""")
    spec = {"adds": side("ev", "WHERE k = 0"), "cxl": side("ev", "WHERE k = 1"), "act": side("tr", "")}
    return pl.concat([_long(_q(con, sql), cand, date, ["aq", "bq", "an", "bn"]) for cand, sql in spec.items()])


def zzal(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZAL candidates (module docstring)."""
    depth = con.sql(f"""
        SELECT symbol_id, {M5} AS minute,
          arg_max(av5, time) - arg_min(av5, time) AS da_d5, arg_max(bv5, time) - arg_min(bv5, time) AS db_d5,
          arg_max(av10, time) - arg_min(av10, time) AS da_d10, arg_max(bv10, time) - arg_min(bv10, time) AS db_d10,
          arg_max(atot, time) - arg_min(atot, time) AS da_dtot, arg_max(btot, time) - arg_min(btot, time) AS db_dtot
        FROM sn GROUP BY ALL""").pl()
    cnt = _q(con, f"""SELECT symbol_id, {M5} AS minute,
                        count(*) FILTER (k = 1 AND dir = {SELL}) AS an_cxl, count(*) FILTER (k = 1 AND dir = {BUY}) AS bn_cxl,
                        count(*) FILTER (k = 0 AND dir = {SELL}) AS an_add, count(*) FILTER (k = 0 AND dir = {BUY}) AS bn_add
                      FROM ev GROUP BY ALL""")
    w = cnt.join(depth, on=["symbol_id", "minute"], how="full", coalesce=True)
    out = []
    for n in ("cxl", "add"):
        for d in ("d5", "d10", "dtot"):
            out.append(_long(w.select("symbol_id", "minute", pl.col(f"an_{n}").alias("an"), pl.col(f"bn_{n}").alias("bn"),
                                      pl.col(f"da_{d}").alias("da"), pl.col(f"db_{d}").alias("db")),
                             f"{n}_{d}", date, ["an", "bn", "da", "db"]))
    return pl.concat(out)


# --------------------------------------------------------------------------- daily families
def zzsq(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZSQ candidates (module docstring)."""
    def day(src: str, where: str) -> pl.DataFrame:
        w = _q(con, f"""SELECT symbol_id, -1 AS minute, sum(volume) FILTER (dir = {BUY}) AS a,
                               sum(volume) FILTER (dir = {SELL}) AS b FROM {src} {where} GROUP BY ALL""")
        return _pairs(w, {"ab": ("a", "b")})
    spec = {"adds": day("ev", "WHERE k = 0"), "adds_cont": day("ev", "WHERE k = 0 AND cont"),
            "cxl": day("ev", "WHERE k = 1"), "act": day("tr", "")}
    return pl.concat([_long(w, cand, date, ["a", "b", "ab"]) for cand, w in spec.items()])


def zzcr(con: duckdb.DuckDBPyConnection, date: str) -> pl.DataFrame:
    """ZZCR candidates (module docstring)."""
    px = con.sql(f"""
        SELECT q.symbol_id, arg_max(q.open, q.time) AS o, arg_max(q.high, q.time) AS h,
               arg_max(q.low, q.time) AS l, arg_max(q.price, q.time) AS c, arg_max(q.pre_close, q.time) AS pc
        FROM raw_quotation q JOIN uni u USING (symbol_id, market_id)
        WHERE q.date = {int(date)} AND q.price > 0 GROUP BY ALL""").pl()
    px = px.with_columns(
        *[pl.col(k).alias(f"{k}@px") for k in "ohlc"],
        *[(pl.col(k) / pl.col("pc") - 1).alias(f"{k}@rel") for k in "ohlc"],
        *[(pl.col(k) / pl.col("pc")).log().alias(f"{k}@log") for k in "ohlc"])
    prev = con.sql(f"""SELECT CAST(symbol AS INTEGER) AS symbol_id, sum(volume) AS pvol FROM bars_1min
                       WHERE date = (SELECT max(date) FROM bars_1min WHERE date < {int(date)}) GROUP BY ALL""").pl() \
        if _has_view(con, "bars_1min") else pl.DataFrame(schema={"symbol_id": pl.Int32, "pvol": pl.Int64})
    out = []
    for cand, (qn, val) in {"q90_vol": ("q90", "volume"), "q95_vol": ("q95", "volume"),
                            "q90_amt": ("q90", "volume * price")}.items():
        w = _q(con, f"""SELECT symbol_id, sum(volume) AS vol, sum({val}) AS tot,
                          sum({val}) FILTER (dir = {SELL} AND osize >= th."as.{qn}") AS ms,
                          sum({val}) FILTER (dir = {BUY} AND osize >= th."ab.{qn}") AS mb
                        FROM tr JOIN thr th ON th.sid = tr.symbol_id GROUP BY ALL""")
        w = (w.join(px, on="symbol_id", how="left")
             .join(prev.with_columns(pl.col("symbol_id").cast(w.schema["symbol_id"])), on="symbol_id", how="left")
             .with_columns(pl.lit(-1).alias("minute"),
                           ((pl.col("mb") - pl.col("ms")).cast(pl.Float64) / pl.col("tot")).alias("msr"),
                           pl.col("vol").cast(pl.Float64).alias("vr@vol"),
                           (pl.col("vol").cast(pl.Float64) / pl.col("pvol")).alias("vr@prev")))
        out.append(_long(w.drop("vol", "tot", "pc", "pvol", *list("ohlc")), cand, date,
                         ["o", "h", "l", "c", "ms", "msr", "mb", "vr"]))
    return pl.concat(out)


def _has_view(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(con.sql(f"SELECT count(*) FROM duckdb_views() WHERE view_name = '{name}'").fetchone()[0])


FAMILIES_MORE = {"ZZHL": zzhl, "ZZMS": zzms, "ZZGW": zzgw, "ZZXC": zzxc, "ZZTS": zzts, "ZZQO": zzqo,
                 "ZZWA": zzwa, "ZZAL": zzal, "ZZSQ": zzsq, "ZZCR": zzcr}
NEEDS = {  # temp tables each family reads
    "ZZHL": {"tr"}, "ZZMS": {"tr"}, "ZZGW": {"tr", "ev"}, "ZZXC": {"tr"}, "ZZTS": {"tr"},
    "ZZQO": {"tr", "sn"}, "ZZWA": {"tr", "ev"}, "ZZAL": {"ev", "sn"}, "ZZSQ": {"tr", "ev"}, "ZZCR": {"tr"},
}
