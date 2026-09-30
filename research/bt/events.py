"""Build per-stock market-by-order event arrays for the backtest from the decoded feitu streams.

One row per book event, in the exchange's own sequence order ``(channel_id, seq_id)`` -- never
by timestamp (timestamps tie at 10 ms on SZSE and would reorder same-instant events, changing
queue positions). Encoding (all int64):

=========  ==============================================================================
column     meaning
=========  ==============================================================================
kind       0 = add, 1 = cancel, 2 = trade
xts        exchange time, ns UTC
ats        arrival (capture) time, ns UTC -- the point-in-time clock
side       add/cancel: order side (+1 buy / -1 sell); trade: aggressor side (0 unknown)
tick       price in 0.01 CNY ticks (add: limit price; trade: trade price; cancel: 0)
qty        shares
id_a       add/cancel: order key; trade: buy order key
id_b       trade: sell order key; else 0
otype      add: 2 limit, 3 own-side best (SZSE); SSE adds are 2; SZSE market orders carry
           their recovered subtype tag ``MTAG_*`` (>= 11), see below
=========  ==============================================================================

SZSE market-order subtypes (pre-pass, mirrors ``sht-demo/prototype/simulation/
market_subtype_recover.hpp``: ``build_recover_map`` + ``classify_szse_core``). The feed merges
the four official market types into ``orderType == 1``; each order's own trading profile tells
them apart: filled volume, number of distinct fill prices, whether its remainder was later hit as
the *resting* side (``has_maker_fill``), and whether a cancel followed within
``AUTO_CANCEL_MS`` with no maker fill (the exchange's automatic cancel). Tags:

======  ===========  ========================================  ==============================
tag     name         profile                                   remainder in the replay
======  ===========  ========================================  ==============================
11      cp_best      one price, partial, not auto-cancelled    rests at its fill price
12      fak          partial, auto-cancelled                   never rests (cancel follows)
13      sweep_full   fully filled across >= 2 prices           none
14      full_ambig   fully filled at one price                 none
15      fok_zero     zero fill, auto-cancelled in full         never rests
16      unfilled     zero fill, not auto-cancelled             rests if its price is learned
17      ambiguous    >= 2 prices, remainder not auto-cancelled rests at its fill price
======  ===========  ========================================  ==============================

SZSE trade aggressor: the leg with the **larger** sequence number (it arrived later) -- the
prototype's rule, not the vendor ``dir`` field. SSE trades keep the exchange's own BS flag, and
SSE orders are replayed exactly as published (their records carry nothing to deduce).

Venue differences, as measured in ``research/mbo_book.py``: the order key is ``order_id`` on SSE
and ``seq_id`` on SZSE; SSE cancels ride the order stream (``update_type == 2``), SZSE cancels
the trade stream (``trade_type == 2``, the cancelled order is whichever of buy/sell id is set).
SSE publishes a marketable order only *after* matching (its remainder), so its trades reference
an aggressor that is never added -- harmless, the book only needs the passive side.

Output: ``/work/crucible_data/store/bt_events/date=<D>/events.parquet`` (plus ``limits``), sorted
by symbol then sequence, for the chosen universe only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA = Path("/work/crucible_data")
RAW = DATA / "feitu_raw"
OUT = DATA / "store" / "bt_events"
SSE, SZSE = 3553, 3554
BUY, SELL = 1063, 1550
#: bump when the encoding changes: cached event files with another version are rebuilt
EVENTS_VERSION = 2
#: cancel latency at or below which a cancel counts as the exchange's automatic one (prototype
#: default for capnp dumps; its TDF config uses 1 ms because TDF stamps milliseconds)
AUTO_CANCEL_MS = 100
MTAG_CP_BEST, MTAG_FAK, MTAG_SWEEP_FULL, MTAG_FULL_AMBIG = 11, 12, 13, 14
MTAG_FOK_ZERO, MTAG_UNFILLED, MTAG_AMBIGUOUS = 15, 16, 17
MTAG_NAMES = {11: "cp_best", 12: "fak", 13: "sweep_full", 14: "full_ambig", 15: "fok_zero",
              16: "unfilled", 17: "ambiguous"}


def szse_market_tags(ev: pl.DataFrame, auto_ms: int = AUTO_CANCEL_MS) -> pl.DataFrame:
    """Subtype tag per SZSE market order: (symbol_id, market_id, id_a, tag).

    ``ev`` is the combined event frame; only SZSE market adds and the SZSE trades / cancels that
    reference them are used. One vectorised pass, no replay needed.
    """
    sz = ev.filter(pl.col("market_id") == SZSE)
    mkt = sz.filter((pl.col("kind") == 0) & (pl.col("otype") == 1)).select(
        "symbol_id", "market_id", pl.col("id_a").alias("key"), pl.col("qty").alias("order_vol"),
        pl.col("xts").alias("t_order"))
    if mkt.height == 0:
        return pl.DataFrame(schema={"symbol_id": pl.Int32, "market_id": pl.Int16,
                                    "id_a": pl.Int64, "tag": pl.Int64})
    tr = sz.filter(pl.col("kind") == 2)
    legs = pl.concat([
        tr.select("symbol_id", "market_id", pl.col("id_a").alias("key"), pl.col("id_b").alias("other"),
                  "qty", "tick"),
        tr.select("symbol_id", "market_id", pl.col("id_b").alias("key"), pl.col("id_a").alias("other"),
                  "qty", "tick"),
    ]).join(mkt.select("symbol_id", "market_id", "key"), on=["symbol_id", "market_id", "key"], how="semi")
    fills = legs.group_by("symbol_id", "market_id", "key").agg(
        pl.col("qty").sum().alias("filled"),
        pl.col("tick").n_unique().alias("distinct"),
        (pl.col("other") > pl.col("key")).any().alias("maker"),  # hit later by a larger seq = resting
    )
    cxl = sz.filter(pl.col("kind") == 1).group_by("symbol_id", "market_id", pl.col("id_a").alias("key")).agg(
        pl.col("xts").min().alias("t_cancel"))
    p = (mkt.join(fills, on=["symbol_id", "market_id", "key"], how="left")
         .join(cxl, on=["symbol_id", "market_id", "key"], how="left")
         .with_columns(pl.col("filled").fill_null(0), pl.col("distinct").fill_null(0),
                       pl.col("maker").fill_null(False)))
    auto = (pl.col("t_cancel").is_not_null() & ~pl.col("maker")
            & ((pl.col("t_cancel") - pl.col("t_order")) <= auto_ms * 1_000_000)
            & ((pl.col("t_cancel") - pl.col("t_order")) >= 0))
    tag = (
        pl.when(pl.col("filled") == 0).then(pl.when(auto).then(MTAG_FOK_ZERO).otherwise(MTAG_UNFILLED))
        .when(pl.col("filled") >= pl.col("order_vol"))
        .then(pl.when(pl.col("distinct") >= 2).then(MTAG_SWEEP_FULL).otherwise(MTAG_FULL_AMBIG))
        .when(pl.col("distinct") >= 2).then(pl.when(auto).then(MTAG_FAK).otherwise(MTAG_AMBIGUOUS))
        .otherwise(pl.when(auto).then(MTAG_FAK).otherwise(MTAG_CP_BEST))
    )
    return p.select("symbol_id", "market_id", pl.col("key").alias("id_a"), tag.cast(pl.Int64).alias("tag"))


def _side(col: str) -> pl.Expr:
    return pl.when(pl.col(col) == BUY).then(1).when(pl.col(col) == SELL).then(-1).otherwise(0)


def _tick(col: str) -> pl.Expr:
    return (pl.col(col) * 100).round(0).cast(pl.Int64)


def build(date: str, universe: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Events and per-stock price limits for ``universe`` (symbol_id, market_id) on ``date``."""
    keys = universe.select(pl.col("symbol_id").cast(pl.Int32), pl.col("market_id").cast(pl.Int16))
    orders = (
        pl.scan_parquet(RAW / "kind=order" / f"date={date}" / "part-*.parquet")
        .join(keys.lazy(), on=["symbol_id", "market_id"], how="semi")
        .filter(pl.col("update_type").is_in([1, 2]))
        .select(
            "symbol_id", "market_id", "channel_id", "seq_id",
            pl.when(pl.col("update_type") == 1).then(0).otherwise(1).alias("kind"),
            pl.col("time").alias("xts"), pl.col("spider_ts").alias("ats"),
            _side("dir").alias("side"),
            pl.when(pl.col("update_type") == 1).then(_tick("price")).otherwise(0).alias("tick"),
            pl.col("volume").alias("qty"),
            pl.when(pl.col("market_id") == SSE).then(pl.col("order_id")).otherwise(pl.col("seq_id"))
            .alias("id_a"),
            pl.lit(0, pl.Int64).alias("id_b"),
            pl.when(pl.col("market_id") == SSE).then(2).otherwise(pl.col("order_type").cast(pl.Int64))
            .alias("otype"),
        )
    )
    trades = (
        pl.scan_parquet(RAW / "kind=transaction" / f"date={date}" / "part-*.parquet")
        .join(keys.lazy(), on=["symbol_id", "market_id"], how="semi")
        .filter(pl.col("trade_type").is_in([1, 2]))
        .select(
            "symbol_id", "market_id", "channel_id", "seq_id",
            pl.when(pl.col("trade_type") == 1).then(2).otherwise(1).alias("kind"),
            pl.col("time").alias("xts"), pl.col("spider_ts").alias("ats"),
            pl.when(pl.col("trade_type") == 1).then(_side("dir"))
            .otherwise(pl.when(pl.col("buy_seq_id") > 0).then(1).otherwise(-1)).alias("side"),
            pl.when(pl.col("trade_type") == 1).then(_tick("price")).otherwise(0).alias("tick"),
            pl.col("volume").alias("qty"),
            pl.when(pl.col("trade_type") == 1).then(pl.col("buy_seq_id"))
            .otherwise(pl.max_horizontal("buy_seq_id", "sell_seq_id")).alias("id_a"),
            pl.when(pl.col("trade_type") == 1).then(pl.col("sell_seq_id")).otherwise(0).alias("id_b"),
            pl.lit(0, pl.Int64).alias("otype"),
        )
    )
    cast = {c: pl.Int64 for c in ("kind", "xts", "ats", "side", "tick", "qty", "id_a", "id_b", "otype", "seq_id")}
    ev = (
        pl.concat([orders.with_columns(**{k: pl.col(k).cast(v) for k, v in cast.items()}),
                   trades.with_columns(**{k: pl.col(k).cast(v) for k, v in cast.items()})])
        .sort("symbol_id", "market_id", "channel_id", "seq_id")
        .collect(engine="streaming")
    )
    # SZSE trade aggressor = the later-arriving leg (larger sequence number)
    is_sz_trade = (pl.col("market_id") == SZSE) & (pl.col("kind") == 2)
    ev = ev.with_columns(
        pl.when(is_sz_trade).then(pl.when(pl.col("id_a") > pl.col("id_b")).then(1).otherwise(-1))
        .otherwise(pl.col("side")).cast(pl.Int64).alias("side")
    )
    # SZSE market-order subtype tags ride in otype
    tags = szse_market_tags(ev)
    ev = (
        ev.join(tags.with_columns(pl.lit(0, pl.Int64).alias("kind")),
                on=["symbol_id", "market_id", "id_a", "kind"], how="left")
        .with_columns(pl.coalesce("tag", "otype").alias("otype"))
        .drop("tag")
    )
    limits = (
        pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
        .join(keys.lazy(), on=["symbol_id", "market_id"], how="semi")
        .filter(pl.col("high_limited") > 0)
        .group_by("symbol_id", "market_id")
        .agg(_tick("high_limited").max().alias("up_tick"), _tick("low_limited").min().alias("dn_tick"),
             _tick("pre_close").max().alias("pre_close_tick"))
        .collect(engine="streaming")
    )
    return ev, limits


def load(date: str, universe: pl.DataFrame, *, rebuild: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Cached :func:`build`: rebuilt when missing, forced, of another version, or the universe grew."""
    d = OUT / f"date={date}"
    ev_p, lim_p, ver_p = d / "events.parquet", d / "limits.parquet", d / "VERSION"
    current = ver_p.exists() and ver_p.read_text().strip() == str(EVENTS_VERSION)
    if ev_p.exists() and lim_p.exists() and current and not rebuild:
        lim = pl.read_parquet(lim_p)
        have = lim.select("symbol_id", "market_id")
        want = universe.select(pl.col("symbol_id").cast(pl.Int32), pl.col("market_id").cast(pl.Int16))
        if want.join(have, on=["symbol_id", "market_id"], how="anti").height == 0:
            ev = pl.read_parquet(ev_p).join(want, on=["symbol_id", "market_id"], how="semi")
            return ev, lim.join(want, on=["symbol_id", "market_id"], how="semi")
    ev, lim = build(date, universe)
    d.mkdir(parents=True, exist_ok=True)
    ev.write_parquet(ev_p, compression="zstd")
    lim.write_parquet(lim_p)
    ver_p.write_text(str(EVENTS_VERSION))
    return ev, lim
