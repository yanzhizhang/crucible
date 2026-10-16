"""Stage 2: rebuild the PM's ``1min_src`` fields from our raw data and score candidate definitions.

``1min_src/<D>.nc`` is one variable ``x(S, D, I, V)``: S = stock code as a bare number (1800
HS300 + ZZ500 + ZZ1000 names), I = 257 minute labels (09:15-11:30, 13:00-15:00, local time),
V = volumeTotal, dollarVolumeTotal, tradesTotal, lastTradePrice, askPrice, askSize, bidPrice,
bidSize, close, midPriceLastValid. Read on 97 for 000001 on 20260430: the totals are cumulative
and start at 09:31 (NaN before), are flat from 14:57 to 15:00 (close auction not in them);
ask / bid / mid are the book at the minute's end (0 during the close auction); close is the
minute's last trade (NaN without one). ``index/1min_src/<D>.nc`` has cumVol / cumTo /
lastTradePrice for six indices (S = 10000000 + code).

The open questions are settled by scoring, not by guessing: every field gets candidates on
three axes and each candidate is compared cell by cell with the PM cube:

* clock: exchange time or local arrival time (``exchange_ts`` / ``arrival_ts``);
* edge: a record exactly on the label belongs to that minute (``(L-1, L]``) or the next;
* scope (totals and close): all trades, or continuous-session trades only.

Book fields are the last snapshot at or before the label (as-of, carried forward).

    python research/compare_bars_pm.py --date 20260430

Output: ``data/reports/stage2/bars_<D>.csv`` (every field x candidate: cells, exact share,
share within 1e-4 relative, correlation) and the best candidate per field is printed.
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

RAW = Path("/work/crucible_data/feitu_raw")
PROD = Path("/data/prod")
warnings.filterwarnings("ignore", message="Sortedness of columns")
NS = 1_000_000_000
LOCAL = 8 * 3600
CONT = ((9 * 3600 + 30 * 60, 11 * 3600 + 30 * 60), (13 * 3600, 14 * 3600 + 57 * 60))  # (open, close] local sec


def pm_cube(path: Path) -> pl.DataFrame:
    """PM cube -> long frame (code, minute (sec of day), V, pm)."""
    ds = xr.open_dataset(path)
    da = ds["x"].isel(D=0).transpose("S", "I", "V")
    s = [str(v) for v in ds["S"].values]
    i = [int(t[:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8]) for t in (str(v) for v in ds["I"].values)]
    v = [str(x) for x in ds["V"].values]
    grid = np.array(list(itertools.product(s, i, v)), dtype=object)
    return pl.DataFrame({"code": grid[:, 0].astype(str), "label": grid[:, 1].astype(np.int64),
                         "V": grid[:, 2].astype(str), "pm": da.values.reshape(-1).astype(np.float64)})


def _sod(col: str) -> pl.Expr:
    return ((pl.col(col) // NS + LOCAL) % 86400).alias("sod")


def _label(edge: str) -> pl.Expr:
    """Right label of the minute holding second-of-day ``sod``: (L-1, L] or [L-1, L)."""
    ceil = ((pl.col("sod_ns") + 60 * NS - 1) // (60 * NS)) * 60
    floor_next = (pl.col("sod_ns") // (60 * NS) + 1) * 60
    return (ceil if edge == "incl" else floor_next).alias("label")


def _in_cont(sod: pl.Expr) -> pl.Expr:
    return ((sod > CONT[0][0]) & (sod <= CONT[0][1])) | ((sod > CONT[1][0]) & (sod <= CONT[1][1]))


def trade_candidates(date: str, labels: list[int]) -> pl.DataFrame:
    """Cumulative totals and the minute's last trade under every (clock, edge, scope)."""
    t = pl.scan_parquet(RAW / "kind=transaction" / f"date={date}" / "part-*.parquet").filter(
        pl.col("trade_type") == 1).select("symbol_id", "time", "spider_ts", "seq_id", "price", "volume")
    out = []
    for clock, edge, scope in itertools.product(("time", "spider_ts"), ("incl", "excl"), ("all", "cont")):
        tt = t.with_columns(pl.col("symbol_id").cast(pl.String).alias("code"),
                            ((pl.col(clock) + LOCAL * NS) % (86400 * NS)).alias("sod_ns"))
        tt = tt.with_columns(_label(edge), (pl.col("sod_ns") / NS).alias("sod"))
        if scope == "cont":
            tt = tt.filter(_in_cont(pl.col("sod")))
        per = (tt.group_by("code", "label").agg(
            pl.col("volume").sum().alias("vol"), (pl.col("volume") * pl.col("price")).sum().alias("amt"),
            pl.len().alias("n"), pl.col("price").sort_by(clock, "seq_id").last().alias("last"))
               .collect().sort("code", "label"))
        cum = per.with_columns(pl.col("vol").cum_sum().over("code").alias("volumeTotal"),
                               pl.col("amt").cum_sum().over("code").alias("dollarVolumeTotal"),
                               pl.col("n").cum_sum().over("code").alias("tradesTotal"))
        grid = cum.select("code").unique().join(pl.DataFrame({"label": labels}, schema={"label": pl.Int64}), how="cross").sort("label")
        g = grid.join_asof(cum.sort("label"), on="label", by="code", strategy="backward")
        g = g.join(per.select("code", "label", pl.col("last").alias("close")), on=["code", "label"], how="left")
        cand = f"{'exch' if clock == 'time' else 'local'}/{edge}/{scope}"
        for v in ("volumeTotal", "dollarVolumeTotal", "tradesTotal", "close"):
            out.append(g.select("code", "label", pl.lit(v).alias("V"), pl.lit(cand).alias("cand"),
                                pl.col(v).cast(pl.Float64).alias("ours")))
    return pl.concat(out)


def auction_totals(date: str) -> pl.DataFrame:
    """Opening-auction totals per stock: trades before 09:30:00 exchange time."""
    return (pl.scan_parquet(RAW / "kind=transaction" / f"date={date}" / "part-*.parquet")
            .filter((pl.col("trade_type") == 1) & ((pl.col("time") + LOCAL * NS) % (86400 * NS) < CONT[0][0] * NS))
            .group_by(pl.col("symbol_id").cast(pl.String).alias("code"))
            .agg(pl.col("volume").sum().alias("total_volume_auct"),
                 (pl.col("volume") * pl.col("price")).sum().alias("total_amount_auct"), pl.len().alias("total_no_auct"))
            .collect())


def book_candidates(date: str, labels: list[int]) -> pl.DataFrame:
    """Book at the minute's end (last snapshot at or before the label) under (clock, edge)."""
    q = pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
    pre = pl.col("pre_close") if "pre_close" in q.collect_schema().names() else pl.lit(None, pl.Float64).alias("pre_close")
    q = q.select(
        "symbol_id", "time", "spider_ts", "price", pre, "total_volume", "total_amount", "total_no",
        pl.col("ask_px").arr.get(0).alias("askPrice"),
        pl.col("ask_vol").arr.get(0).alias("askSize"), pl.col("bid_px").arr.get(0).alias("bidPrice"),
        pl.col("bid_vol").arr.get(0).alias("bidSize"))
    valid = (pl.col("askPrice") > 0) & (pl.col("bidPrice") > 0)
    q = q.with_columns(pl.when(valid).then((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("mid_valid"))
    auct = auction_totals(date)
    out = []
    # sess: 11:30 reads the last snapshot at or before 13:00:00, 15:00 the day's last snapshot (PM's session closes)
    variants = [*((c, e, False) for c, e in itertools.product(("time", "spider_ts"), ("incl", "excl"))), ("time", "incl", True)]
    for clock, edge, sess in variants:
        qq = q.with_columns(pl.col("symbol_id").cast(pl.String).alias("code"),
                            ((pl.col(clock) + LOCAL * NS) % (86400 * NS)).alias("sod_ns"))
        qq = qq.with_columns(_label(edge), (pl.col("sod_ns") / NS).alias("sod")).sort(clock)
        # snapshot cumulative totals right after the open auction (last snapshot at or before 09:30)
        base = qq.filter(pl.col("sod") <= CONT[0][0]).group_by("code").agg(
            pl.col("total_volume", "total_amount", "total_no").last().name.suffix("_open"))
        # PM's 11:30 value = its 13:00 value: last snapshot at or before 13:00:00 (SZ's 13:00:00 one holds the first pm trades)
        am = qq.filter(pl.col("sod") <= CONT[1][0]).group_by("code").agg(
            pl.col("total_volume", "total_amount", "total_no").last().name.suffix("_am"))
        qq = qq.group_by("code", "label").agg(
            pl.col("price", "pre_close", "askPrice", "askSize", "bidPrice", "bidSize", "total_volume", "total_amount",
                   "total_no").last(),
            pl.col("mid_valid").drop_nulls().last()).collect().sort("code", "label")
        qq = qq.join(base.collect(), on="code", how="left").join(auct, on="code", how="left").join(
            am.collect(), on="code", how="left").with_columns(
            ((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("mid_plain"),
            pl.when((pl.col("askPrice") > 0) & (pl.col("bidPrice") > 0))
            .then((pl.col("askPrice") + pl.col("bidPrice")) / 2).otherwise(0.0).alias("mid_or0"),
            pl.col("price").alias("lastTradePrice"),
            # one side empty (limit up / down): that side's price, or the last trade
            pl.when(pl.col("askPrice") > 0).then(
                pl.when(pl.col("bidPrice") > 0).then((pl.col("askPrice") + pl.col("bidPrice")) / 2).otherwise(pl.col("askPrice")))
            .when(pl.col("bidPrice") > 0).then(pl.col("bidPrice"))
            .otherwise((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("mid_side"),  # neither side: 0 in the close auction
            pl.when((pl.col("askPrice") > 0) & (pl.col("bidPrice") > 0))
            .then((pl.col("askPrice") + pl.col("bidPrice")) / 2).otherwise(pl.col("price")).alias("mid_last"))
        akey = pl.col("label")
        if sess:
            akey = pl.when(pl.col("label") == CONT[0][1]).then(CONT[1][0]).when(pl.col("label") == 15 * 3600).then(10**9).otherwise(akey)
        grid = (qq.select("code").unique().join(pl.DataFrame({"label": labels}, schema={"label": pl.Int64}), how="cross")
                .with_columns(akey.cast(pl.Int64).alias("akey")).sort("akey"))
        g = grid.join_asof(qq.drop("mid_valid").with_columns(pl.col("label").alias("slabel")).rename({"label": "akey"})
                           .sort("akey"), on="akey", by="code", strategy="backward")
        # last valid mid carried forward: as-of over the minutes that had one
        mv = qq.filter(pl.col("mid_valid").is_not_null()).select("code", pl.col("label").alias("akey"), "mid_valid").sort("akey")
        g = g.join_asof(mv, on="akey", by="code", strategy="backward")
        # no carry-forward: the book only counts when a snapshot fell inside the minute (15:00 = the day's last)
        fresh = (pl.col("slabel") == pl.col("akey")) | (pl.col("akey") == 10**9)
        book_cols = ("askPrice", "askSize", "bidPrice", "bidSize", "lastTradePrice", "mid_plain", "mid_side")
        g = g.with_columns(*[pl.when(fresh).then(pl.col(c)).alias(f"nc_{c}") for c in book_cols]).drop(
            "akey", "slabel").sort("code", "label")
        cont = _in_cont(pl.col("label").cast(pl.Float64))
        g = g.with_columns(
            *[pl.when(cont).then(pl.col(c) - pl.col(f"{c}_open"))
              .alias(f"snap_{c}") for c in ("total_volume", "total_amount", "total_no")],
            *[pl.when(cont).then(pl.col(c) - pl.col(f"{c}_auct").fill_null(0))
              .alias(f"auct_{c}") for c in ("total_volume", "total_amount", "total_no")],
            *[pl.when(~cont).then(None).when(pl.col("label") == CONT[0][1])
              .then(pl.col(f"{c}_am") - pl.col(f"{c}_auct").fill_null(0))
              .otherwise(pl.col(c) - pl.col(f"{c}_auct").fill_null(0)).alias(f"amc_{c}")
              for c in ("total_volume", "total_amount", "total_no")],
            # whole day: empty through 09:30, 13:00 as 11:30, after 14:57 held at 14:57 (close auction not counted)
            *[pl.when(pl.col("label") <= CONT[0][0]).then(None)
              .when(pl.col("label") > CONT[1][1])
              .then(pl.col(c).filter(pl.col("label") == CONT[1][1]).first().over("code") - pl.col(f"{c}_auct").fill_null(0))
              .otherwise(pl.col(c) - pl.col(f"{c}_auct").fill_null(0)).alias(f"full_{c}")
              for c in ("total_volume", "total_amount", "total_no")],
            pl.when(pl.col("total_volume").diff().over("code") > 0).then(pl.col("price")).alias("snap_close"))
        # PM close: last trade price carried forward through the day, empty up to 09:30
        g = g.with_columns(pl.when(pl.col("label") > CONT[0][0]).then(pl.col("snap_close").forward_fill().over("code"))
                           .alias("snap_close_ffill"))
        # a stock that never trades (suspended): fall back to the snapshot price / the pre-close
        g = g.with_columns(pl.when(pl.col("label") > CONT[0][0]).then(pl.coalesce("snap_close_ffill", "price"))
                           .alias("snap_close_ffill_px"),
                           pl.when(pl.col("label") > CONT[0][0]).then(pl.coalesce("snap_close_ffill", "pre_close"))
                           .alias("snap_close_ffill_pre"))
        tag = f"{'exch' if clock == 'time' else 'local'}/{edge}{'_sess' if sess else ''}"
        pairs = [(v, v, "book") for v in ("lastTradePrice", "askPrice", "askSize", "bidPrice", "bidSize")]
        pairs += [("midPriceLastValid", "mid_plain", "mid_plain"), ("midPriceLastValid", "mid_or0", "mid_or0"),
                  ("midPriceLastValid", "mid_valid", "mid_lastvalid"), ("midPriceLastValid", "mid_side", "mid_side"),
                  ("midPriceLastValid", "mid_last", "mid_last"),
                  ("volumeTotal", "snap_total_volume", "snapcum"), ("dollarVolumeTotal", "snap_total_amount", "snapcum"),
                  ("tradesTotal", "snap_total_no", "snapcum"), ("close", "snap_close", "snaplast"),
                  ("close", "snap_close_ffill", "snaplast_ffill"),
                  ("close", "snap_close_ffill_px", "snaplast_ffill_px"), ("close", "snap_close_ffill_pre", "snaplast_ffill_pre"),
                  ("volumeTotal", "auct_total_volume", "snap_ex_auction"),
                  ("dollarVolumeTotal", "auct_total_amount", "snap_ex_auction"),
                  ("tradesTotal", "auct_total_no", "snap_ex_auction"),
                  ("volumeTotal", "amc_total_volume", "snap_ex_auction_1300"),
                  ("dollarVolumeTotal", "amc_total_amount", "snap_ex_auction_1300"),
                  ("tradesTotal", "amc_total_no", "snap_ex_auction_1300"),
                  ("volumeTotal", "full_total_volume", "snap_ex_auction_day"),
                  ("dollarVolumeTotal", "full_total_amount", "snap_ex_auction_day"),
                  ("tradesTotal", "full_total_no", "snap_ex_auction_day")]
        if sess:
            pairs += [(v, f"nc_{v}", "book_nocarry") for v in ("lastTradePrice", "askPrice", "askSize", "bidPrice", "bidSize")]
            pairs += [("midPriceLastValid", "nc_mid_side", "mid_side_nocarry"),
                      ("midPriceLastValid", "nc_mid_plain", "mid_plain_nocarry")]
        for v, col, kind in pairs:
            out.append(g.select("code", "label", pl.lit(v).alias("V"), pl.lit(f"{tag}/{kind}").alias("cand"),
                                pl.col(col).cast(pl.Float64).alias("ours")))
    return pl.concat(out)


def index_candidates(date: str, labels: list[int]) -> pl.DataFrame:
    """Index candidates: last record at or before each label, cumulative and ex-opening-auction."""
    x = pl.scan_parquet(RAW / "kind=index" / f"date={date}" / "part-*.parquet").select(
        "symbol_id", "market_id", "time", "spider_ts", "total_volume", "total_amount", "last_price")
    out = []
    for clock, edge in itertools.product(("time", "spider_ts"), ("incl", "excl")):
        xx = x.with_columns((pl.col("symbol_id") + 10_000_000).cast(pl.String).alias("code"),
                            ((pl.col(clock) + LOCAL * NS) % (86400 * NS)).alias("sod_ns"))
        xx = xx.with_columns(_label(edge), (pl.col("sod_ns") / NS).alias("sod")).sort(clock)
        base = xx.filter(pl.col("sod") <= CONT[0][0]).group_by("code").agg(
            pl.col("total_volume", "total_amount").last().name.suffix("_open")).collect()
        xx = xx.group_by("code", "label").agg(
            pl.col("total_volume", "total_amount", "last_price").last()).collect().sort("code", "label")
        grid = xx.select("code").unique().join(pl.DataFrame({"label": labels}, schema={"label": pl.Int64}), how="cross").sort("label")
        g = grid.join_asof(xx.sort("label"), on="label", by="code", strategy="backward").join(base, on="code", how="left")
        cont = _in_cont(pl.col("label").cast(pl.Float64))
        g = g.with_columns(*[pl.when(cont).then(pl.col(c) - pl.col(f"{c}_open").fill_null(0)).alias(f"x_{c}")
                             for c in ("total_volume", "total_amount")])
        tag = f"{'exch' if clock == 'time' else 'local'}/{edge}"
        for v, c, kind in (("cumVol", "total_volume", "index"), ("cumTo", "total_amount", "index"),
                           ("cumVol", "x_total_volume", "index_ex_open"), ("cumTo", "x_total_amount", "index_ex_open"),
                           ("lastTradePrice", "last_price", "index")):
            out.append(g.select("code", "label", pl.lit(v).alias("V"), pl.lit(f"{tag}/{kind}").alias("cand"),
                                pl.col(c).cast(pl.Float64).alias("ours")))
    return pl.concat(out)


def _hm(label: pl.Expr) -> pl.Expr:
    return pl.format("{}:{}", (label // 3600).cast(pl.String).str.zfill(2), (label % 3600 // 60).cast(pl.String).str.zfill(2))


def diagnose(pm: pl.DataFrame, ours: pl.DataFrame, v: str, cand: str) -> None:
    """Where one candidate disagrees: empty-vs-value cells, then values by exchange, minute, shift and book context."""
    mine = ours.filter((pl.col("V") == v) & (pl.col("cand") == cand)).select("code", "label", "ours")
    j = pm.filter(pl.col("V") == v).join(mine, on=["code", "label"], how="inner").with_columns(
        _exch(pl.col("code")).alias("exch"), _hm(pl.col("label")).alias("t"))
    a, b = pl.col("pm"), pl.col("ours")
    pm_na, our_na = a.is_null() | a.is_nan(), b.is_null() | b.is_nan()
    eq = lambda x, y: (x - y).abs() <= 1e-9 * x.abs().clip(1.0, None)  # noqa: E731  relative 1e-9
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=6):
        na = j.filter(pm_na != our_na).with_columns(pl.when(pm_na).then(pl.lit("pm empty")).otherwise(pl.lit("ours empty")).alias("side"))
        print(f"== diagnose {v} / {cand}: {j.height} cells, {na.height} empty on one side only")
        if na.height:
            print(na.group_by("side", "exch", "t").len().sort("len", descending=True).head(12))
        j = j.filter(~pm_na & ~our_na).with_columns(eq(a, b).alias("ok"), (b - a).alias("ours_minus_pm"))
        print(f"both present: {j.height} cells, exact {j['ok'].mean():.4f}")
        print(j.group_by("exch").agg(pl.len(), pl.col("ok").mean()).sort("exch"))
        print(j.group_by((pl.col("label") // 3600).alias("hour")).agg(pl.len(), pl.col("ok").mean()).sort("hour"))
        bad = j.filter(~pl.col("ok"))
        print("stocks with any mismatch:", bad["code"].n_unique(), "of", j["code"].n_unique())
        print(bad.group_by("exch", "t").len().sort("len", descending=True).head(12))
        if bad.is_empty():
            return
        # is PM one minute ahead / behind us, or equal to one of our book fields at that label?
        ctx = bad.select("code", "label", "pm", "ours")
        for name, dl in (("ours one minute earlier", -60), ("ours one minute later", 60)):
            sh = mine.with_columns((pl.col("label") - dl).alias("label")).rename({"ours": "o2"})
            m = ctx.join(sh, on=["code", "label"], how="left")
            print(f"misses where pm == {name}: {m.select(eq(pl.col('pm'), pl.col('o2')).fill_null(False).mean()).item():.4f}")
        tag = "/".join(cand.split("/")[:2])
        for f in ("askPrice", "bidPrice", "lastTradePrice"):
            o = ours.filter((pl.col("V") == f) & (pl.col("cand") == f"{tag}/book")).select("code", "label", pl.col("ours").alias(f))
            ctx = ctx.join(o, on=["code", "label"], how="left")
        if "askPrice" in ctx.columns:
            ctx = ctx.with_columns(((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("our_mid"))
            print("misses where pm equals our:", {f: round(ctx.select(eq(pl.col("pm"), pl.col(f)).fill_null(False).mean()).item(), 4)
                                                  for f in ("askPrice", "bidPrice", "lastTradePrice", "our_mid")})
        print(bad.group_by("code").agg(pl.len().alias("bad_cells"), pl.col("ours_minus_pm").median().alias("median_diff"))
              .sort("bad_cells", descending=True).head(8))
        print(ctx.join(bad.select("code", "label", "t"), on=["code", "label"]).sort("code", "label")
              .drop("label").head(15))


DELTAS_MS = (-3000, -1000, -500, -200, -100, 0, 50, 100, 200, 300, 500, 1000, 1500, 2000, 3000, 5000, 10000)


def _cut_table(m: pl.DataFrame, name: str) -> pl.DataFrame:
    """Share of matched cells whose cutoff window [lo, hi) holds label + delta, per delta."""
    lo, hi = pl.col("lo_ms"), pl.col("hi_ms")
    return pl.DataFrame({"delta_ms": list(DELTAS_MS), name: [
        m.select(((lo <= d) & (hi.is_null() | (d < hi))).mean()).item() for d in DELTAS_MS]})


def implied_cut(pm: pl.DataFrame, date: str) -> None:
    """Where PM's volumeTotal cutoff lies: match each cell to the trade / snapshot whose running total equals it."""
    cells = pm.filter((pl.col("V") == "volumeTotal") & pl.col("pm").is_not_nan()).select(
        "code", "label", pl.col("pm").round(0).cast(pl.Int64).alias("cum"))
    xsod = ((pl.col("time") + LOCAL * NS) % (86400 * NS) / NS).alias("xsod")
    t = (pl.scan_parquet(RAW / "kind=transaction" / f"date={date}" / "part-*.parquet")
         .filter(pl.col("trade_type") == 1).select("symbol_id", "time", "spider_ts", "seq_id", "volume")
         .with_columns(pl.col("symbol_id").cast(pl.String).alias("code"), xsod)
         .filter(_in_cont(pl.col("xsod"))).collect())  # continuous session by exchange time
    q = (pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
         .select("symbol_id", "time", "spider_ts", pl.col("total_volume").cast(pl.Int64))
         .with_columns(pl.col("symbol_id").cast(pl.String).alias("code"), xsod).collect())
    base = q.filter(pl.col("xsod") <= CONT[0][0]).sort("time").group_by("code").agg(
        pl.col("total_volume").last().alias("base"))
    q = q.join(base, on="code", how="left").with_columns(
        (pl.col("total_volume") - pl.col("base").fill_null(0)).alias("cum"))
    n = cells.height
    tables, found = [], []
    for src, frame in (("trades", t), ("snap", q)):
        for clock in ("time", "spider_ts"):
            f = frame.sort("code", clock, *(["seq_id"] if src == "trades" else []))
            if src == "trades":
                f = f.with_columns(pl.col("volume").cum_sum().over("code").alias("cum"))
            f = f.with_columns(((pl.col(clock) + LOCAL * NS) % (86400 * NS)).alias("at")).with_columns(
                pl.col("at").shift(-1).over("code").alias("next_at"))
            m = (cells.join(f.select("code", "cum", "at", "next_at"), on=["code", "cum"], how="inner")
                 .group_by("code", "label").agg(pl.col("at").min(), pl.col("next_at").max())
                 .with_columns(((pl.col("at") - pl.col("label") * NS) / 1e6).alias("lo_ms"),
                               ((pl.col("next_at") - pl.col("label") * NS) / 1e6).alias("hi_ms")))
            name = f"{src}/{'exch' if clock == 'time' else 'local'}"
            found.append({"source": name, "matched": m.height / n,
                          "lo_ms_p10": m["lo_ms"].quantile(0.1), "lo_ms_p50": m["lo_ms"].median(),
                          "lo_ms_p90": m["lo_ms"].quantile(0.9), "hi_ms_p50": m["hi_ms"].median()})
            tables.append(_cut_table(m, name))
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=4):
        print(f"== implied cutoff for volumeTotal: {n} PM cells (lo/hi = window vs the label, ms)")
        print(pl.DataFrame(found))
        out = tables[0]
        for tb in tables[1:]:
            out = out.join(tb, on="delta_ms")
        print(out)
        miss = cells.join(t.select("code").unique(), on="code", how="semi").join(
            t.sort("code", "time", "seq_id").with_columns(pl.col("volume").cum_sum().over("code").alias("cum"))
            .select("code", "cum"), on=["code", "cum"], how="anti")
        print("cells no exchange-time trade prefix reaches:", miss.height)
        print(miss.head(8))


SNAP_FIELDS = (("volumeTotal", "total_volume", 0), ("tradesTotal", "total_no", 0), ("dollarVolumeTotal", "total_amount", 2))


def _exch(code: pl.Expr) -> pl.Expr:
    return pl.when((code.str.len_chars() == 6) & code.str.starts_with("6")).then(pl.lit("SH")).otherwise(pl.lit("SZ"))


def _hms(ns: pl.Expr) -> pl.Expr:
    sec = ns // NS
    return pl.format("{}:{}:{}", (sec // 3600).cast(pl.String).str.zfill(2), (sec % 3600 // 60).cast(pl.String).str.zfill(2),
                     (sec % 60).cast(pl.String).str.zfill(2))


def snap_base(pm: pl.DataFrame, date: str) -> None:
    """PM totals vs the raw snapshot total at the last snapshot (exchange time) at or before the label: one offset per stock, and which one."""
    labels = [x for x in sorted(pm["label"].unique().to_list())
              if any(lo < x <= hi for lo, hi in CONT)]
    q = (pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
         .select("symbol_id", "time", "total_volume", "total_no", "total_amount")
         .with_columns(pl.col("symbol_id").cast(pl.String).alias("code"),
                       ((pl.col("time") + LOCAL * NS) % (86400 * NS)).alias("key"))
         .collect().sort("key"))
    grid = (q.select("code").unique().join(pl.DataFrame({"label": labels}, schema={"label": pl.Int64}), how="cross")
            .with_columns((pl.col("label") * NS).alias("key")).sort("key"))
    g = grid.join_asof(q.select("code", "key", "total_volume", "total_no", "total_amount"), on="key", by="code",
                       strategy="backward")
    mode_b = None
    with pl.Config(tbl_rows=30, tbl_width_chars=200, float_precision=4):
        print(f"== snapshot base: last snapshot with exchange time <= label, continuous labels only ({len(labels)})")
        for v, col, nd in SNAP_FIELDS:
            j = (pm.filter((pl.col("V") == v) & pl.col("pm").is_not_nan()).join(g, on=["code", "label"])
                 .filter(pl.col(col).is_not_null())
                 .with_columns((pl.col(col).cast(pl.Float64) - pl.col("pm")).round(nd).alias("b")))
            per = j.group_by("code").agg(pl.len().alias("n"), pl.col("b").mode().first().alias("b_mode")).join(
                j.join(j.group_by("code").agg(pl.col("b").mode().first().alias("b_mode")), on="code")
                .group_by("code").agg((pl.col("b") == pl.col("b_mode")).mean().alias("mode_share")), on="code")
            per = per.with_columns(_exch(pl.col("code")).alias("exch"))
            hit = j.join(per.select("code", "b_mode"), on="code").select((pl.col("b") == pl.col("b_mode")).mean()).item()
            print(f"{v}: {j.height} cells, {per.height} stocks; cells equal to their stock's modal offset {hit:.4f}")
            print(per.group_by("exch").agg(pl.len().alias("stocks"), (pl.col("mode_share") >= 0.99).mean().alias("const_99"),
                                           pl.col("mode_share").median().alias("median_share"),
                                           (pl.col("b_mode") == 0).mean().alias("offset_0")).sort("exch"))
            if v == "volumeTotal":
                mode_b = per
        # which opening value is the volume offset
        t = (pl.scan_parquet(RAW / "kind=transaction" / f"date={date}" / "part-*.parquet")
             .filter(pl.col("trade_type") == 1).select("symbol_id", "time", "volume")
             .with_columns(pl.col("symbol_id").cast(pl.String).alias("code"),
                           ((pl.col("time") + LOCAL * NS) % (86400 * NS)).alias("key"))
             .filter(pl.col("key") < CONT[0][0] * NS).group_by("code").agg(pl.col("volume").sum().alias("auction_trades"))
             .collect())
        base = q.filter(pl.col("key") <= CONT[0][0] * NS).group_by("code").agg(
            pl.col("total_volume").last().alias("our_base"))
        m = mode_b.join(t, on="code", how="left").join(base, on="code", how="left")
        early = q.filter(pl.col("key") <= (CONT[0][0] + 120) * NS).select("code", "key", pl.col("total_volume").alias("b_mode"))
        first = early.join(m.select("code", pl.col("b_mode").cast(pl.Int64)), on=["code", "b_mode"]).group_by("code").agg(
            pl.col("key").min().alias("first_ns"), pl.col("key").max().alias("last_ns"))
        m = m.join(first, on="code", how="left")
        print("volume offset per stock vs candidates:")
        print(m.group_by("exch").agg(
            pl.len().alias("stocks"), (pl.col("b_mode") == pl.col("our_base")).mean().alias("= our base (snap<=09:30:00)"),
            (pl.col("b_mode") == pl.col("auction_trades")).mean().alias("= auction trades"),
            pl.col("first_ns").is_not_null().mean().alias("= some snap<=09:32")).sort("exch"))
        print("time of the first / last snapshot whose total equals the offset (top):")
        print(m.filter(pl.col("first_ns").is_not_null()).group_by("exch", _hms(pl.col("first_ns")).alias("first"),
                                                                   _hms(pl.col("last_ns")).alias("last"))
              .len().sort("len", descending=True).head(12))
        print("stocks whose offset is not constant (lowest share):")
        print(m.sort("mode_share").select("code", "exch", "n", "mode_share", "b_mode", "our_base", "auction_trades").head(8))
        print(m.filter(pl.col("code") == "100").select("code", "n", "mode_share", "b_mode", "our_base", "auction_trades"))


def snap_match(pm: pl.DataFrame, date: str, at: str) -> None:
    """For one label: which snapshot's total minus the opening auction equals PM's volumeTotal, and PM vs its own later labels."""
    lab = int(at[:2]) * 3600 + int(at[3:5]) * 60
    cells = pm.filter((pl.col("V") == "volumeTotal") & (pl.col("label") == lab) & pl.col("pm").is_not_nan()).select(
        "code", pl.col("pm").round(0).cast(pl.Int64).alias("cum"))
    q = (pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet")
         .select(pl.col("symbol_id").cast(pl.String).alias("code"), "time", "spider_ts", pl.col("total_volume").cast(pl.Int64))
         .with_columns(((pl.col("time") + LOCAL * NS) % (86400 * NS)).alias("key"),
                       ((pl.col("spider_ts") + LOCAL * NS) % (86400 * NS)).alias("lkey"))
         .collect().join(auction_totals(date).select("code", "total_volume_auct"), on="code", how="left")
         .with_columns((pl.col("total_volume") - pl.col("total_volume_auct").fill_null(0)).alias("cum")))
    m = cells.join(q, on=["code", "cum"]).group_by("code").agg(
        pl.col("key").min().alias("first"), pl.col("key").max().alias("last"), pl.col("lkey").min().alias("first_local"))
    cells = cells.with_columns(_exch(pl.col("code")).alias("exch"))
    with pl.Config(tbl_rows=30, tbl_width_chars=200, float_precision=4):
        print(f"== snapshot whose total - auction equals PM volumeTotal at {at}: {cells.height} stocks")
        print(cells.join(m, on="code", how="left").group_by("exch").agg(
            pl.len().alias("stocks"), pl.col("first").is_not_null().mean().alias("matched")).sort("exch"))
        print("exchange time of the first / last matching snapshot, and local arrival of the first (top):")
        print(m.join(cells.select("code", "exch"), on="code").group_by(
            "exch", _hms(pl.col("first")).alias("first"), _hms(pl.col("last")).alias("last"),
            _hms(pl.col("first_local")).alias("first_local")).len().sort("len", descending=True).head(15))
        # PM against its own neighbouring labels
        wide = pm.filter(pl.col("V") == "volumeTotal").pivot(on="label", index="code", values="pm")
        names = {str(x): f"{x // 3600:02d}:{x % 3600 // 60:02d}" for x in (lab - 60, lab, 13 * 3600, 13 * 3600 + 60, 13 * 3600 + 120)}
        wide = wide.select("code", *[pl.col(k).alias(v) for k, v in names.items() if k in wide.columns])
        print(f"PM {at} equal to PM at:", {c: round(wide.select((pl.col(at) == pl.col(c)).mean()).item(), 4)
                                         for c in wide.columns if c not in ("code", at)})
        miss = cells.join(m, on="code", how="anti").head(6)
        if miss.height:
            near = q.join(miss.select("code"), on="code").filter(
                (pl.col("key") >= (lab - 30) * NS) & (pl.col("key") <= (13 * 3600 + 30) * NS))
            print("unmatched stocks: snapshot totals - auction around the break vs PM")
            print(near.join(miss.select("code", pl.col("cum").alias("pm")), on="code")
                  .select("code", _hms(pl.col("key")).alias("exch_t"), "cum", "pm", (pl.col("cum") - pl.col("pm")).alias("diff"))
                  .head(30))


def score(pm: pl.DataFrame, ours: pl.DataFrame) -> pl.DataFrame:
    """Per (field, candidate) agreement with the PM cube."""
    j = pm.join(ours, on=["code", "label", "V"], how="inner")
    a, b = pl.col("pm"), pl.col("ours")
    both = a.is_not_null() & a.is_not_nan() & b.is_not_null() & b.is_not_nan()
    return (j.group_by("V", "cand").agg(
        pl.len().alias("cells"),
        (a.is_nan() | a.is_null()).sum().alias("pm_nan"),
        both.sum().alias("both"),
        ((a - b).abs() <= 1e-9 * a.abs().clip(1.0, None)).filter(both).mean().alias("exact"),
        ((a - b).abs() <= 1e-4 * a.abs().clip(1.0, None)).filter(both).mean().alias("close_1e4"),
        pl.corr(a.filter(both), b.filter(both)).alias("corr"),
        (a / b).filter(both & (b != 0)).median().alias("ratio"),
        ((a.is_nan() | a.is_null()) == (b.is_nan() | b.is_null())).mean().alias("nan_agree"))
            .with_columns(pl.col("ratio").fill_null(1.0))
            .pipe(lambda d: d.join(
                j.join(d.select("V", "cand", "ratio"), on=["V", "cand"]).group_by("V", "cand").agg(
                    ((a - b * pl.col("ratio")).abs() <= 1e-6 * a.abs().clip(1.0, None)).filter(both).mean()
                    .alias("exact_scaled")), on=["V", "cand"], how="left"))
            .sort("V", "exact", "nan_agree", "exact_scaled", "close_1e4", descending=[False, True, True, True, True]))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--prod", type=Path, default=PROD)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage2"))
    ap.add_argument("--implied-cut", action="store_true",
                    help="locate PM's volumeTotal cutoff against trade / snapshot running totals, then exit")
    ap.add_argument("--snap-match", metavar="HH:MM",
                    help="which snapshot PM's volumeTotal at this label comes from, then exit (e.g. 11:30)")
    ap.add_argument("--snap-base", action="store_true",
                    help="PM totals vs the snapshot total at the last snapshot <= label: per-stock offset, then exit")
    ap.add_argument("--diagnose", nargs=2, metavar=("V", "CAND"), action="append",
                    help="break one field/candidate down, e.g. --diagnose volumeTotal local/incl/cont")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pm = pm_cube(a.prod / "1min_src" / f"{a.date}.nc")
    if a.snap_match:
        snap_match(pm, a.date, a.snap_match)
        return
    if a.snap_base:
        snap_base(pm, a.date)
        return
    if a.implied_cut:
        implied_cut(pm, a.date)
        return
    labels = sorted(pm["label"].unique().to_list())
    ours = pl.concat([trade_candidates(a.date, labels), book_candidates(a.date, labels)])
    res = score(pm, ours)
    pmi = a.prod / "index" / "1min_src" / f"{a.date}.nc"
    if pmi.exists():
        res = pl.concat([res, score(pm_cube(pmi), index_candidates(a.date, labels)).with_columns(
            pl.concat_str(pl.lit("index:"), pl.col("V")).alias("V"))])
    for v, cand in a.diagnose or []:
        diagnose(pm, ours, v, cand)
    if a.diagnose:
        return
    res.write_parquet(a.out / f"bars_{a.date}.parquet")
    res.write_csv(a.out / f"bars_{a.date}.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=220, float_precision=4, fmt_str_lengths=40):
        print(res.group_by("V", maintain_order=True).head(2))
    print(f"full table: {a.out / f'bars_{a.date}.csv'}")


if __name__ == "__main__":
    main()
