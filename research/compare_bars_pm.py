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
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

RAW = Path("/work/crucible_data/feitu_raw")
PROD = Path("/data/prod")
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


def book_candidates(date: str, labels: list[int]) -> pl.DataFrame:
    """Book at the minute's end (last snapshot at or before the label) under (clock, edge)."""
    q = pl.scan_parquet(RAW / "kind=quotation" / f"date={date}" / "part-*.parquet").select(
        "symbol_id", "time", "spider_ts", "price", "total_volume", "total_amount", "total_no",
        pl.col("ask_px").arr.get(0).alias("askPrice"),
        pl.col("ask_vol").arr.get(0).alias("askSize"), pl.col("bid_px").arr.get(0).alias("bidPrice"),
        pl.col("bid_vol").arr.get(0).alias("bidSize"))
    valid = (pl.col("askPrice") > 0) & (pl.col("bidPrice") > 0)
    q = q.with_columns(pl.when(valid).then((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("mid_valid"))
    out = []
    for clock, edge in itertools.product(("time", "spider_ts"), ("incl", "excl")):
        qq = q.with_columns(pl.col("symbol_id").cast(pl.String).alias("code"),
                            ((pl.col(clock) + LOCAL * NS) % (86400 * NS)).alias("sod_ns"))
        qq = qq.with_columns(_label(edge), (pl.col("sod_ns") / NS).alias("sod")).sort(clock)
        # snapshot cumulative totals right after the open auction (last snapshot at or before 09:30)
        base = qq.filter(pl.col("sod") <= CONT[0][0]).group_by("code").agg(
            pl.col("total_volume", "total_amount", "total_no").last().name.suffix("_open"))
        qq = qq.group_by("code", "label").agg(
            pl.col("price", "askPrice", "askSize", "bidPrice", "bidSize", "total_volume", "total_amount",
                   "total_no").last(),
            pl.col("mid_valid").drop_nulls().last()).collect().sort("code", "label")
        qq = qq.join(base.collect(), on="code", how="left").with_columns(
            ((pl.col("askPrice") + pl.col("bidPrice")) / 2).alias("mid_plain"),
            pl.when((pl.col("askPrice") > 0) & (pl.col("bidPrice") > 0))
            .then((pl.col("askPrice") + pl.col("bidPrice")) / 2).otherwise(0.0).alias("mid_or0"),
            pl.col("price").alias("lastTradePrice"))
        grid = qq.select("code").unique().join(pl.DataFrame({"label": labels}, schema={"label": pl.Int64}), how="cross").sort("label")
        g = grid.join_asof(qq.drop("mid_valid").sort("label"), on="label", by="code", strategy="backward")
        # last valid mid carried forward: as-of over the minutes that had one
        mv = qq.filter(pl.col("mid_valid").is_not_null()).select("code", "label", "mid_valid").sort("label")
        g = g.join_asof(mv, on="label", by="code", strategy="backward").sort("code", "label")
        cont = _in_cont(pl.col("label").cast(pl.Float64))
        g = g.with_columns(
            *[pl.when(cont).then(pl.col(c) - pl.col(f"{c}_open"))
              .alias(f"snap_{c}") for c in ("total_volume", "total_amount", "total_no")],
            pl.when(pl.col("total_volume").diff().over("code") > 0).then(pl.col("price")).alias("snap_close"))
        tag = f"{'exch' if clock == 'time' else 'local'}/{edge}"
        pairs = [(v, v, "book") for v in ("lastTradePrice", "askPrice", "askSize", "bidPrice", "bidSize")]
        pairs += [("midPriceLastValid", "mid_plain", "mid_plain"), ("midPriceLastValid", "mid_or0", "mid_or0"),
                  ("midPriceLastValid", "mid_valid", "mid_lastvalid"),
                  ("volumeTotal", "snap_total_volume", "snapcum"), ("dollarVolumeTotal", "snap_total_amount", "snapcum"),
                  ("tradesTotal", "snap_total_no", "snapcum"), ("close", "snap_close", "snaplast")]
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


def diagnose(pm: pl.DataFrame, ours: pl.DataFrame, v: str, cand: str) -> None:
    """Where one candidate disagrees: by exchange, by time of day, and sample cells."""
    j = pm.filter(pl.col("V") == v).join(ours.filter((pl.col("V") == v) & (pl.col("cand") == cand)),
                                          on=["code", "label", "V"], how="inner")
    a, b = pl.col("pm"), pl.col("ours")
    j = j.filter(a.is_not_nan() & b.is_not_null() & b.is_not_nan()).with_columns(
        ((a - b).abs() <= 1e-9 * a.abs().clip(1.0, None)).alias("ok"),
        pl.when((pl.col("code").str.len_chars() == 6) & pl.col("code").str.starts_with("6"))
        .then(pl.lit("SH")).otherwise(pl.lit("SZ")).alias("exch"),  # codes are unpadded: '1' = 000001
        (pl.col("label") // 3600).alias("hour"), (b - a).alias("ours_minus_pm"))
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=6):
        print(f"== diagnose {v} / {cand}: {j.height} cells, exact {j['ok'].mean():.4f}")
        print(j.group_by("exch").agg(pl.len(), pl.col("ok").mean()).sort("exch"))
        print(j.group_by("hour").agg(pl.len(), pl.col("ok").mean()).sort("hour"))
        bad = j.filter(~pl.col("ok"))
        print("stocks with any mismatch:", bad["code"].n_unique(), "of", j["code"].n_unique())
        print(bad.group_by("code").agg(pl.len().alias("bad_cells"), pl.col("ours_minus_pm").median().alias("median_diff"))
              .sort("bad_cells", descending=True).head(8))
        print(bad.with_columns((pl.col("label").map_elements(lambda x: f"{x // 3600:02d}:{x % 3600 // 60:02d}",
                                                              return_dtype=pl.String)).alias("t"))
              .select("code", "t", "pm", "ours", "ours_minus_pm").head(12))


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
            .sort("V", "exact", "exact_scaled", "close_1e4", descending=[False, True, True, True]))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--prod", type=Path, default=PROD)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage2"))
    ap.add_argument("--diagnose", nargs=2, metavar=("V", "CAND"), action="append",
                    help="break one field/candidate down, e.g. --diagnose volumeTotal local/incl/cont")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    pm = pm_cube(a.prod / "1min_src" / f"{a.date}.nc")
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
    with pl.Config(tbl_rows=80, tbl_width_chars=200, float_precision=4):
        print(res.group_by("V", maintain_order=True).head(2))
    print(f"full table: {a.out / f'bars_{a.date}.csv'}")


if __name__ == "__main__":
    main()
