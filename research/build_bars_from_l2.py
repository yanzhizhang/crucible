"""Build right-labelled 1-minute stock bars from the tick-by-tick trade stream.

Convention (matches :func:`quarry.load_bars` and :meth:`almanac.TradingCalendar.slot_grid`):

* bar ``t`` covers exchange time ``(t - 1min, t]``, timestamps tz-naive Asia/Shanghai;
* opening-auction trades (09:25) fold into the first continuous bar (09:31), and a trade
  stamped exactly on a session open (09:30:00 / 13:00:00) goes to that session's first bar;
* closing-auction trades (15:00) form the single 15:00 bar;
* the panel is rectangular: every stock that traded that day gets every slot; a slot with no
  trade has null OHLC, zero volume, and ``last`` carries the previous close forward
  (masked, not dropped -- crucible convention);
* ``available_ts`` is the point-in-time moment the bar became knowable on this capture:
  the latest of (a) the label itself, (b) the arrival of the bar's last trade, and (c) the
  exchange watermark -- the first arrival of any trade on that exchange stamped in the next
  slot within the same session, i.e. the moment the feed demonstrably moved past the label
  (session-final bars are closed by the schedule). At the open the capture
  lags by tens of seconds, so (b) and (c) dominate exactly when volume peaks.

These choices are the working definition, to be calibrated against Wind ``w.wsi`` and PM's
``1min_src`` (``--open-auction own_bar`` switches the auction treatment).

Memory: each minute file is reduced to partial bars on its own (first/last carried with
their (exchange_ts, seq) keys), and partials are merged at the end, so a full day never has
to fit in memory at once.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.resmon import ResourceMonitor, append_history

from almanac.calendar import EQUITY, TradingCalendar
from quarry.feitu import normalize_columns

_LOCAL = pl.duration(hours=8)


def _stock_keys(quote_dir: Path) -> pl.DataFrame:
    """(exchange, symbol, pre_close) of every stock in the day's snapshots."""
    return (
        normalize_columns(pl.scan_parquet(quote_dir / "part-*.parquet"), "quotation")
        .filter(pl.col("is_a_share"))
        .group_by("exchange", "symbol")
        .agg(pl.col("pre_close").filter(pl.col("pre_close") > 0).first())
        .collect(engine="streaming")
    )


def _labels(day: dt.date, open_auction: str) -> tuple[pl.Expr, pl.Series]:
    grid = TradingCalendar("XSHG", EQUITY).slot_grid(day, "1min")
    t = (pl.from_epoch("exchange_ts", "ns") + _LOCAL).alias("t")
    ceil = t.dt.truncate("1m") + pl.when(t.dt.truncate("1m") == t).then(
        pl.duration(minutes=0)
    ).otherwise(pl.duration(minutes=1))
    d = dt.datetime.combine
    am_open, am_first = d(day, dt.time(9, 30)), d(day, dt.time(9, 31))
    pm_open, pm_first = d(day, dt.time(13, 0)), d(day, dt.time(13, 1))
    close_auc = d(day, dt.time(15, 0))
    label = (
        pl.when(ceil <= am_open)
        .then(pl.lit(am_first if open_auction == "first_bar" else am_open))
        .when((ceil > d(day, dt.time(11, 30))) & (ceil <= pm_open))
        .then(pl.lit(pm_first))
        .when(ceil > d(day, dt.time(14, 57)))
        .then(pl.lit(close_auc))
        .otherwise(ceil)
    )
    if open_auction == "own_bar":
        grid = pl.concat([pl.Series("ts", [am_open], dtype=pl.Datetime("ns")), grid])
    return label.cast(pl.Datetime("ns")).alias("ts"), grid


def _partial(f: Path, label: pl.Expr, stocks: pl.DataFrame) -> pl.DataFrame:
    tr = (
        normalize_columns(pl.scan_parquet(f), "transaction")
        .filter(pl.col("record_type") == "trade")
        .join(stocks.lazy().select("exchange", "symbol"), on=["exchange", "symbol"], how="semi")
        .with_columns(label)
    )
    return (
        tr.group_by("exchange", "symbol", "ts")
        .agg(
            pl.col("exchange_ts").min().alias("k0_t"),
            pl.col("seq_id")
            .filter(pl.col("exchange_ts") == pl.col("exchange_ts").min())
            .min()
            .alias("k0_s"),
            pl.col("price").sort_by("exchange_ts", "seq_id").first().alias("open"),
            pl.col("exchange_ts").max().alias("k1_t"),
            pl.col("seq_id")
            .filter(pl.col("exchange_ts") == pl.col("exchange_ts").max())
            .max()
            .alias("k1_s"),
            pl.col("price").sort_by("exchange_ts", "seq_id").last().alias("close"),
            pl.col("price").max().alias("high"),
            pl.col("price").min().alias("low"),
            pl.col("volume").sum().alias("volume"),
            (pl.col("price") * pl.col("volume")).sum().alias("amount"),
            pl.len().alias("n_trades"),
            pl.col("volume").filter(pl.col("side") == "1").sum().alias("buy_volume"),
            pl.col("volume").filter(pl.col("side") == "2").sum().alias("sell_volume"),
            pl.col("arrival_ts").max().alias("arrival_max"),
            pl.col("arrival_ts").min().alias("arrival_min"),
        )
        .collect()
    )


def require_quality(quality: Path, date: str, source: str, *, allow_fail: bool) -> str:
    """Refuse a day the market-data quality gate failed (or never graded), unless overridden."""
    f = quality / f"date={date}" / f"source={source}" / "verdict.json"
    if not f.exists():
        if allow_fail:
            print(f"WARNING {date}: no quality verdict, building anyway (--allow-fail)")
            return "UNGRADED"
        raise SystemExit(f"{date}: no quality verdict at {f}; run research/md_quality/run.py")
    v = json.loads(f.read_text())["verdict"]
    if v == "FAIL":
        if not allow_fail:
            raise SystemExit(f"{date}: quality verdict FAIL -- refusing (see checks.parquet)")
        print(f"WARNING {date}: quality verdict FAIL, building anyway (--allow-fail)")
    return str(v)


def build_day(raw: Path, date: str, out: Path, *, open_auction: str, threads: int) -> int:
    """Build and write one day's bars; return the number of bar rows."""
    day = dt.date(int(date[:4]), int(date[4:6]), int(date[6:]))
    tdir = raw / "kind=transaction" / f"date={date}"
    files = sorted(tdir.glob("part-*.parquet"))
    if not files:
        raise SystemExit(f"no transaction files for {date}: bars need the trade stream")
    stocks = _stock_keys(raw / "kind=quotation" / f"date={date}")
    label, grid = _labels(day, open_auction)

    with ThreadPoolExecutor(threads) as ex:
        parts = list(ex.map(lambda f: _partial(f, label, stocks), files))
    p = pl.concat([x for x in parts if x.height], how="vertical")

    # merge partials: open from the earliest (exchange_ts, seq) key, close from the latest
    bars = (
        p.sort("k0_t", "k0_s")
        .group_by("exchange", "symbol", "ts", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("close").sort_by("k1_t", "k1_s").last(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("volume").sum(),
            pl.col("amount").sum(),
            pl.col("n_trades").sum(),
            pl.col("buy_volume").sum(),
            pl.col("sell_volume").sum(),
            pl.col("arrival_max").max().alias("arrival_max"),
        )
    )
    off_grid = bars.filter(~pl.col("ts").is_in(grid.implode()))
    if off_grid.height:
        raise ValueError(
            f"{off_grid.height} bars fell off the slot grid, e.g. "
            f"{off_grid.head(3).select('symbol', 'ts').rows()}"
        )

    # exchange watermark: first arrival of any trade in the following slot. Only within a
    # session: a session-final bar (11:30, 14:57) is closed by the schedule, not by the next
    # trade an hour later.
    first_in_slot = p.group_by("exchange", "ts").agg(pl.col("arrival_min").min().alias("first"))
    grid_df = grid.to_frame("ts").with_row_index("slot")
    wm = (
        first_in_slot.join(grid_df, on="ts")
        .with_columns((pl.col("slot") - 1).alias("slot"))
        .join(grid_df, on="slot", suffix="_prev")
        .filter(pl.col("ts") - pl.col("ts_prev") == pl.duration(minutes=1))
        .select("exchange", pl.col("ts_prev").alias("ts"), pl.col("first").alias("watermark"))
    )
    label_ns = pl.col("ts").dt.epoch("ns") - 8 * 3600 * 10**9
    bars = (
        bars.join(wm, on=["exchange", "ts"], how="left")
        .with_columns(
            (
                pl.from_epoch(
                    pl.max_horizontal(label_ns, pl.col("arrival_max"), pl.col("watermark")), "ns"
                )
                + _LOCAL
            ).alias("available_ts")
        )
        .drop("arrival_max", "watermark")
    )

    traded = bars.select("exchange", "symbol").unique()
    full = (
        traded.join(grid.to_frame("ts"), how="cross")
        .join(bars, on=["exchange", "symbol", "ts"], how="left")
        .join(stocks, on=["exchange", "symbol"], how="left")
        .sort("exchange", "symbol", "ts")
        .with_columns(
            pl.col("volume", "amount", "n_trades", "buy_volume", "sell_volume").fill_null(0),
            (pl.col("amount") / pl.col("volume")).alias("vwap"),
            pl.col("close")
            .forward_fill()
            .over("exchange", "symbol")
            .fill_null(pl.col("pre_close"))
            .alias("last"),
        )
        .with_columns(pl.when(pl.col("volume") > 0).then(pl.col("vwap")).alias("vwap"))
        .select(
            "ts",
            "symbol",
            "exchange",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "n_trades",
            "buy_volume",
            "sell_volume",
            "last",
            "pre_close",
            "available_ts",
        )
    )
    dst = out / "bars" / "1min" / f"date={date}"
    dst.mkdir(parents=True, exist_ok=True)
    full.write_parquet(dst / "part.parquet", compression="zstd")
    return full.height


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/store"))
    ap.add_argument("--open-auction", choices=("first_bar", "own_bar"), default="first_bar")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quality", type=Path, default=Path("/work/crucible_data/quality"))
    ap.add_argument("--source", default="feitu_v3")
    ap.add_argument(
        "--allow-fail", action="store_true", help="build even if the quality gate failed (logged)"
    )
    a = ap.parse_args()
    verdict = require_quality(a.quality, a.date, a.source, allow_fail=a.allow_fail)
    print(f"{a.date}: quality verdict {verdict}")
    with ResourceMonitor("W2.bars_1min", params={"date": a.date}) as mon:
        mon.rows = build_day(a.raw, a.date, a.out, open_auction=a.open_auction, threads=a.threads)
    assert mon.stats is not None
    print(mon.stats.line())
    append_history(mon.stats, a.out.parent / "bench" / "history.parquet")


if __name__ == "__main__":
    main()
