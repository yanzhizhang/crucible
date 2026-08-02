"""Turn the raw CSV cache into a crucible store, standing in for a prism dump.

Emits the same partitioned layout :mod:`quarry` expects from the real producer,
including a ``_fingerprint.json`` sidecar, so the downstream pipeline runs
through the genuine loader and producer-validation path rather than a shortcut.

    <root>/daily/date=YYYYMMDD/part.parquet
    <root>/factor_frame/date=YYYYMMDD/part.parquet
    <root>/factor_frame/_fingerprint.json
    <root>/index/date=YYYYMMDD/part.parquet
    <root>/listings/part.parquet

Documented approximations
-------------------------
**Market cap history.** The free endpoints give only a *current* market cap.
Shares outstanding are held constant and cap is back-cast as
``shares * unadjusted_close``. This is right for the many names with no
share-count change over the window and wrong for those that issued or bought
back stock. It affects the ``size`` factor and size-neutralisation, not the
price-based factors.

**Listing dates.** Taken as the symbol's first observed bar when that falls
after the panel start (a genuine mid-sample entry), and otherwise treated as
seasoned. A top-80-by-market-cap name cannot be a recent listing, so this is
exact in effect for this universe.

**Index membership.** Not written. Point-in-time constituent history is not
available from these endpoints, and :class:`almanac.Universe` deliberately
refuses a membership table without effective dates rather than let today's
constituents be applied to the past. The pipeline therefore runs on the
``"all"`` universe rule and the selection bias is stated rather than hidden.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factors import FACTORS, compute_factors  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
STORE = ROOT / "data" / "store"

PRODUCER = "research-stub-v1"
"""Producer id baked into the fingerprint. A real dump would carry the prism
build hash; anything computed here is explicitly marked as not-prism."""


def load_raw() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read the fetched CSVs with explicit dtypes."""
    bars = pl.read_csv(
        RAW / "bars.csv",
        schema_overrides={
            "date": pl.Utf8,
            "symbol": pl.Utf8,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "amount": pl.Float64,
            "turnover": pl.Float64,
            "adj_factor": pl.Float64,
        },
    )
    uni = pl.read_csv(
        RAW / "universe.csv",
        schema_overrides={"symbol": pl.Utf8, "name": pl.Utf8, "industry": pl.Utf8, "mcap": pl.Float64},
    )
    return bars, uni


def prepare(bars: pl.DataFrame, uni: pl.DataFrame) -> pl.DataFrame:
    """Build the analysis panel: adjusted prices, cap history, market return."""
    df = (
        bars.with_columns(
            ts=pl.col("date").str.strptime(pl.Datetime("ns"), "%Y%m%d").dt.offset_by("15h")
        )
        .join(uni.select("symbol", "industry", "mcap"), on="symbol", how="left")
        .sort(["symbol", "ts"], maintain_order=True)
    )

    # Drop non-trading rows: zero volume is a suspension and its printed price
    # is stale. Keeping the row lets build_masks flag it, which is what we want,
    # so these are RETAINED here and masked later -- not silently dropped.
    df = df.with_columns(
        adj_close=pl.col("close") * pl.col("adj_factor"),
        vwap=pl.when(pl.col("volume") > 0)
        .then(pl.col("amount") / pl.col("volume"))
        .otherwise(pl.col("close")),
        prev_close=pl.col("close").shift(1).over("symbol"),
        log_volume=(pl.col("volume") + 1.0).log(),
    )

    # Shares outstanding from the current snapshot, held constant (see module
    # docstring). Uses the LAST unadjusted close per symbol as the anchor.
    last_close = df.group_by("symbol").agg(pl.col("close").last().alias("_last_close"))
    df = df.join(last_close, on="symbol", how="left").with_columns(
        market_cap=pl.col("mcap") / pl.col("_last_close") * pl.col("close")
    )

    # Equal-weighted market return: the benchmark for beta and idiosyncratic
    # vol. Equal-weighted rather than cap-weighted because this universe is all
    # large caps, so a cap-weighted mean would be dominated by a handful of names.
    df = df.with_columns(
        _ret1=pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0
    )
    mkt = df.group_by("ts").agg(pl.col("_ret1").mean().alias("mkt_ret")).sort("ts")
    df = df.join(mkt, on="ts", how="left")

    # Listing dates: a genuine mid-sample entry keeps its true first bar; a name
    # present from the start is seasoned by construction (top-80 by market cap).
    panel_start = df["ts"].min()
    first = df.group_by("symbol").agg(pl.col("ts").min().alias("_first"))
    seasoned = panel_start - dt.timedelta(days=400)  # type: ignore[operator]
    df = df.join(first, on="symbol", how="left").with_columns(
        list_date=pl.when(pl.col("_first") > panel_start + pl.duration(days=5))
        .then(pl.col("_first"))
        .otherwise(pl.lit(seasoned))
        .cast(pl.Datetime("ns"))
    )

    return df.drop("_last_close", "_first").sort(["symbol", "ts"], maintain_order=True)


def write_partitioned(df: pl.DataFrame, out_dir: Path, *, by_symbol: bool = False) -> int:
    """Write one Parquet part per date (and optionally per symbol)."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    n = 0
    for (stamp,), block in df.partition_by("ts", as_dict=True).items():  # type: ignore[misc]
        tag = stamp.strftime("%Y%m%d")
        if by_symbol:
            for (sym,), sub in block.partition_by("symbol", as_dict=True).items():  # type: ignore[misc]
                d = out_dir / f"date={tag}" / f"symbol={sym}"
                d.mkdir(parents=True, exist_ok=True)
                sub.write_parquet(d / "part.parquet", compression="zstd")
                n += 1
        else:
            d = out_dir / f"date={tag}"
            d.mkdir(parents=True, exist_ok=True)
            block.write_parquet(d / "part.parquet", compression="zstd")
            n += 1
    return n


def main() -> None:
    print("loading raw CSV ...")
    bars, uni = load_raw()
    print(f"  {bars.height:,} bars | {uni.height} universe rows")

    panel = prepare(bars, uni)
    n_sym = panel["symbol"].n_unique()
    n_ts = panel["ts"].n_unique()
    print(f"  panel: {n_sym} symbols x {n_ts} sessions = {panel.height:,} rows")
    print(f"  span : {panel['ts'].min()} .. {panel['ts'].max()}")

    print(f"\ncomputing {len(FACTORS)} factors (prism stand-in) ...")
    ff = compute_factors(panel)
    coverage = {
        name: 1.0 - ff[name].null_count() / ff.height for name in FACTORS
    }
    thin = {k: v for k, v in coverage.items() if v < 0.5}
    print(f"  mean non-null coverage: {sum(coverage.values()) / len(coverage):.1%}")
    if thin:
        print(f"  ! low-coverage factors (<50%): {thin}")

    STORE.mkdir(parents=True, exist_ok=True)

    daily = panel.select(
        "ts", "symbol", "open", "high", "low", "close", "prev_close", "vwap",
        "volume", "amount", "turnover", "adj_close", "adj_factor",
        "market_cap", "industry", "list_date",
    ).with_columns(is_st=pl.lit(False))
    n = write_partitioned(daily, STORE / "daily")
    print(f"\nwrote daily         : {n} partitions")

    n = write_partitioned(ff, STORE / "factor_frame")
    print(f"wrote factor_frame  : {n} partitions")

    index = (
        panel.group_by("ts")
        .agg(pl.col("mkt_ret").first().alias("ret"))
        .sort("ts")
        .with_columns(
            symbol=pl.lit("EW.MKT"),
            close=((pl.col("ret").fill_null(0.0) + 1.0).cum_prod() * 1000.0),
        )
        .select("ts", "symbol", "close", "ret")
    )
    n = write_partitioned(index, STORE / "index")
    print(f"wrote index         : {n} partitions")

    listings = (
        panel.group_by("symbol")
        .agg(pl.col("list_date").first())
        .with_columns(
            list_date=pl.col("list_date").dt.date(),
            delist_date=pl.lit(None, dtype=pl.Date),
        )
        .sort("symbol")
    )
    (STORE / "listings").mkdir(parents=True, exist_ok=True)
    listings.write_parquet(STORE / "listings" / "part.parquet", compression="zstd")
    print(f"wrote listings      : {listings.height} symbols")

    fingerprint = {
        "producer": PRODUCER,
        "factors": [
            {"name": n, "params": {"family": fam}, "dtype": "DOUBLE"}
            for n, (fam, _, _) in FACTORS.items()
        ],
    }
    from crucible.determinism import stable_hash

    fingerprint["digest"] = stable_hash(
        "schema-v1",
        PRODUCER,
        tuple(
            (n, tuple(sorted({"family": fam}.items())), "DOUBLE")
            for n, (fam, _, _) in FACTORS.items()
        ),
    )
    (STORE / "factor_frame" / "_fingerprint.json").write_text(
        json.dumps(fingerprint, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote fingerprint   : {fingerprint['digest'][:16]}...")

    print(f"\nstore ready at {STORE}")


if __name__ == "__main__":
    main()
