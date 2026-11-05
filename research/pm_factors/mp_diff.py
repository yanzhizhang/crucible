r"""PM's own feed (``.mp``, decoded by ``research/decode_pm_mp.py``) against feitu, record by record, for one day.

The samplerS gaps left over (ZZQO: PM's order volume <= ours, one-sided; ZZAL: book totals off by ~0.1%) look like a
feed difference, not a formula. This joins the two decodes of the same day:

* orders -- SSE by (symbol, add / delete, order_id), SZSE by (symbol, seq_id): present in one feed only, volume equal /
  larger / smaller; for SSE adds that differ, whether PM's volume = feitu's + the order's own aggressive fills (PM
  carries the original quantity, feitu the remainder); PM-only SSE adds that are wholly filled on arrival;
* per (stock, exchange-time minute): new-order volume PM / feitu - 1, the ZZQO cell view;
* trades by (symbol, buy id, sell id) and snapshots by (symbol, exchange time): counts, matched share, cum volume;
* receive clocks: median PM local time - feitu spider_ts on matched records (PM's SSE bar clock is ours - 25 ms).

    python research/pm_factors/mp_diff.py --date 20260430 --mp-raw /work/crucible_data/mp_raw
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

SSE, SZSE, BUY = 3553, 3554, 1063
NS_LOCAL, NS_DAY, NS_MIN = 8 * 3600 * 10**9, 86_400 * 10**9, 60 * 10**9


def scan(root: Path, kind: str, d: str, pattern: str = "part-*.parquet") -> pl.LazyFrame:
    """One kind of one day from a feitu-layout root."""
    return pl.scan_parquet(root / f"kind={kind}" / f"date={d}" / pattern)


def q(s: pl.Series) -> str:
    """p10 / p50 / p90 of a series."""
    s = s.drop_nulls().filter(s.drop_nulls().is_finite()) if s.dtype.is_float() else s.drop_nulls()
    return "/".join(f"{s.quantile(x):+.4f}" for x in (0.1, 0.5, 0.9)) if s.len() else "-"


def fm(s: pl.Series) -> str:
    """Mean of a boolean series, '-' when empty."""
    v = s.mean()
    return "-" if v is None else f"{v:.4f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--mp-raw", type=Path, default=Path("/work/crucible_data/mp_raw"), help="decode_pm_mp.py --out")
    ap.add_argument("--feitu-raw", type=Path, default=Path("/work/crucible_data/feitu_raw"))
    ap.add_argument("--idx", default="HS300", help="which .mp index files (part-<EX>_<idx>*)")
    a = ap.parse_args()
    d = a.date
    mp = {k: scan(a.mp_raw, k, d, f"part-*_{a.idx}*.parquet") for k in ("order", "transaction", "quotation")}
    syms = mp["quotation"].select("symbol_id", "market_id").unique().collect()
    ft = {k: scan(a.feitu_raw, k, d).join(syms.lazy(), on=["symbol_id", "market_id"], how="semi") for k in mp}
    print(f"{d}: {syms.height} stocks ({a.idx})")

    # orders
    oc = ["symbol_id", "market_id", "update_type", "order_id", "seq_id", "volume", "price", "dir", "time", "spider_ts"]
    om = mp["order"].select(oc).collect()
    of = ft["order"].select(oc).collect()
    ag = (ft["transaction"].filter((pl.col("trade_type") == 1) & (pl.col("market_id") == SSE) & pl.col("dir").is_in([BUY, 1550]))
          .with_columns(pl.when(pl.col("dir") == BUY).then(pl.col("buy_seq_id")).otherwise(pl.col("sell_seq_id")).alias("order_id"))
          .group_by("symbol_id", "order_id").agg(pl.col("volume").sum().alias("fv")).collect())  # SSE aggressive fills per order
    for mk, keys in ((SSE, ["symbol_id", "update_type", "order_id"]), (SZSE, ["symbol_id", "seq_id"])):
        for ut in ((1, 2) if mk == SSE else (1,)):
            x, y = (f.filter((pl.col("market_id") == mk) & (pl.col("update_type") == ut)) for f in (om, of))
            j = x.join(y, on=keys, how="full", coalesce=True, suffix="_f")
            both = j.filter(pl.col("volume").is_not_null() & pl.col("volume_f").is_not_null())
            mo, fo = j.filter(pl.col("volume_f").is_null()), j.filter(pl.col("volume").is_null())
            name = f"{'SH' if mk == SSE else 'SZ'} {'add' if ut == 1 else 'del'}"
            print(f"{name}: mp {x.height:,} feitu {y.height:,} both {both.height:,} mp-only {mo.height:,} ({mo['volume'].sum() / max(y['volume'].sum(), 1):.4%} of vol) "
                  f"feitu-only {fo.height:,} ({fo['volume_f'].sum() / max(y['volume'].sum(), 1):.4%} of vol); total vol mp/feitu-1 {x['volume'].sum() / max(y['volume'].sum(), 1) - 1:+.4%}")
            if both.height:
                print(f"   matched: vol eq {fm(both['volume'] == both['volume_f'])} mp>feitu {fm(both['volume'] > both['volume_f'])} "
                      f"mp<feitu {fm(both['volume'] < both['volume_f'])} price eq {fm((both['price'] - both['price_f']).abs() < 1e-9)} "
                      f"time eq {fm(both['time'] == both['time_f'])}; mp local - feitu spider (ms) {q((both['spider_ts'] - both['spider_ts_f']).cast(pl.Float64) / 1e6)}")
            if mk == SSE and ut == 1:
                ne = both.filter(pl.col("volume") != pl.col("volume_f")).join(ag, on=["symbol_id", "order_id"], how="left")
                if ne.height:
                    print(f"   differing adds {ne.height:,}: mp = feitu + own fills {fm(ne['volume'] == ne['volume_f'] + ne['fv'].fill_null(0))}")
                mf = mo.join(ag, on=["symbol_id", "order_id"], how="left")
                if mf.height:
                    print(f"   mp-only adds: wholly filled on arrival (vol = own fills) {fm(mf['volume'] == mf['fv'])}, any own fill {fm(mf['fv'].is_not_null())}")

    # new-order volume per (stock, minute), the ZZQO cell view
    def per_min(f: pl.DataFrame) -> pl.DataFrame:
        return (f.filter(pl.col("update_type") == 1).with_columns(((pl.col("time") + NS_LOCAL) % NS_DAY // NS_MIN).alias("m"))
                .group_by("symbol_id", "market_id", "m").agg(pl.col("volume").sum()))
    pm_ = per_min(om).join(per_min(of), on=["symbol_id", "market_id", "m"], how="full", coalesce=True, suffix="_f").filter(pl.col("m") >= 570)
    for mk in (SSE, SZSE):
        z = pm_.filter(pl.col("market_id") == mk).with_columns((pl.col("volume").fill_null(0) / pl.col("volume_f") - 1).alias("r"))
        print(f"{'SH' if mk == SSE else 'SZ'} per-minute new-order vol mp/feitu-1: eq {fm(z['r'] == 0)} p10/p50/p90 {q(z['r'])} "
              f"share mp<feitu {fm(z['r'] < 0)} mp>feitu {fm(z['r'] > 0)}")

    # trades
    tc = ["symbol_id", "market_id", "trade_type", "buy_seq_id", "sell_seq_id", "volume", "price", "time", "spider_ts"]
    tm, tf = mp["transaction"].select(tc).collect(), ft["transaction"].select(*tc, "dir").collect()
    # .mp trades carry no BS flag: how often does "the larger order number is the aggressor" disagree with feitu's flag (SSE, continuous)?
    sx = tf.filter((pl.col("market_id") == SSE) & (pl.col("trade_type") == 1)).with_columns(
        (((pl.col("time") + NS_LOCAL) % NS_DAY) / NS_MIN).alias("mx")).filter((pl.col("mx") >= 570) & (pl.col("mx") < 897))
    sx = sx.with_columns(pl.when(pl.col("buy_seq_id") > pl.col("sell_seq_id")).then(BUY).otherwise(1550).alias("big"))
    fl = sx.filter(pl.col("dir").is_in([BUY, 1550]))
    print(f"SH continuous trades {sx.height:,}: BS flag N {fm(~pl.Series(sx['dir'].is_in([BUY, 1550])))}; "
          f"flag = larger order no {fm(fl['dir'] == fl['big'])} (by volume {(fl.filter(pl.col('dir') == pl.col('big'))['volume'].sum() / max(fl['volume'].sum(), 1)):.4f})")
    for mk in (SSE, SZSE):
        x, y = tm.filter(pl.col("market_id") == mk), tf.filter(pl.col("market_id") == mk)
        j = x.join(y, on=["symbol_id", "trade_type", "buy_seq_id", "sell_seq_id"], how="full", coalesce=True, suffix="_f")
        both = j.filter(pl.col("volume").is_not_null() & pl.col("volume_f").is_not_null())
        print(f"{'SH' if mk == SSE else 'SZ'} trades: mp {x.height:,} feitu {y.height:,} both {both.height:,} vol mp/feitu-1 {x['volume'].sum() / max(y['volume'].sum(), 1) - 1:+.4%}; "
              f"matched vol eq {fm(both['volume'] == both['volume_f'])}; mp local - feitu spider (ms) {q((both['spider_ts'] - both['spider_ts_f']).cast(pl.Float64) / 1e6)}")

    # snapshots
    sc = ["symbol_id", "market_id", "time", "total_volume", "total_amount", "spider_ts"]
    sm, sf = mp["quotation"].select(sc).collect(), ft["quotation"].select(sc).collect()
    for mk in (SSE, SZSE):
        x, y = sm.filter(pl.col("market_id") == mk), sf.filter(pl.col("market_id") == mk)
        both = x.with_columns((pl.col("time") + 500_000) // 1_000_000).join(  # .mp times are float seconds: match to the ms
            y.with_columns((pl.col("time") + 500_000) // 1_000_000), on=["symbol_id", "time"], how="inner", suffix="_f")
        print(f"{'SH' if mk == SSE else 'SZ'} snaps: mp {x.height:,} feitu {y.height:,} same exch time {both.height:,}; cum vol eq {fm(both['total_volume'] == both['total_volume_f'])} "
              f"cum amt eq {fm((both['total_amount'] - both['total_amount_f']).abs() < 1e-6)}; mp local - feitu spider (ms) {q((both['spider_ts'] - both['spider_ts_f']).cast(pl.Float64) / 1e6)}")


if __name__ == "__main__":
    main()
