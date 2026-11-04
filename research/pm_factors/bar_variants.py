r"""Which 5-minute bar definition do PM's rolling-median trade columns use? Bar label, trade side set and auctions, crossed.

``window_probe.py`` put ZZXC q ~ median_22(vol) at 0.96 bit-exact on SZSE but 0.57 on SSE, worst in the 09:35 bar,
with no single day to blame -- a bar definition, not a data day. Per trading day this rebuilds vol / amt / b_amt per
(stock, 5-minute bar) from ``raw_transaction`` under 16 variants and scores each window median against PM by exchange:

* ``lab`` c = right-closed (t-1min, t] like ``basics.py``, f = left-closed [t-1min, t) (a trade on the minute opens the next bar);
* ``dir`` bs = trades with a buy / sell side only (``basics.py``: drops SSE auction trades, side N), all = every fill;
* ``open`` / ``close`` 1 = the auction folded into the 09:35 / 15:00 bar, 0 = left out.

Day tables are cached under ``store/pm_factors/bar_variants/date=<D>.parquet``.

    python research/pm_factors/bar_variants.py --days 20261020,20261021,20261022,20261023,20261026,20261027
"""

from __future__ import annotations

import argparse
import sys
import warnings
from itertools import product
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_candidates import pm_frame
from sampler_r import BUY, CATALOG, DATA, SELL
from window_match import trading_days
from window_probe import hit

OUT = DATA / "store" / "pm_factors" / "bar_variants"
NS_MIN, NS_DAY, NS_LOCAL = 60_000_000_000, 86_400_000_000_000, 8 * 3600 * 1_000_000_000
VARIANTS = [dict(zip(("lab", "dir", "open", "close"), v)) for v in product("cf", ("bs", "all"), (1, 0), (1, 0))]
CURRENT = "c bs 1 1"  # what basics.py builds


def vname(v: dict) -> str:
    """Short variant label: lab dir open close."""
    return f"{v['lab']} {v['dir']} {v['open']} {v['close']}"


def raw_day(con: duckdb.DuckDBPyConnection, d: str) -> pl.DataFrame:
    """Sum volume and amount per (code, phase, sided, buy, bar_c, bar_f)."""
    m = f"(((time + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE)"

    def lab(x: str) -> str:  # minute label -> 5-minute bar end, auctions to 09:35 / 15:00, an 11:30:00 / 13:00:00 print to its session
        x = f"(CASE WHEN m < 570 THEN 571 WHEN m > 897 THEN 900 WHEN {x} > 690 AND {x} <= 780 THEN (CASE WHEN m < 700 THEN 690 ELSE 781 END) ELSE {x} END)"
        return f"(CASE WHEN {x} >= 896 THEN 900 ELSE (({x} - 571) // 5) * 5 + 575 END)"

    return con.sql(f"""
        WITH t AS (
          SELECT symbol_id::BIGINT AS code, dir, volume::DOUBLE AS vol, volume * price AS amt, {m} AS m
          FROM raw_transaction WHERE date = {int(d)} AND trade_type = 1 AND symbol_id IN (SELECT code FROM codes)),
        b AS (SELECT *, CAST(ceil(m) AS INTEGER) AS mc, CAST(floor(m) AS INTEGER) + 1 AS mf FROM t)
        SELECT code, CASE WHEN m < 570 THEN 'open' WHEN m > 897 THEN 'close' ELSE 'cont' END AS phase,
               dir IN ({BUY}, {SELL}) AS sided, dir = {BUY} AS buy, {lab('mc')} AS bar_c, {lab('mf')} AS bar_f,
               sum(vol) AS vol, sum(amt) AS amt
        FROM b GROUP BY ALL""").pl()


def variants(raw: pl.DataFrame) -> pl.DataFrame:
    """Long table (code, minute, variant, vol, amt, b_amt) for every variant."""
    out = []
    for v in VARIANTS:
        keep = (pl.col("phase") == "cont") | ((pl.col("phase") == "open") & bool(v["open"])) | ((pl.col("phase") == "close") & bool(v["close"]))
        flows = [pl.col("vol").filter(pl.col("sided")).sum(), pl.col("amt").filter(pl.col("sided")).sum()] if v["dir"] == "bs" else [pl.col("vol").sum(), pl.col("amt").sum()]
        out.append(raw.filter(keep).group_by("code", pl.col(f"bar_{v['lab']}").alias("minute")).agg(
            *flows, pl.col("amt").filter(pl.col("buy")).sum().alias("b_amt"))
            .with_columns(pl.lit(vname(v)).alias("variant")))
    return pl.concat(out).with_columns(pl.col("minute").cast(pl.Int64))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combos", default="ZZXC:q:vol,ZZXC:n:amt,ZZTS:q:b_amt", help="family:V:basic,... (basic: vol / amt / b_amt)")
    ap.add_argument("--days", required=True, help="PM days to score")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    N, days = a.n, a.days.split(",")
    combos = [c.split(":") for c in a.combos.split(",")]
    cells = {}
    for fam, V, _ in combos:
        for d in days:
            f = a.cache / f"{fam}_{d}.parquet"
            if f.exists():
                pm, vals, _ = pm_frame(f, None, None, None, None, None)
                cells[fam, V, d] = (pm.filter(pl.col("V") == V).select(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64), pl.col(vals[0]).alias("y"))
                                    .filter(pl.col("y").is_finite()))
    if not cells:
        sys.exit("no PM files")
    tdays = trading_days(a.hstats, combos[0][0])
    need = sorted({x for d in days if d in tdays for x in tdays[tdays.index(d) - N + 1: tdays.index(d) + 1]})
    codes = pl.concat([c.select("code") for c in cells.values()]).unique()
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET threads={a.threads}")
    con.register("codes", codes)
    a.out.mkdir(parents=True, exist_ok=True)
    tabs = {}
    for x in need:
        p = a.out / f"date={x}.parquet"
        if not p.exists():
            variants(raw_day(con, x)).write_parquet(p)
        tabs[x] = pl.read_parquet(p)
    print(f"{len(need)} trading days {need[0]}..{need[-1]}", flush=True)
    for fam, V, col in combos:
        rows, bars = [], {}
        for v in map(vname, VARIANTS):
            parts = []
            for d in days:
                if (fam, V, d) not in cells or d not in tdays:
                    continue
                c = cells[fam, V, d]
                win = tdays[tdays.index(d) - N + 1: tdays.index(d) + 1]
                stack = np.stack([c.join(tabs[x].filter(pl.col("variant") == v).select("code", "minute", col), on=["code", "minute"], how="left")[col]
                                  .fill_null(0.0).to_numpy() for x in win])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    h, _ = hit(c["y"].to_numpy(), np.median(stack, axis=0))
                if h.size:
                    parts.append(c.select("code", "minute").with_columns(pl.Series("hit", h)))
            if not parts:
                continue
            t = pl.concat(parts).filter(pl.col("hit").is_not_nan()).with_columns((pl.col("code") >= 600000).alias("sh"))
            rows.append({"variant": v, "all": t["hit"].mean(), "SH": t.filter(pl.col("sh"))["hit"].mean(), "SZ": t.filter(~pl.col("sh"))["hit"].mean()})
            bars[v] = t.group_by("minute").agg(pl.col("hit").filter(pl.col("sh")).mean().alias("SH"), pl.col("hit").filter(~pl.col("sh")).mean().alias("SZ")).sort("minute")
        if not rows:
            print(f"{fam} {V}: nothing scored")
            continue
        r = pl.DataFrame(rows).sort("all", descending=True)
        top = r.head(6)
        if CURRENT not in top["variant"].to_list():
            top = pl.concat([top, r.filter(pl.col("variant") == CURRENT)])
        print(f"== {fam} {V} ~ median_{N}({col}), exact by variant (lab dir open close; current = {CURRENT})")
        print("  " + "\n  ".join(f"{v:<10} all {x:.3f}  SH {s:.3f}  SZ {z:.3f}" for v, x, s, z in top.iter_rows()))
        b = bars[r["variant"][0]]
        worst = b.with_columns(pl.min_horizontal("SH", "SZ").alias("w")).sort("w").head(5)
        print(f"  best {r['variant'][0]} worst bars " + " ".join(f"{m // 60:02d}{m % 60:02d} SH {s:.2f} SZ {z:.2f}" for m, s, z, _ in worst.iter_rows()))


if __name__ == "__main__":
    main()
