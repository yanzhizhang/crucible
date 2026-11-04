r"""Which bar / price definition do PM's rolling-median columns use? Candidates rebuilt from the raw ticks, scored by exchange.

``window_probe.py`` put the trade columns (ZZXC q / n, ZZTS q ~ median_22 of vol / amt / b_amt) at 0.96 bit-exact on
SZSE but 0.57 on SSE, and ZZDS h / l at 1.000 SZSE vs 0.985 SSE -- a definition, not a data day. Per trading day this
rebuilds from ``raw_transaction`` / ``raw_quotation``:

* bar flows per (stock, 5-minute bar) under ``lab dir open close`` variants: lab c = right-closed (t-1min, t] like
  ``basics.py``, f = left-closed; dir bs = sided fills only (``basics.py``: drops SSE auction fills, side N), all = every
  fill; open / close 1 = that auction folded into the 09:35 / 15:00 bar;
* the same with SSE trade times moved (``sh_s`` floored to the second, ``sh+500`` = +500 ms, ...), and ``snap`` = the bar's
  change of the snapshot running totals (no b_amt);
* daily prices for ZZDS: fills (all / sided / continuous only), snapshot open / high / low / last and pre-close.

Each candidate's window median (last N trading days, d included) is scored against PM by exchange. Day tables are cached
under ``store/pm_factors/bar_variants/date=<D>/``.

    python research/pm_factors/bar_variants.py --days 20261020,20261021,20261022,20261023,20261026,20261027
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Callable
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
NS_SEC, NS_MIN, NS_DAY, NS_LOCAL = 1_000_000_000, 60_000_000_000, 86_400_000_000_000, 8 * 3600 * 1_000_000_000
VARIANTS = [dict(zip(("lab", "dir", "open", "close"), v)) for v in product("cf", ("bs", "all"), (1, 0), (1, 0))]
CURRENT = "c bs 1 1"  # what basics.py builds
SH = "symbol_id >= 600000"
SHIFTS = {"sh_s": f"CASE WHEN {SH} THEN time - time % {NS_SEC} ELSE time END",
          **{f"sh{k:+d}": f"CASE WHEN {SH} THEN time + {k * 1_000_000} ELSE time END" for k in (-1000, -500, -200, 200, 500, 1000)}}
SHIFT_BASE = {"lab": "c", "dir": "bs", "open": 0, "close": 1}  # the shifted times are tried on this variant
PRICES = ("high_all", "low_all", "high_bs", "low_bs", "open_all", "open_cont", "close_all", "close_cont", "close_bs", "vwap_all",
          "snap_open", "snap_high", "snap_low", "snap_last", "snap_last_cont", "snap_preclose")


def vname(v: dict) -> str:
    """Short variant label: lab dir open close."""
    return f"{v['lab']} {v['dir']} {v['open']} {v['close']}"


def minute(t: str) -> str:
    """Local minute of day (fractional) of an exchange ns timestamp."""
    return f"((({t} + {NS_LOCAL}) % {NS_DAY}) / {NS_MIN}::DOUBLE)"


def bar(x: str) -> str:
    """Minute label -> 5-minute bar end; auctions to 09:35 / 15:00, an 11:30:00 / 13:00:00 print to its session."""
    x = f"(CASE WHEN m < 570 THEN 571 WHEN m > 897 THEN 900 WHEN {x} > 690 AND {x} <= 780 THEN (CASE WHEN m < 700 THEN 690 ELSE 781 END) ELSE {x} END)"
    return f"(CASE WHEN {x} >= 896 THEN 900 ELSE (({x} - 571) // 5) * 5 + 575 END)"


def raw_day(con: duckdb.DuckDBPyConnection, d: str, t: str = "time") -> pl.DataFrame:
    """Sum volume and amount per (code, phase, sided, buy, bar_c, bar_f), trade time taken as ``t``."""
    return con.sql(f"""
        WITH r AS (
          SELECT symbol_id::BIGINT AS code, dir, volume::DOUBLE AS vol, volume * price AS amt, {minute(t)} AS m
          FROM raw_transaction WHERE date = {int(d)} AND trade_type = 1 AND symbol_id IN (SELECT code FROM codes)),
        b AS (SELECT *, CAST(ceil(m) AS INTEGER) AS mc, CAST(floor(m) AS INTEGER) + 1 AS mf FROM r)
        SELECT code, CASE WHEN m < 570 THEN 'open' WHEN m > 897 THEN 'close' ELSE 'cont' END AS phase,
               dir IN ({BUY}, {SELL}) AS sided, dir = {BUY} AS buy, {bar('mc')} AS bar_c, {bar('mf')} AS bar_f,
               sum(vol) AS vol, sum(amt) AS amt
        FROM b GROUP BY ALL""").pl()


def variants(raw: pl.DataFrame, vs: list[dict], tag: str = "") -> pl.DataFrame:
    """Long table (code, minute, variant, vol, amt, b_amt) for the given variants."""
    out = []
    for v in vs:
        keep = (pl.col("phase") == "cont") | ((pl.col("phase") == "open") & bool(v["open"])) | ((pl.col("phase") == "close") & bool(v["close"]))
        flows = [pl.col("vol").filter(pl.col("sided")).sum(), pl.col("amt").filter(pl.col("sided")).sum()] if v["dir"] == "bs" else [pl.col("vol").sum(), pl.col("amt").sum()]
        out.append(raw.filter(keep).group_by("code", pl.col(f"bar_{v['lab']}").alias("minute")).agg(
            *flows, pl.col("amt").filter(pl.col("buy")).sum().alias("b_amt"))
            .with_columns(pl.lit(vname(v) + tag).alias("variant")))
    return pl.concat(out).with_columns(pl.col("minute").cast(pl.Int64))


def snap_day(con: duckdb.DuckDBPyConnection, d: str) -> pl.DataFrame:
    """Bar flows from the snapshot running totals: last total in the bar minus the last before it (the open auction is in the base)."""
    t = con.sql(f"""
        WITH q AS (SELECT symbol_id::BIGINT AS code, time, {minute('time')} AS m, total_volume::DOUBLE AS tv, total_amount::DOUBLE AS ta
                   FROM raw_quotation WHERE date = {int(d)} AND symbol_id IN (SELECT code FROM codes) AND total_volume > 0),
        b AS (SELECT *, CASE WHEN m <= 570 THEN 570 ELSE {bar('CAST(ceil(m) AS INTEGER)')} END AS minute FROM q)
        SELECT code, minute, arg_max(tv, time) AS tv, arg_max(ta, time) AS ta FROM b GROUP BY ALL""").pl().sort("code", "minute")
    t = t.with_columns((pl.col("tv") - pl.col("tv").shift(1).over("code")).alias("vol"), (pl.col("ta") - pl.col("ta").shift(1).over("code")).alias("amt"))
    return t.filter(pl.col("minute") > 570).select("code", pl.col("minute").cast(pl.Int64), "vol", "amt", pl.lit(None, pl.Float64).alias("b_amt"),
                                                    pl.lit("snap").alias("variant"))


def price_day(con: duckdb.DuckDBPyConnection, d: str) -> pl.DataFrame:
    """Daily price candidates per code from the fills and the snapshots."""
    tr = con.sql(f"""
        WITH r AS (SELECT symbol_id::BIGINT AS code, dir IN ({BUY}, {SELL}) AS sided, price, volume::DOUBLE AS vol, time, {minute('time')} AS m
                   FROM raw_transaction WHERE date = {int(d)} AND trade_type = 1 AND price > 0 AND symbol_id IN (SELECT code FROM codes))
        SELECT code, max(price) AS high_all, min(price) AS low_all, max(price) FILTER (sided) AS high_bs, min(price) FILTER (sided) AS low_bs,
               arg_min(price, time) AS open_all, arg_min(price, time) FILTER (m >= 570) AS open_cont, arg_max(price, time) AS close_all,
               arg_max(price, time) FILTER (m <= 897) AS close_cont, arg_max(price, time) FILTER (sided) AS close_bs,
               sum(price * vol) / sum(vol) AS vwap_all
        FROM r GROUP BY code""").pl()
    sn = con.sql(f"""
        WITH q AS (SELECT symbol_id::BIGINT AS code, time, {minute('time')} AS m, open, high, low, price, pre_close
                   FROM raw_quotation WHERE date = {int(d)} AND symbol_id IN (SELECT code FROM codes))
        SELECT code, arg_max(open, time) FILTER (open > 0) AS snap_open, arg_max(high, time) FILTER (high > 0) AS snap_high,
               arg_max(low, time) FILTER (low > 0) AS snap_low, arg_max(price, time) FILTER (price > 0) AS snap_last,
               arg_max(price, time) FILTER (price > 0 AND m <= 897) AS snap_last_cont, arg_max(pre_close, time) FILTER (pre_close > 0) AS snap_preclose
        FROM q GROUP BY code""").pl()
    return tr.join(sn, on="code", how="full", coalesce=True).with_columns(pl.col(c).cast(pl.Float64) for c in PRICES)


def cached(p: Path, build: Callable[[], pl.DataFrame]) -> pl.DataFrame:
    """Read ``p``, building and writing it first when missing."""
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        build().write_parquet(p)
    return pl.read_parquet(p)


def score(cells: dict, tdays: list[str], N: int, value: Callable[[str, pl.DataFrame], np.ndarray]) -> tuple[pl.DataFrame | None, float]:
    """Per-cell exact flags of median_N(value(day)) against PM over the scored days; value(day) -> array aligned to the cells."""
    parts = []
    for d, c in cells.items():
        win = tdays[tdays.index(d) - N + 1: tdays.index(d) + 1]
        stack = np.stack([value(x, c) for x in win])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            h, _ = hit(c["y"].to_numpy(), np.nanmedian(stack, axis=0))
        if h.size:
            parts.append(c.drop("y").with_columns(pl.Series("hit", h)))
    if not parts:
        return None, np.nan
    t = pl.concat(parts).filter(pl.col("hit").is_not_nan()).with_columns((pl.col("code") >= 600000).alias("sh"))
    return t, t["hit"].mean()


def by_exch(t: pl.DataFrame) -> tuple[float, float]:
    """Exact share on SSE and on SZSE."""
    return t.filter(pl.col("sh"))["hit"].mean(), t.filter(~pl.col("sh"))["hit"].mean()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combos", default="ZZXC:q:vol,ZZXC:n:amt,ZZTS:q:b_amt", help="bar family:V:basic,... (basic: vol / amt / b_amt)")
    ap.add_argument("--daily", default="ZZDS:h,ZZDS:l,ZZDS:c,ZZDS:o", help="daily family:V,... scored against every price candidate")
    ap.add_argument("--days", required=True, help="PM days to score")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    N, days = a.n, a.days.split(",")
    jobs = [(fam, V, col) for fam, V, col in (c.split(":") for c in a.combos.split(",") if c)]
    jobs += [(fam, V, None) for fam, V in (c.split(":") for c in a.daily.split(",") if c)]
    tdays = trading_days(a.hstats, jobs[0][0])
    cells = {}
    for fam, V, col in jobs:
        for d in days:
            f = a.cache / f"{fam}_{d}.parquet"
            if f.exists() and d in tdays:
                pm, vals, _ = pm_frame(f, None, None, None, None, None)
                keys = ["code", "minute"] if col else ["code"]
                cells.setdefault((fam, V), {})[d] = (pm.filter(pl.col("V") == V).select(*[pl.col(k).cast(pl.Int64) for k in keys], pl.col(vals[0]).alias("y"))
                                                     .filter(pl.col("y").is_finite()))
    if not cells:
        sys.exit("no PM files")
    need = sorted({x for d in days if d in tdays for x in tdays[tdays.index(d) - N + 1: tdays.index(d) + 1]})
    codes = pl.concat([c.select("code") for dd in cells.values() for c in dd.values()]).unique()
    con = duckdb.connect(str(CATALOG), read_only=True)
    con.execute(f"SET threads={a.threads}")
    con.register("codes", codes)
    bars, prices = {}, {}
    for x in need:
        dst = a.out / f"date={x}"
        parts = [cached(dst / "base.parquet", lambda x=x: variants(raw_day(con, x), VARIANTS)), cached(dst / "snap.parquet", lambda x=x: snap_day(con, x))]
        parts += [cached(dst / f"{k}.parquet", lambda e=e, k=k, x=x: variants(raw_day(con, x, e), [SHIFT_BASE], f" {k}")) for k, e in SHIFTS.items()]
        bars[x] = pl.concat(parts, how="vertical_relaxed")
        prices[x] = cached(dst / "prices.parquet", lambda x=x: price_day(con, x))
        print(f"\r{x} cached", end="", flush=True)
    print(f"\r{len(need)} trading days {need[0]}..{need[-1]}", flush=True)
    names = bars[need[-1]]["variant"].unique(maintain_order=True).to_list()
    for fam, V, col in jobs:
        if (fam, V) not in cells:
            print(f"{fam} {V}: no PM file")
            continue
        cc = cells[fam, V]
        if col:
            def val(x: str, c: pl.DataFrame, v: str, col: str = col) -> np.ndarray:
                return c.join(bars[x].filter(pl.col("variant") == v).select("code", "minute", col), on=["code", "minute"], how="left")[col].fill_null(0.0).to_numpy()
            cands = [(v, lambda x, c, v=v: val(x, c, v)) for v in names if not (col == "b_amt" and v == "snap")]
            title = f"{fam} {V} ~ median_{N}({col}) by bar variant (lab dir open close [SSE time]; current = {CURRENT})"
        else:
            def pval(x: str, c: pl.DataFrame, p: str) -> np.ndarray:
                return c.join(prices[x].select("code", p), on="code", how="left")[p].to_numpy().astype(float)
            cands = [(p, lambda x, c, p=p: pval(x, c, p)) for p in PRICES]
            title = f"{fam} {V} ~ median_{N}(price candidate)"
        rows, tabs = [], {}
        for name, fn in cands:
            t, ex = score(cc, tdays, N, fn)
            if t is not None:
                tabs[name] = t
                rows.append((name, ex, *by_exch(t)))
        if not rows:
            print(f"{fam} {V}: nothing scored")
            continue
        rows.sort(key=lambda r: -r[1])
        top = rows[:6] + [r for r in rows[6:] if r[0] in (CURRENT, "snap") or " sh" in r[0]]
        print(f"== {title}")
        print("  " + "\n  ".join(f"{n:<16} all {x:.3f}  SH {s:.3f}  SZ {z:.3f}" for n, x, s, z in top))
        t = tabs[rows[0][0]]
        if col:
            b = t.group_by("minute").agg(pl.col("hit").filter(pl.col("sh")).mean().alias("SH"), pl.col("hit").filter(~pl.col("sh")).mean().alias("SZ"))
            worst = b.sort(pl.min_horizontal("SH", "SZ")).head(5)
            print(f"  best {rows[0][0]}: worst bars " + " ".join(f"{m // 60:02d}{m % 60:02d} SH {s:.2f} SZ {z:.2f}" for m, s, z in worst.iter_rows()))
        sc = t.group_by("code").agg(pl.col("hit").mean()).sort("hit")
        print(f"  best {rows[0][0]}: {sc.filter(pl.col('hit') == 0).height} stocks never exact of {sc.height}: "
              + " ".join(f"{c:06d}" for c in sc.filter(pl.col("hit") == 0)["code"].head(8)))


if __name__ == "__main__":
    main()
