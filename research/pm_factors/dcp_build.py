r"""Stage 4: rebuild PM's day-end state fields from our ticks, and score them key by key against PM.

Every rule here was confirmed on 9 trade days (0429, 0922-0930, 1008, 1009) by ``diag_dcp*`` /
``dcp_multiday`` and earlier runs of this script (docs/PM_REPRO.md, "day-end state"). PM's live
process loses SSE trades (SZSE is complete), so keys are scored per exchange and SZSE is the
reference.

Built keys (``build``):

* day prices from the trades: ``zzpk.high/low/open``, ``zzvx.high/low``, ``zzxu.high/low``,
  ``zzxu.last`` / ``zzxo.last`` / ``cdc.lp`` (last trade);
* ``zzmh.hm``: highest snapshot last price over 09:30-14:57 exchange time (misses highs that fall
  between snapshots, so it is not the trade high);
* ``zzqj.v`` / ``zzqj.tpv`` / ``zzvm.c.v`` / ``zzvm.c.tpv``: volume / amount of all trades from
  09:30:00 exchange time (close auction included);
* ``zzxo.r.0..29`` / ``zzui.r.0..9``: last snapshot received (local arrival) at or before
  14:57:00 - (n-1-k) * 10 s;
* ``clock.fast``: local arrival (trading clock) of the last snapshot received by PM's process stop;
  ``clock.slow``: local arrival of the last trade (live days PM is 3-7 ms earlier: its trade feed
  is faster than feitu's capture, not a different rule). Trade keys only count trades received by
  the stop (repos and bonds trade 15:00-15:30, after PM stopped);
* ``zzgh.nb`` (best candidate so far, not exact): continuous-session amount imbalance.

``--stop auto`` (default) takes each day's stop from the latest ``zzba.p.t`` in its state file.
A key written ``base|variant`` is scored against PM's ``base`` key, for trying candidate rules.
``compare`` also matches each clock against every ``*.t`` key of the state file; the printout
keeps the ``*.t`` keys the plain clock matches and shows one row per clock.

    python research/pm_factors/dcp_build.py --dates 20260429,20260922,20261009
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import warnings
from pathlib import Path

import duckdb
import polars as pl

warnings.filterwarnings("ignore", message="Sortedness of columns")
CATALOG = Path("/work/crucible_data/catalog.duckdb")
DCP = Path("/data/prod/h/HS300/p1_intrading_dcp")
SOD = "((time / 1e9 + 28800) % 86400)"
LSOD = "((spider_ts / 1e9 + 28800) % 86400)"
CLOCK_TOL = 0.01  # seconds; PM's receive clock differs from ours by a few ms


def tc(s: str) -> str:
    """Trading clock: seconds since 09:30 with the lunch break removed."""
    return f"CASE WHEN {s} <= 41400 THEN {s} - 34200 WHEN {s} < 46800 THEN 7200 ELSE {s} - 46800 + 7200 END"


def build(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop_sod: float) -> pl.DataFrame:
    """One row per stock with every rebuilt key."""
    con.register("want", pl.DataFrame({"sid": sids}))
    cont = f"{SOD} >= 34200 AND {SOD} < 53820"
    tr = con.sql(f"""SELECT symbol_id AS sid, any_value(market_id) AS mkt,
        max(price) AS high, min(price) AS low, arg_min(price, time) AS open, arg_max(price, time) AS last,
        (sum(volume) FILTER ({SOD} >= 34200))::DOUBLE AS v, sum(volume * price) FILTER ({SOD} >= 34200) AS tpv,
        (sum(volume * price) FILTER (dir = 1063 AND {cont}) - sum(volume * price) FILTER (dir = 1550 AND {cont}))
         / sum(volume * price) FILTER ({cont}) AS nb,
        max({tc(LSOD)}) AS clock_slow
        FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id
        WHERE r.date = {date} AND r.trade_type = 1 AND {LSOD} <= {stop_sod}  -- PM never saw trades after its stop
        GROUP BY 1""").pl()
    snap = con.sql(f"""SELECT symbol_id AS sid, max({tc(LSOD)}) FILTER ({LSOD} <= {stop_sod}) AS clock_fast,
                       max(price) FILTER (price > 0 AND {cont}) AS hm
                       FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} GROUP BY 1""").pl()
    sn = con.sql(f"""SELECT symbol_id AS sid, {LSOD} AS g, price FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id
                     WHERE q.date = {date} AND price > 0 AND {SOD} BETWEEN 53300 AND 53900""").pl().with_columns(
        pl.col("sid").cast(pl.Int64), pl.col("g").cast(pl.Float64)).sort("g")
    lists = []
    for fam, n in (("zzxo", 30), ("zzui", 10)):  # grid points 10 s apart ending at 14:57:00, by local arrival
        g = pl.DataFrame({"sid": sids}).join(pl.DataFrame({"k": list(range(n))}), how="cross").with_columns(
            (53820 - (n - 1 - pl.col("k")) * 10).cast(pl.Float64).alias("g")).sort("g")
        g = g.join_asof(sn, on="g", by="sid", strategy="backward").with_columns(pl.format(f"{fam}.r.{{}}", "k").alias("key"))
        lists.append(g.select("sid", "key", "price"))
    wide_lists = pl.concat(lists).pivot(on="key", index="sid", values="price")
    con.unregister("want")
    out = tr.with_columns(pl.col("sid").cast(pl.Int64)).join(snap.with_columns(pl.col("sid").cast(pl.Int64)), on="sid", how="left")
    out = out.select(
        "sid", "mkt",
        *[pl.col(c).alias(k) for c, ks in (("high", ("zzpk.high", "zzvx.high", "zzxu.high")),
                                            ("low", ("zzpk.low", "zzvx.low", "zzxu.low")), ("open", ("zzpk.open",)),
                                            ("last", ("zzxu.last", "zzxo.last", "cdc.lp")), ("hm", ("zzmh.hm",)),
                                            ("v", ("zzqj.v", "zzvm.c.v")), ("tpv", ("zzqj.tpv", "zzvm.c.tpv")),
                                            ("nb", ("zzgh.nb",)), ("clock_fast", ("clock.fast",)),
                                            ("clock_slow", ("clock.slow",))) for k in ks])
    return out.join(wide_lists, on="sid", how="left")


def load_state(path: Path) -> pl.DataFrame:
    """Load PM's state file as a long frame (sid, key, pm), histogram bins dropped."""
    with gzip.open(path) as f:
        d = json.loads(f.read())
    bins = re.compile(r"\.c\.\d+$")  # histogram bins only; zzvm.c.v and the like stay
    rows = [(int(s), k, float(v)) for s, kv in d.items() for k, v in kv.items() if v is not None and not bins.search(k)]
    return pl.DataFrame(rows, schema={"sid": pl.Int64, "key": pl.String, "pm": pl.Float64}, orient="row")


def compare(ours: pl.DataFrame, pm: pl.DataFrame, date: int) -> pl.DataFrame:
    """Per key: exact share overall and per exchange; clocks also against every PM ``*.t`` key."""
    keys = [c for c in ours.columns if c not in ("sid", "mkt")]
    lo = ours.unpivot(index=["sid", "mkt"], on=keys, variable_name="key", value_name="ours").with_columns(
        pl.col("key").str.replace(r"\|.*$", "").alias("pmkey"))  # base|variant is scored against PM's base
    j = lo.filter(~pl.col("key").str.starts_with("clock.")).join(pm.rename({"key": "pmkey"}), on=["sid", "pmkey"])
    ok = (pl.col("pm") - pl.col("ours")).abs() <= pl.max_horizontal(pl.lit(1e-6), 1e-9 * pl.col("pm").abs())
    rows = j.group_by("key").agg(pl.len().alias("n"), ok.mean().alias("all"), ok.filter(pl.col("mkt") == 3553).mean().alias("SSE"),
                                 ok.filter(pl.col("mkt") == 3554).mean().alias("SZSE"),
                                 ((pl.col("pm") - pl.col("ours")) / pl.col("ours")).abs().median().alias("mre"))
    clocks = []
    tkeys = pm.filter(pl.col("key").str.ends_with(".t"))
    for ck in [k for k in keys if k.startswith("clock.")]:
        c = lo.filter(pl.col("key") == ck).select("sid", "mkt", "ours").join(tkeys, on="sid")
        okc = (pl.col("pm") - pl.col("ours")).abs() <= CLOCK_TOL
        clocks.append(c.group_by("key").agg(pl.len().alias("n"), okc.mean().alias("all"),
                                            okc.filter(pl.col("mkt") == 3553).mean().alias("SSE"),
                                            okc.filter(pl.col("mkt") == 3554).mean().alias("SZSE"),
                                            (pl.col("pm") - pl.col("ours")).median().alias("mre"))
                      .with_columns(pl.format(f"{ck} ~ {{}}", "key").alias("key")))
    return pl.concat([rows, *clocks]).with_columns(pl.lit(date).alias("date"))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=DCP)
    ap.add_argument("--stop", default="auto",
                    help="PM process stop, local time; 'auto' = the latest zzba.p.t in that day's state file")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    a = ap.parse_args()
    fixed = None if a.stop == "auto" else int(a.stop[:2]) * 3600 + int(a.stop[3:5]) * 60 + float(a.stop[6:])
    con = duckdb.connect(str(CATALOG), read_only=True)
    res, built = [], []
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        f = a.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pm = load_state(f)
        stop = fixed
        if stop is None:  # every stock's last snapshot precedes the stop, so the latest one is the stop (+5 ms for clock skew)
            t = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
            stop = (t + 34200 if t <= 7200 else t - 7200 + 46800) + 0.005
            print(f"{d}: stop {int(stop // 3600):02d}:{int(stop % 3600 // 60):02d}:{stop % 60:06.3f}")
        ours = build(con, int(d), sorted(pm["sid"].unique().to_list()), stop)
        if ours.is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        built.append(ours.with_columns(pl.lit(int(d)).alias("date")))
        res.append(compare(ours, pm, int(d)))
        print(f"done {d}")
    if not res:
        return
    r = pl.concat(res)
    a.out.mkdir(parents=True, exist_ok=True)
    r.write_csv(a.out / "dcp_build_scores.csv")  # every key, before the print filter
    pl.concat(built, how="diagonal").write_parquet(a.out / "dcp_build.parquet")
    # clock rows: keep the *.t keys the plain clock matches on some day, then one row per clock variant
    ck = r.filter(pl.col("key").str.starts_with("clock.")).with_columns(
        pl.col("key").str.extract(r"^([^ |]+)").alias("base"), pl.col("key").str.extract(r" ~ (.+)$").alias("tk"))
    good = ck.filter(pl.col("key") == pl.col("base") + " ~ " + pl.col("tk")).group_by("base", "tk").agg(
        pl.col("all").max().alias("best")).filter(pl.col("best") >= 0.5)
    for b, tks in good.group_by("base").agg(pl.col("tk").sort()).sort("base").iter_rows():
        print(f"{b} matches {len(tks)} keys: {' '.join(tks)}")
    ck = ck.join(good.select("base", "tk"), on=["base", "tk"]).with_columns(
        pl.col("key").str.replace(r" ~ .+$", " ~ *.t (mean)"))
    r = pl.concat([r.filter(~pl.col("key").str.starts_with("clock.")), ck.select(r.columns)])
    # the price lists print as one row per rule (mean over their 30 / 10 points)
    r = r.with_columns(pl.col("key").str.replace(r"^(zzxo|zzui)\.r\.\d+(\|.*)?$", "$1.r.*$2 (mean)")).group_by("key", "date").agg(
        pl.col("n").sum(), pl.col("all", "SSE", "SZSE", "mre").mean())
    with pl.Config(tbl_rows=120, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=45, float_precision=4):
        print("\n== SZSE exact share per key and day (SZSE is complete in PM's feed; clocks within 10 ms)")
        print(r.pivot(on="date", index="key", values="SZSE", sort_columns=True).sort("key"))
        print("\n== SSE exact share per key and day (PM loses SSE trades: volume keys short)")
        print(r.pivot(on="date", index="key", values="SSE", sort_columns=True).sort("key"))
        print("\n== median relative error (clock rows: median PM - ours, s)")
        print(r.pivot(on="date", index="key", values="mre", sort_columns=True).sort("key"))
    print(f"scores: {a.out / 'dcp_build_scores.csv'}, built values: {a.out / 'dcp_build.parquet'}")


if __name__ == "__main__":
    main()
