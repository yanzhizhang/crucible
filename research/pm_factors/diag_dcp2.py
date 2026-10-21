r"""Stage 4: pin down three parts of PM's day-end state (``h/HS300/p1_intrading_dcp``) exactly.

A. Price lists ``zzxo.r.0..29`` / ``zzui.r.0..9``: trade prices of the last ~5 minutes of continuous
   trading (``r.29`` matched 14:57, ``r.0`` ~14:52). Tested as samples on a fixed grid: value of key
   ``r.k`` = last price at or before ``end - (N-1-k) * step`` (or ``end - k * step``), from snapshots
   and from trades, over a grid of steps and end times; scored by the exact share.
B. ``zzqj.tpv`` / ``.v`` (and ``zzvm.c.*``): turnover / volume, 0.5 % off the all-day totals --
   session-window variants from trades and from snapshot totals.
C. Day-to-day (``--prev``, the previous trading day's file): how far each ``.t`` moves (the clock
   unit), how each ``.wgt`` changes, and how much every histogram's total grows per day.

    python research/pm_factors/diag_dcp2.py --date 20260429 \
        --dcp /data/prod/h/HS300/p1_intrading_dcp/20260429.jz \
        --prev /data/prod/h/HS300/p1_intrading_dcp/20261012.jz --next /data/prod/h/HS300/p1_intrading_dcp/20261013.jz
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

CATALOG = Path("/work/crucible_data/catalog.duckdb")
NS = 1_000_000_000
LOCAL = 8 * 3600
warnings.filterwarnings("ignore", message="Sortedness of columns")
STEPS = (1, 2, 3, 5, 6, 10, 12, 15, 20, 30, 60)
ENDS = {"14:56:57": 53817, "14:57:00": 53820, "14:56:59.999": 53819.999, "15:00:00": 54000}


def load(path: Path) -> dict:
    """Load one dcp file as {sid: {key: value}}."""
    with gzip.open(path) as f:
        return json.loads(f.read())


def part_a(d: dict, con: duckdb.DuckDBPyConnection, date: int) -> None:
    """Price lists as samples on a time grid."""
    for fam, n in (("zzxo", 30), ("zzui", 10)):
        pm = pl.DataFrame([(int(s), k, kv.get(f"{fam}.r.{k}")) for s, kv in d.items() for k in range(n)],
                          schema={"sid": pl.Int64, "k": pl.Int64, "pm": pl.Float64}, orient="row").drop_nulls()
        if pm.is_empty():
            print(f"\n== A. {fam}: no keys in this file")
            continue
        sids = pl.DataFrame({"sid": pm["sid"].unique()})
        con.register("want", sids)
        src = {
            "snap": con.sql(f"""SELECT symbol_id AS sid, ((time / {NS}) + {LOCAL}) % 86400 AS ts, price FROM raw_quotation q
                               JOIN want w ON w.sid = q.symbol_id WHERE date = {date} AND price > 0""").pl(),
            "trade": con.sql(f"""SELECT symbol_id AS sid, ((time / {NS}) + {LOCAL}) % 86400 AS ts, price FROM raw_transaction r
                                JOIN want w ON w.sid = r.symbol_id WHERE date = {date} AND trade_type = 1""").pl(),
        }
        rows = []
        for sname, s in src.items():
            s = s.with_columns(pl.col("sid").cast(pl.Int64), pl.col("ts").cast(pl.Float64)).sort("ts")
            for step in STEPS:
                for ename, end in ENDS.items():
                    for order in ("last_is_end", "first_is_end"):
                        off = (n - 1 - pl.col("k")) if order == "last_is_end" else pl.col("k")
                        tg = pm.with_columns((end - off * step).cast(pl.Float64).alias("ts")).sort("ts")
                        j = tg.join_asof(s, on="ts", by="sid", strategy="backward")
                        ex = j.select(((pl.col("pm") - pl.col("price")).abs() < 1e-6).mean()).item()
                        rows.append({"family": fam, "source": sname, "step_s": step, "end": ename, "order": order, "exact": ex})
        res = pl.DataFrame(rows).sort("exact", descending=True)
        print(f"\n== A. {fam}.r.0..{n - 1}: best sampling grids (exact = share of stock x k cells equal)")
        print(res.head(10))


def part_b(d: dict, con: duckdb.DuckDBPyConnection, date: int) -> None:
    """Turnover / volume window variants."""
    ks = [k for k in ("zzqj.tpv", "zzqj.v", "zzvm.c.tpv", "zzvm.c.v") if any(kv.get(k) is not None for kv in d.values())]
    pm = pl.DataFrame([{"sid": int(s), **{k: kv.get(k) for k in ks}} for s, kv in d.items()],
                      schema={"sid": pl.Int64, **dict.fromkeys(ks, pl.Float64)})
    t = f"(((time // {NS}) + {LOCAL}) % 86400)"
    tr = con.sql(f"""SELECT symbol_id AS sid,
        sum(volume * price) AS amt_all, sum(volume)::DOUBLE AS vol_all,
        sum(volume * price) FILTER ({t} >= 34200) AS amt_no_open, (sum(volume) FILTER ({t} >= 34200))::DOUBLE AS vol_no_open,
        sum(volume * price) FILTER ({t} < 53820) AS amt_no_close, (sum(volume) FILTER ({t} < 53820))::DOUBLE AS vol_no_close,
        sum(volume * price) FILTER ({t} >= 34200 AND {t} < 53820) AS amt_cont, (sum(volume) FILTER ({t} >= 34200 AND {t} < 53820))::DOUBLE AS vol_cont
        FROM raw_transaction WHERE date = {date} AND trade_type = 1 GROUP BY ALL""").pl()
    sn = con.sql(f"""SELECT symbol_id AS sid,
        arg_max(total_amount, time) AS amt_snap_last, arg_max(total_volume, time)::DOUBLE AS vol_snap_last,
        arg_max(total_amount, time) FILTER ({t} < 53820) AS amt_snap_1457, (arg_max(total_volume, time) FILTER ({t} < 53820))::DOUBLE AS vol_snap_1457
        FROM raw_quotation WHERE date = {date} AND price > 0 GROUP BY ALL""").pl()
    j = pm.join(tr, on="sid", how="left").join(sn, on="sid", how="left")
    print("\n== B. turnover / volume: share exactly equal (1e-6 relative) and median relative error")
    for k in ks:
        for c in [c for c in j.columns if c != "sid" and c not in ks and (("amt" in c) == ("tpv" in k))]:
            r = j.select(((pl.col(k) - pl.col(c)).abs() <= 1e-6 * pl.col(k).abs()).mean().alias("exact"),
                         (pl.col(k) / pl.col(c) - 1).abs().median().alias("mre")).row(0)
            print(f"  {k:12s} vs {c:16s} exact {r[0]:.4f}  mre {r[1]:.6f}")


def part_c(prev: dict, nxt: dict) -> None:
    """Day-to-day: clock step of .t, change of .wgt, growth of histogram totals."""
    sids = sorted(set(prev) & set(nxt))
    keys = set(nxt[sids[0]])
    rows = []
    for k in sorted(x for x in keys if x.endswith((".t", ".wgt"))):
        a = pl.DataFrame({"p": [prev[s].get(k) for s in sids], "n": [nxt[s].get(k) for s in sids]}).drop_nulls()
        a = a.filter((pl.col("p") != 0) & (pl.col("n") != 0))
        if a.height:
            rows.append({"key": k, "stocks": a.height, "diff_median": (a["n"] - a["p"]).median(), "diff_min": (a["n"] - a["p"]).min(),
                         "diff_max": (a["n"] - a["p"]).max(), "ratio_median": (a["n"] / a["p"]).median()})
    hist: dict[str, list] = {}
    for k in keys:
        m = re.match(r"(.+)\.c\d*\.\d+$", k)
        if m:
            hist.setdefault(m.group(1), []).append(k)
    hrows = []
    for p, ks in sorted(hist.items()):
        tp = pl.Series([sum(prev[s].get(k) or 0 for k in ks) for s in sids])
        tn = pl.Series([sum(nxt[s].get(k) or 0 for k in ks) for s in sids])
        hrows.append({"histogram": p, "bins": len(ks), "total_prev_median": tp.median(), "growth_median": (tn - tp).median(),
                      "growth_min": (tn - tp).min(), "growth_max": (tn - tp).max(), "ratio_median": (tn / tp).median()})
    with pl.Config(tbl_rows=120, tbl_width_chars=220, fmt_float="mixed", fmt_str_lengths=30):
        print(f"\n== C. day-to-day over {len(sids)} stocks: .t and .wgt (next - prev, next / prev)")
        print(pl.DataFrame(rows))
        print("\n== C. histogram totals: growth per day")
        print(pl.DataFrame(hrows))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="trade day of --dcp (its ticks must be decoded)")
    ap.add_argument("--dcp", type=Path, required=True)
    ap.add_argument("--prev", type=Path, help="a state file of trading day D-1 (with --next = day D)")
    ap.add_argument("--next", type=Path)
    a = ap.parse_args()
    d = load(a.dcp)
    con = duckdb.connect(str(CATALOG), read_only=True)
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=4):
        part_a(d, con, int(a.date))
        part_b(d, con, int(a.date))
    if a.prev and a.next:
        part_c(load(a.prev), load(a.next))


if __name__ == "__main__":
    main()
