r"""Stage 4: what drives the state clock ``.t``, and where ``tpv`` / ``v`` differ from turnover per exchange.

``diag_dcp2`` showed (0922 -> 0923): the fast averages share one per-stock ``t`` that moved +8.4 on
median but -48 .. +54 across stocks, the T-variants another that barely moves -- an activity-driven
("business time") or decayed clock, not time of day; and ``zzqj.tpv`` / ``.v`` equal turnover /
volume without the close auction for only 57 % of stocks (one exchange?). This prints:

* the day-to-day change of each distinct clock against the day's per-stock activity (volume, amount,
  trades, aggressor orders, turnover rate vs the previous day, price range) -- rank correlation and a
  least-squares slope -- and the clocks' levels against the same;
* for ``tpv`` / ``v``: exact share per exchange, and for the mismatched stocks which slice of the
  day's trades (open auction, trades stamped exactly 09:30:00 / 14:57:00, the close auction) equals
  PM minus ours.

    python research/pm_factors/diag_dcp3.py --date 20260923 \
        --prev /data/prod/h/HS300/p1_intrading_dcp/20260922.jz --next /data/prod/h/HS300/p1_intrading_dcp/20260923.jz
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import duckdb
import polars as pl

CATALOG = Path("/work/crucible_data/catalog.duckdb")
NS = 1_000_000_000
LOCAL = 8 * 3600
CLOCKS = ("zzba.p.t", "zzbd.pEmaT.t", "zzhj.p.t", "zzui.r.t")
WGTS = ("zzba.p.wgt", "zzab2.r.wgt", "zzbd.pEmaS.wgt", "zzbd.pEmaT.wgt", "zzqj.r.wgt", "zzxu.a.wgt")


def load(path: Path) -> dict:
    """Load one dcp file as {sid: {key: value}}."""
    with gzip.open(path) as f:
        return json.loads(f.read())


def day_stats(con: duckdb.DuckDBPyConnection, dates: list[int]) -> pl.DataFrame:
    """Per (date, sid) activity."""
    dl = ", ".join(str(d) for d in dates)
    return con.sql(f"""
      SELECT date, symbol_id AS sid, any_value(market_id) AS mkt, sum(volume)::DOUBLE AS vol, sum(volume * price) AS amt,
             count(*)::DOUBLE AS n, count(DISTINCT CASE dir WHEN 1063 THEN buy_seq_id ELSE sell_seq_id END)::DOUBLE AS ords,
             ln(max(price) / min(price)) AS range
      FROM raw_transaction WHERE date IN ({dl}) AND trade_type = 1 GROUP BY ALL""").pl()


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="trade day of --next")
    ap.add_argument("--prev", type=Path, required=True)
    ap.add_argument("--next", type=Path, required=True)
    ap.add_argument("--prev-date", help="trade day of --prev (for the turnover-rate ratio), if decoded")
    a = ap.parse_args()
    p, n = load(a.prev), load(a.next)
    sids = sorted(set(p) & set(n))
    con = duckdb.connect(str(CATALOG), read_only=True)
    dates = [int(a.date)] + ([int(a.prev_date)] if a.prev_date else [])
    st = day_stats(con, dates)
    cur = st.filter(pl.col("date") == int(a.date)).drop("date")
    if a.prev_date:
        pv = st.filter(pl.col("date") == int(a.prev_date)).select("sid", pl.col("vol").alias("vol_prev"))
        cur = cur.join(pv, on="sid", how="left").with_columns((pl.col("vol") / pl.col("vol_prev")).alias("vol_vs_prev"))
    rows = []
    for s in sids:
        r = {"sid": int(s)}
        for k in CLOCKS + WGTS:
            a0, a1 = p[s].get(k), n[s].get(k)
            r[f"{k}"] = a1
            r[f"d:{k}"] = None if a0 is None or a1 is None else a1 - a0
            r[f"r:{k}"] = None if not a0 or a1 is None else a1 / a0
        rows.append(r)
    j = pl.DataFrame(rows).join(cur, on="sid", how="left")
    stats = [c for c in ("vol", "amt", "n", "ords", "range", "vol_vs_prev") if c in j.columns]
    out = []
    for target in [f"d:{k}" for k in CLOCKS] + list(CLOCKS) + [f"r:{k}" for k in WGTS]:
        for c in stats:
            for f, name in ((pl.col(c), c), (pl.col(c).log(), f"log {c}")):
                k = j.select(pl.col(target).alias("y"), f.alias("x")).filter(pl.col("x").is_finite() & pl.col("y").is_finite())
                if k.height < 30:
                    continue
                rho = k.select(pl.corr("x", "y", method="spearman")).item()
                slope = k.select(pl.cov("x", "y") / pl.col("x").var()).item()
                out.append({"target": target, "vs": name, "rho": rho, "slope": slope})
    res = pl.DataFrame(out).with_columns(pl.col("rho").abs().alias("abs_rho")).sort("target", "abs_rho", descending=[False, True])
    with pl.Config(tbl_rows=120, tbl_width_chars=220, fmt_float="mixed", fmt_str_lengths=30):
        print("== clocks / weight ratios vs the day's activity (best 3 per target)")
        print(res.group_by("target", maintain_order=True).head(3).drop("abs_rho"))
        print("\n== 8 stocks: clock changes and activity")
        print(j.select("sid", *[f"d:{k}" for k in CLOCKS], *[f"r:{k}" for k in WGTS[:3]], *stats).sort("sid").head(8))

    # tpv / v per exchange, and which slice explains the gap
    t = f"(((time // {NS}) + {LOCAL}) % 86400)"
    tr = con.sql(f"""SELECT symbol_id AS sid, any_value(market_id) AS mkt,
        sum(volume * price) FILTER ({t} < 53820) AS amt_no_close, (sum(volume) FILTER ({t} < 53820))::DOUBLE AS vol_no_close,
        sum(volume * price) FILTER ({t} < 34200) AS amt_open_auct, (sum(volume) FILTER ({t} < 34200))::DOUBLE AS vol_open_auct,
        sum(volume * price) FILTER ({t} >= 34200 AND {t} < 34201) AS amt_0930, (sum(volume) FILTER ({t} >= 34200 AND {t} < 34201))::DOUBLE AS vol_0930,
        sum(volume * price) FILTER ({t} >= 53820 AND {t} < 53821) AS amt_1457, (sum(volume) FILTER ({t} >= 53820 AND {t} < 53821))::DOUBLE AS vol_1457,
        sum(volume * price) FILTER ({t} >= 53821) AS amt_close_auct, (sum(volume) FILTER ({t} >= 53821))::DOUBLE AS vol_close_auct,
        sum(volume * price) FILTER ({t} >= 53819 AND {t} < 53820) AS amt_last_second, (sum(volume) FILTER ({t} >= 53819 AND {t} < 53820))::DOUBLE AS vol_last_second
        FROM raw_transaction WHERE date = {int(a.date)} AND trade_type = 1 GROUP BY ALL""").pl()
    pm = pl.DataFrame([{"sid": int(s), "tpv": n[s].get("zzqj.tpv"), "v": n[s].get("zzqj.v")} for s in sids],
                      schema={"sid": pl.Int64, "tpv": pl.Float64, "v": pl.Float64})
    k = pm.join(tr.with_columns(pl.col("sid").cast(pl.Int64)), on="sid", how="left").with_columns(
        (pl.col("v") - pl.col("vol_no_close")).alias("gap_v"), (pl.col("tpv") - pl.col("amt_no_close")).alias("gap_tpv"))
    print("\n== zzqj.v vs volume without the close auction, per exchange (3553 SSE, 3554 SZSE)")
    print(k.group_by("mkt").agg(pl.len(), (pl.col("gap_v").abs() < 0.5).mean().alias("v_exact"),
                                (pl.col("gap_tpv").abs() < 0.5).mean().alias("tpv_exact"),
                                pl.col("gap_v").median().alias("gap_v_median")).sort("mkt"))
    bad = k.filter(pl.col("gap_v").abs() >= 0.5)
    print(f"\n== {bad.height} mismatched stocks: which slice equals PM - ours (volume)")
    for c in ("vol_open_auct", "vol_0930", "vol_1457", "vol_close_auct", "vol_last_second"):
        for sign in (1, -1):
            m = bad.select(((pl.col("gap_v") - sign * pl.col(c).fill_null(0)).abs() < 0.5).mean()).item()
            print(f"  gap == {'+' if sign > 0 else '-'}{c:16s}: {m:.4f}")
    with pl.Config(tbl_rows=12, tbl_width_chars=220, fmt_float="mixed"):
        print(bad.select("sid", "mkt", "v", "vol_no_close", "gap_v", "vol_open_auct", "vol_0930", "vol_1457", "vol_close_auct").head(10))


if __name__ == "__main__":
    main()
