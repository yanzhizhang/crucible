r"""Stage 4: every decoded rule of PM's day-end state, checked on several days at once.

One row per (day, check) so a rule that holds on one day by accident shows up at once. Days are
the trade days of ``--dates`` that have both a state file ``<dcp-dir>/<D>.jz`` and decoded ticks.

Checks (each the share of stocks where the rule holds exactly, plus a spread where useful):

* ``v`` / ``tpv`` (``zzqj``) = volume / amount of all trades from 09:30:00 exchange time, per
  exchange, on the raw trades and after dropping duplicate trades (same channel + seq, or same
  buy / sell order, time, price, volume);
* ``t_fast`` (``zzba.p.t``) = local arrival (trading clock) of some snapshot of that stock, to 10 ms;
  the latest such arrival is PM's process stop time;
* ``t_slow`` (``zzbd.pEmaT.t``) - local arrival of the stock's last trade (median, p5, p95);
* ``zzgh.nb`` = (buy - sell) / total amount of continuous-session trades (exact 1e-6, median error);
* ``zzxo.r.0..29`` = last snapshot price strictly before 14:57:00 - (29 - k) * 10 s;
* ``zzpk.high / low / open`` = the day's snapshot high / low / open.

    python research/pm_factors/dcp_multiday.py --dates 20260429,20260922,20260923,20260924
"""

from __future__ import annotations

import argparse
import gzip
import json
import warnings
from pathlib import Path

import duckdb
import polars as pl

warnings.filterwarnings("ignore", message="Sortedness of columns")
CATALOG = Path("/work/crucible_data/catalog.duckdb")
DCP = Path("/data/prod/h/HS300/p1_intrading_dcp")
SOD = "((time / 1e9 + 28800) % 86400)"
LSOD = "((spider_ts / 1e9 + 28800) % 86400)"


def tc(s: str) -> str:
    """Trading clock: seconds since 09:30 with the lunch break removed."""
    return f"CASE WHEN {s} <= 41400 THEN {s} - 34200 WHEN {s} < 46800 THEN 7200 ELSE {s} - 46800 + 7200 END"


def state(path: Path) -> pl.DataFrame:
    """Return the scalar keys this script checks, one row per stock."""
    with gzip.open(path) as f:
        d = json.loads(f.read())
    keys = ["zzqj.v", "zzqj.tpv", "zzba.p.t", "zzbd.pEmaT.t", "zzgh.nb", "zzpk.high", "zzpk.low", "zzpk.open",
            *[f"zzxo.r.{k}" for k in range(30)]]
    rows = [{"sid": int(s), **{k: (None if kv.get(k) is None else float(kv[k])) for k in keys}} for s, kv in d.items()]
    return pl.DataFrame(rows, schema={"sid": pl.Int64, **dict.fromkeys(keys, pl.Float64)})


def share(df: pl.DataFrame, expr: pl.Expr) -> float:
    """Share of rows where ``expr`` is true (nulls count as false)."""
    return float(df.select(expr.fill_null(False).mean()).item() or 0.0)


def check_day(con: duckdb.DuckDBPyConnection, d: int, pm: pl.DataFrame) -> list[dict]:
    """All checks for one day."""
    con.register("pm", pm)
    out: list[dict] = []

    def add(check: str, value: float, note: str = "") -> None:
        out.append({"date": d, "check": check, "value": value, "note": note})

    # volume / amount from 09:30:00, raw and de-duplicated, per exchange
    vol = con.sql(f"""
      WITH t AS (SELECT r.* FROM raw_transaction r JOIN pm ON pm.sid = r.symbol_id
                 WHERE r.date = {d} AND r.trade_type = 1 AND {SOD} >= 34200),
      raw AS (SELECT symbol_id AS sid, any_value(market_id) AS mkt, sum(volume)::DOUBLE AS v, sum(volume * price) AS a FROM t GROUP BY 1),
      d1 AS (SELECT sid, sum(volume)::DOUBLE AS v1, sum(volume * price) AS a1 FROM
             (SELECT symbol_id AS sid, any_value(volume) AS volume, any_value(price) AS price FROM t GROUP BY symbol_id, channel_id, seq_id) GROUP BY 1),
      d2 AS (SELECT sid, sum(volume)::DOUBLE AS v2, sum(volume * price) AS a2 FROM
             (SELECT symbol_id AS sid, volume, price FROM t GROUP BY symbol_id, buy_seq_id, sell_seq_id, time, price, volume) GROUP BY 1)
      SELECT * FROM raw JOIN d1 USING (sid) JOIN d2 USING (sid)""").pl()
    j = vol.join(pm.select("sid", pl.col("zzqj.v").alias("pv"), pl.col("zzqj.tpv").alias("pa")), on="sid")
    for mkt, name in ((3553, "SSE"), (3554, "SZSE")):
        m = j.filter(pl.col("mkt") == mkt)
        for tag, vc, ac in (("raw", "v", "a"), ("dedup_chan_seq", "v1", "a1"), ("dedup_fields", "v2", "a2")):
            add(f"v {name} {tag}", share(m, (pl.col("pv") - pl.col(vc)).abs() < 0.5), f"n={m.height}")
            add(f"tpv {name} {tag}", share(m, (pl.col("pa") - pl.col(ac)).abs() <= 1e-6 * pl.col("pa").abs() + 0.01))

    # t_fast: some snapshot's local arrival within 10 ms; the latest matched arrival = PM stop time
    tf = con.sql(f"""SELECT pm.sid, max({LSOD}) FILTER (abs({tc(LSOD)} - pm."zzba.p.t") < 0.01) AS hit_local
                     FROM raw_quotation q JOIN pm ON pm.sid = q.symbol_id WHERE q.date = {d} GROUP BY pm.sid""").pl()
    stop = tf["hit_local"].max()
    add("t_fast = a snapshot's local arrival (10ms)", share(tf, pl.col("hit_local").is_not_null()),
        f"latest match {int(stop) // 3600:02d}:{int(stop) % 3600 // 60:02d}:{stop % 60:06.3f}" if stop else "")

    # t_slow vs the last trade's local arrival
    ts = con.sql(f"""SELECT pm.sid, any_value(pm."zzbd.pEmaT.t") - max({tc(LSOD)}) AS d FROM raw_transaction r
                     JOIN pm ON pm.sid = r.symbol_id WHERE r.date = {d} AND r.trade_type = 1 GROUP BY pm.sid""").pl()["d"].drop_nulls()
    if ts.len():
        add("t_slow - last trade local arrival (median s)", float(ts.median()),
            f"p5 {ts.quantile(0.05):.4f} p95 {ts.quantile(0.95):.4f}")

    # zzgh.nb vs continuous-session amount imbalance
    for tag, src in (("raw", "t"), ("dedup_chan_seq", "(SELECT any_value(COLUMNS(*)) FROM t GROUP BY symbol_id, channel_id, seq_id)")):
        nb = con.sql(f"""WITH t AS (SELECT r.* FROM raw_transaction r JOIN pm ON pm.sid = r.symbol_id
                         WHERE r.date = {d} AND r.trade_type = 1 AND {SOD} >= 34200 AND {SOD} < 53820)
            SELECT symbol_id AS sid,
            (sum(volume * price) FILTER (dir = 1063) - sum(volume * price) FILTER (dir = 1550)) / sum(volume * price) AS imb
            FROM {src} GROUP BY 1""").pl().join(pm.select("sid", pl.col("zzgh.nb").alias("nb")), on="sid")
        add(f"zzgh.nb = cont amount imbalance {tag}", share(nb, (pl.col("nb") - pl.col("imb")).abs() <= 1e-6),
            f"median rel err {nb.select(((pl.col('nb') - pl.col('imb')) / pl.col('imb')).abs().median()).item():.5f}")

    # zzxo price list: last snapshot price strictly before 14:57:00 - (29 - k) * 10 s
    sn = con.sql(f"""SELECT symbol_id AS sid, {SOD} AS s, price FROM raw_quotation q JOIN pm ON pm.sid = q.symbol_id
                     WHERE q.date = {d} AND price > 0 AND {SOD} BETWEEN 53400 AND 53820""").pl().with_columns(
        pl.col("sid").cast(pl.Int64), pl.col("s").cast(pl.Float64)).sort("s")
    xo = pm.unpivot(index="sid", on=[f"zzxo.r.{k}" for k in range(30)], variable_name="key", value_name="px").drop_nulls().with_columns(
        (53820 - (29 - pl.col("key").str.extract(r"(\d+)$").cast(pl.Int64)) * 10 - 1e-6).cast(pl.Float64).alias("s")).sort("s")
    xj = xo.join_asof(sn, on="s", by="sid", strategy="backward")
    add("zzxo.r = snapshot price 10s grid before 14:57", share(xj, (pl.col("px") - pl.col("price")).abs() < 1e-6), f"cells={xj.height}")

    # zzpk high / low / open vs the day's snapshots
    hl = con.sql(f"""SELECT symbol_id AS sid, arg_max(high, time) AS h, arg_max(low, time) AS l, arg_max(open, time) AS o
                     FROM raw_quotation q JOIN pm ON pm.sid = q.symbol_id WHERE q.date = {d} AND price > 0 GROUP BY 1""").pl().join(pm, on="sid")
    for k, c in (("zzpk.high", "h"), ("zzpk.low", "l"), ("zzpk.open", "o")):
        add(f"{k} = day snapshot value", share(hl, (pl.col(k) - pl.col(c)).abs() < 1e-6))
    con.unregister("pm")
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True, help="comma-separated trade days")
    ap.add_argument("--dcp-dir", type=Path, default=DCP)
    a = ap.parse_args()
    con = duckdb.connect(str(CATALOG), read_only=True)
    have = {str(r[0]) for r in con.sql("SELECT DISTINCT date FROM raw_quotation").fetchall()} & \
           {str(r[0]) for r in con.sql("SELECT DISTINCT date FROM raw_transaction").fetchall()}
    rows: list[dict] = []
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        f = a.dcp_dir / f"{d}.jz"
        if not f.exists() or d not in have:
            print(f"skip {d}: state file {'ok' if f.exists() else 'missing'}, ticks {'ok' if d in have else 'not decoded'}")
            continue
        rows += check_day(con, int(d), state(f))
        print(f"done {d}")
    if not rows:
        return
    res = pl.DataFrame(rows)
    wide = res.pivot(on="date", index="check", values="value", sort_columns=True, maintain_order=True)
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=250, fmt_str_lengths=50, float_precision=4):
        print("\n== value per day (shares are 0..1; 1.0 = rule holds for every stock)")
        print(wide)
        print("\n== notes")
        print(res.filter(pl.col("note") != "").select("date", "check", "note"))
    out = Path("data/reports/stage4")
    out.mkdir(parents=True, exist_ok=True)
    res.write_csv(out / "dcp_multiday.csv")
    print(f"table: {out / 'dcp_multiday.csv'}")


if __name__ == "__main__":
    main()
