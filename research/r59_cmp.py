"""Tick comparison of two catalogs on one day: PM's mp files (pmmp) against the DataYes ticks on 97 (dysl2).

    python research/r59_cmp.py 20261112 /work/crucible_data/catalog_pmmp.duckdb /work/crucible_data/catalog_dysl2.duckdb

Read-only. DataYes holds the whole market, the mp files only the stocks PM's process reads, so DataYes is restricted to
the (symbol_id, market_id) pairs of the mp side. Per stream: rows, stocks, exchange-time range; keys present on one side
only; on matched keys the share with equal price / volume / exchange time and the local arrival time difference.
"""
import sys

import duckdb

D, PM, DY = sys.argv[1], sys.argv[2], sys.argv[3]
con = duckdb.connect()
con.execute(f"ATTACH '{PM}' AS pm (READ_ONLY)")
con.execute(f"ATTACH '{DY}' AS dy (READ_ONLY)")


def cols(db: str, t: str) -> list:
    return [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM {db}.{t} LIMIT 0").fetchall()]


def one(q: str):
    return con.sql(q).fetchone()


for kind in ("quotation", "order", "transaction"):
    t = f"raw_{kind}"
    try:
        cp, cd = cols("pm", t), cols("dy", t)
    except Exception as e:  # noqa: BLE001
        print(f"  {kind}: {type(e).__name__}: {str(e)[:160]}")
        continue
    common = [c for c in cp if c in cd]
    print(f"== {kind} {D}: {len(cp)} columns in mp, {len(cd)} in DataYes, {len(common)} common; only in DataYes: {[c for c in cd if c not in cp][:12]}")
    w = f"date = '{D}'"
    con.execute(f"CREATE OR REPLACE TEMP TABLE a AS SELECT * FROM pm.{t} WHERE {w}")
    con.execute("CREATE OR REPLACE TEMP TABLE u AS SELECT DISTINCT symbol_id, market_id FROM a")
    con.execute(f"CREATE OR REPLACE TEMP TABLE b AS SELECT x.* FROM dy.{t} x JOIN u USING (symbol_id, market_id) WHERE x.{w}")
    whole = one(f"SELECT count(*), count(DISTINCT (symbol_id, market_id)) FROM dy.{t} WHERE {w}")
    for n in ("a", "b"):
        r = one(f"SELECT count(*), count(DISTINCT (symbol_id, market_id)), min(time), max(time), sum(CASE WHEN market_id = (SELECT min(market_id) FROM u) THEN 1 ELSE 0 END) FROM {n}")
        print(f"  {'mp     ' if n == 'a' else 'DataYes'}: {r[0]:>12,} rows, {r[1]} stocks, exchange time {r[2]} .. {r[3]}, rows of the lower market id {r[4]:,}")
    print(f"  DataYes whole market: {whole[0]:,} rows, {whole[1]} stocks; stocks per market in mp: {con.sql('SELECT market_id, count(*) FROM u GROUP BY 1 ORDER BY 1').fetchall()}")
    if kind == "quotation":
        key = ["symbol_id", "market_id", "time"]
    else:
        key = ["symbol_id", "market_id"] + [c for c in ("channel_id", "seq_id") if c in common]
    k = ", ".join(key)
    dup = [one(f"SELECT count(*) - count(DISTINCT ({k})) FROM {n}")[0] for n in ("a", "b")]
    print(f"  key ({k}); rows beyond the first per key: mp {dup[0]:,}, DataYes {dup[1]:,}")
    val = [c for c in ("price", "volume", "time", "close", "side", "order_type", "total_volume", "total_amount") if c in common and c not in key]
    agg = ", ".join(f"any_value({c}) AS {c}" for c in val + (["nano_ts"] if "nano_ts" in common else []))
    for n in ("a", "b"):
        con.execute(f"CREATE OR REPLACE TEMP TABLE {n}k AS SELECT {k}, {agg} FROM {n} GROUP BY {k}")
    na, nb = one("SELECT count(*) FROM ak")[0], one("SELECT count(*) FROM bk")[0]
    m = one(f"SELECT count(*) FROM ak JOIN bk USING ({k})")[0]
    print(f"  keys: mp {na:,}, DataYes {nb:,}, in both {m:,}; only in mp {na - m:,} ({(na - m) / max(na, 1):.4%}), only in DataYes {nb - m:,} ({(nb - m) / max(nb, 1):.4%})")
    if m:
        eq = ", ".join(f"avg(CASE WHEN ak.{c} IS NOT DISTINCT FROM bk.{c} OR abs(ak.{c} - bk.{c}) < 1e-9 THEN 1.0 ELSE 0.0 END)" for c in val)
        r = one(f"SELECT {eq} FROM ak JOIN bk USING ({k})") if val else ()
        print("  equal on matched keys: " + ", ".join(f"{c} {x:.6f}" for c, x in zip(val, r)))
        if "nano_ts" in common:
            r = one(f"SELECT median(ak.nano_ts - bk.nano_ts), quantile_cont(ak.nano_ts - bk.nano_ts, 0.01), quantile_cont(ak.nano_ts - bk.nano_ts, 0.99), "
                    f"avg(CASE WHEN ak.nano_ts = bk.nano_ts THEN 1.0 ELSE 0.0 END) FROM ak JOIN bk USING ({k})")
            print(f"  local arrival, mp - DataYes (ns): median {r[0]}, p1 {r[1]}, p99 {r[2]}; identical on {r[3]:.4%} of matched keys")
    if na - m:
        r = con.sql(f"SELECT (time // 60000000000) AS tmin, count(*) FROM ak ANTI JOIN bk USING ({k}) GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall() if kind != "quotation" and "time" in val else []
        s = con.sql(f"SELECT symbol_id, count(*) FROM ak ANTI JOIN bk USING ({k}) GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall()
        print(f"  only in mp: top stocks {s}; top minutes (time // 60e9) {r}")
    if nb - m:
        s = con.sql(f"SELECT symbol_id, count(*) FROM bk ANTI JOIN ak USING ({k}) GROUP BY 1 ORDER BY 2 DESC LIMIT 5").fetchall()
        print(f"  only in DataYes: top stocks {s}")
