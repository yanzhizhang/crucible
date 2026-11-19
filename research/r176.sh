#!/usr/bin/env bash
# Round 176 on 97, foreground: 224's own DataYes feed (TDF from 2.23, /work/crucible_data/tdf_224) as a third market-data
# source "s224" next to feitu and dysl2. The layout is exactly what decode_tdf_day.py reads. Three parts, in order:
#  probe   light (~10 s): headers + field mapping + three converted rows per kind for 20261118; writes nothing
#  decode  HEAVY, alone (96 workers, maybe 5-15 min): 20261118 and 20261119 -> /work/crucible_data/s224_raw, then
#          CRUCIBLE_SOURCE=s224 catalog.py -> catalog_s224.duckdb. New directories only; dysl2 and feitu are untouched.
#  cmp     moderate (~2 min, duckdb): 20261118 in s224 against dysl2 (the DataYesSharesL2 package): rows per kind and
#          exchange; for the 300 strategy stocks, snapshots per stock, shared exchange stamps, price / volume equal on the
#          shared ones, the local receive time difference; orders and trades by sequence number present in one and not the other.
#  catalog (re)build catalog_s224.duckdb only (light)
#   bash research/r176.sh probe 2>&1 | tee data/reports/r176_probe.txt     (same for decode, catalog, cmp)
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?probe|decode|catalog|cmp}" in
probe)
  python -u research/decode_tdf_day.py --root $DATA/tdf_224 --date 20261118 --out $DATA/s224_raw --probe 2>&1 | grep -v Warning | cut -c1-260 ;;
decode)
  python -u research/decode_tdf_day.py --root $DATA/tdf_224 --date 20261118,20261119 --out $DATA/s224_raw --workers 96 2>&1 | grep -v Warning | tail -40 | cut -c1-220
  CRUCIBLE_SOURCE=s224 python -u research/catalog.py 2>&1 | tail -6 | cut -c1-200
  du -sh $DATA/s224_raw $DATA/catalog_s224.duckdb 2>&1 ;;
catalog)
  CRUCIBLE_SOURCE=s224 python -u research/catalog.py 2>&1 | tail -12 | cut -c1-200
  CRUCIBLE_SOURCE=s224 python -c "import sys; sys.path.insert(0,'research/pm_factors'); import duckdb, dcp_build as B; c=duckdb.connect(str(B.CATALOG), read_only=True); print('views:', [r[0] for r in c.sql('SELECT view_name FROM duckdb_views()').fetchall() if r[0].startswith('raw_')]); print('quotes per day:', c.sql('SELECT date, count(*), count(DISTINCT symbol_id) FROM raw_quotation GROUP BY 1 ORDER BY 1').fetchall())" 2>&1 | tail -2 | cut -c1-200
  du -sh $DATA/catalog_s224.duckdb ;;
cmp)
  python - <<'PY' 2>&1 | grep -v Warning | cut -c1-230
import duckdb
D = "/work/crucible_data"
day = 20261118
c = duckdb.connect()
c.execute("SET threads=32")
def v(src, kind):
    return f"read_parquet('{D}/{src}_raw/kind={kind}/date={day}/part-*.parquet')"
import polars as pl
sids = pl.read_parquet(f"{D}/pm_signals/date={day}.parquet")["sid"].unique().cast(pl.Int64).to_list()
S = ",".join(str(s) for s in sids)
print(f"== {day}: s224 (224's own feed) vs dysl2 (DataYesSharesL2); {len(sids)} strategy stocks")
for kind in ("quotation", "order", "transaction"):
    for src in ("s224", "dysl2"):
        try:
            r = c.sql(f"SELECT market_id, count(*), count(DISTINCT symbol_id) FROM {v(src, kind)} GROUP BY 1 ORDER BY 1").fetchall()
            print(f"   {kind:<11} {src:<5} (market, rows, symbols): {r}")
        except Exception as e:
            print(f"   {kind:<11} {src:<5} FAILED: {str(e)[:120]}")
q = f"""WITH a AS (SELECT symbol_id, time, any_value(price) p, any_value(total_volume) tv, min(spider_ts) ts FROM {v('s224','quotation')} WHERE symbol_id IN ({S}) GROUP BY 1, 2),
             b AS (SELECT symbol_id, time, any_value(price) p, any_value(total_volume) tv, min(spider_ts) ts FROM {v('dysl2','quotation')} WHERE symbol_id IN ({S}) GROUP BY 1, 2)
        SELECT count(*) FILTER (a.time IS NOT NULL) n_a, count(*) FILTER (b.time IS NOT NULL) n_b, count(*) FILTER (a.time IS NOT NULL AND b.time IS NOT NULL) n_both,
               count(*) FILTER (a.p = b.p AND a.tv = b.tv) same_px_vol,
               quantile_cont((a.ts - b.ts) / 1e6, [0.01, 0.1, 0.5, 0.9, 0.99]) FILTER (a.time IS NOT NULL AND b.time IS NOT NULL) dt_ms
        FROM a FULL JOIN b USING (symbol_id, time)"""
try:
    r = c.sql(q).fetchone()
    print(f"-- snapshots of the strategy stocks, keyed by (stock, exchange stamp): s224 {r[0]:,}, dysl2 {r[1]:,}, in both {r[2]:,} "
          f"({r[2] / max(r[0], 1):.4f} of s224, {r[2] / max(r[1], 1):.4f} of dysl2); price and volume equal on {r[3]:,} of the shared")
    print(f"   local receive time s224 - dysl2, ms, q1 / q10 / q50 / q90 / q99: {[round(x, 2) for x in r[4]]}")
    per = c.sql(f"""SELECT quantile_cont(n, [0.0, 0.1, 0.5, 0.9, 1.0]) FROM (SELECT symbol_id, count(*) n FROM {v('s224','quotation')} WHERE symbol_id IN ({S}) GROUP BY 1)""").fetchone()[0]
    per2 = c.sql(f"""SELECT quantile_cont(n, [0.0, 0.1, 0.5, 0.9, 1.0]) FROM (SELECT symbol_id, count(*) n FROM {v('dysl2','quotation')} WHERE symbol_id IN ({S}) GROUP BY 1)""").fetchone()[0]
    print(f"   snapshot rows per stock min / q10 / q50 / q90 / max: s224 {per}, dysl2 {per2}")
except Exception as e:
    print("-- snapshot compare FAILED:", str(e)[:300])
for kind in ("order", "transaction"):
    q = f"""WITH a AS (SELECT DISTINCT symbol_id, seq_id FROM {v('s224', kind)} WHERE symbol_id IN ({S})),
                 b AS (SELECT DISTINCT symbol_id, seq_id FROM {v('dysl2', kind)} WHERE symbol_id IN ({S}))
            SELECT count(*) FILTER (a.seq_id IS NOT NULL), count(*) FILTER (b.seq_id IS NOT NULL), count(*) FILTER (a.seq_id IS NOT NULL AND b.seq_id IS NOT NULL)
            FROM a FULL JOIN b USING (symbol_id, seq_id)"""
    try:
        r = c.sql(q).fetchone()
        print(f"-- {kind}s of the strategy stocks by (stock, sequence number): s224 {r[0]:,}, dysl2 {r[1]:,}, in both {r[2]:,}; only in dysl2 {r[1] - r[2]:,} ({(r[1] - r[2]) / max(r[1], 1):.5f}), only in s224 {r[0] - r[2]:,}")
        m = c.sql(f"""WITH a AS (SELECT DISTINCT symbol_id, seq_id FROM {v('s224', kind)} WHERE symbol_id IN ({S})),
                           b AS (SELECT symbol_id, seq_id, market_id, time FROM {v('dysl2', kind)} WHERE symbol_id IN ({S}))
                      SELECT market_id, count(*), min(strftime(make_timestamp(CAST(time / 1000 AS BIGINT)) + INTERVAL 8 HOUR, '%H:%M')), max(strftime(make_timestamp(CAST(time / 1000 AS BIGINT)) + INTERVAL 8 HOUR, '%H:%M'))
                      FROM b ANTI JOIN a USING (symbol_id, seq_id) GROUP BY 1 ORDER BY 1""").fetchall()
        print(f"   missing from s224 (market, rows, first / last exchange time): {m}")
    except Exception as e:
        print(f"-- {kind} compare FAILED:", str(e)[:300])
PY
  ;;
esac
echo "[$(date +%T)] r176 $1 done"
