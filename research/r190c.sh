#!/usr/bin/env bash
# Round 190c on 97, foreground, read-only; prints only, light. The order replay of R190 with (1) each SO line paired to the
# stock's LAST decision before the SO stamp (R190b: the decision is received ~0.5 s before the stamp), (2) the book taken
# from each tick store in turn (s224's level-1 sizes are mostly 0), (3) five readings of the size cap: level 1 as shares,
# level 1 / levels 1-5 / all levels as lots, and no cap. First a direct look at the level-1 sizes in both stores.
#   bash research/r190c.sh 2>&1 | tee data/reports/r190c.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
D=${D:-20261118}
for S in s224 dysl2; do
  python - <<PY 2>&1 | cut -c1-260
import duckdb
c = duckdb.connect("$DATA/catalog_$S.duckdb", read_only=True)
r = c.sql("""SELECT count(*), avg((bid_vol[1] = 0)::INT), avg((ask_vol[1] = 0)::INT), median(bid_vol[1]), median(ask_vol[1]), avg((bid_px[1] > 0)::INT), avg(len(bid_vol)), median(list_sum(bid_vol))
             FROM raw_quotation WHERE date = $D AND symbol_id IN (600000, 1, 300750, 688981) AND ((time / 1e9 + 28800) % 86400) BETWEEN 34200 AND 53820""").fetchone()
print("== $S level 1 on 4 stocks, continuous session: rows %d; bid_vol[1] == 0 on %.3f, ask_vol[1] == 0 on %.3f; median bid / ask size %s / %s; bid_px[1] > 0 on %.3f; levels per row %.1f; median sum of all bid levels %s" % r)
print("   three rows of 600000 near 10:00:", c.sql("SELECT strftime(to_timestamp(time / 1e9 + 28800), '%H:%M:%S') AS t, bid_px[1:3] AS bp, bid_vol[1:3] AS bv, ask_px[1:3] AS ap, ask_vol[1:3] AS av FROM raw_quotation WHERE date = $D AND symbol_id = 600000 AND ((time / 1e9 + 28800) % 86400) >= 36000 ORDER BY time LIMIT 3").fetchall())
PY
done
for S in dysl2 s224; do
  for P in backward nearest; do
    echo "[$(date +%T)] ==== book from $S, pairing $P"
    python -u research/pm_factors/runs_ord.py --orders $DATA/pm_orders --date $D --catalog $DATA/catalog_$S.duckdb --pair $P --show 6 2>&1 | grep -v -E "Warning|warn|pj = " | sed '/^-- D/,$d' | cut -c1-300
  done
done
echo "[$(date +%T)] r190c done"
