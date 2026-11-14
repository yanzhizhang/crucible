#!/usr/bin/env bash
# Round 93 on 97, foreground, after R92. Writes data/backfill_3s (replaces the DataYes days that feitu also has) and
# data/reports. A few minutes per day re-run.
# R91: on 1104-1106 PM's production state matches feitu (43-50 keys fully equal on SZSE, median 1.000) and not DataYes (13-26),
# and seeding changes nothing -- the feed is the difference. So: which of the DataYes days does the feitu catalog hold, rerun
# those from feitu in the backfill, refold the state engine table, and the returns IC of the rerun days (ours from feitu;
# PM's live prediction alongside, which came from 224's DataYes-like feed).
#   bash research/r93.sh 2>&1 | tee data/reports/r93.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
B=/work/crucible/research/featgen/cpp/build/cmake
echo "[$(date +%T)] feitu catalog: snapshot rows per day from 1104"
python - <<'PY' 2>&1 | cut -c1-200
import duckdb
con = duckdb.connect("/work/crucible_data/catalog.duckdb", read_only=True)
print("  " + " | ".join(f"{d} {n:,}" for d, n in con.sql("SELECT date, count(*) FROM raw_quotation WHERE date >= 20261104 GROUP BY 1 ORDER BY 1").fetchall()))
PY
echo "[$(date +%T)] rerun the DataYes days from feitu where it has them"
python -u research/featgen/backfill_days.py --dates 20261104-20261231 --prefer-feitu --grid 3 --keep pred --pq $B/lgbpred_pq --out data/backfill_3s 2>&1 | grep -E "^20|days:" | cut -c1-230
echo "[$(date +%T)] state engine over all days"
python -u research/featgen/verify_days.py data/backfill_3s 2>&1 | head -1 | cut -c1-200
python -u research/featgen/verify_days.py data/backfill_3s 2>&1 | grep -E "^per day" | tr '|' '\n' | tail -12 | cut -c1-120
echo "[$(date +%T)] returns IC on the rerun days"
for D in 20261104 20261105 20261106 20261109 20261110 20261111 20261112 20261113; do
  f=data/backfill_3s/days/$D/pred/pred_$D.parquet
  [ -f $f ] || continue
  S=$(grep -oE "^$D: (feitu|dysl2)" data/reports/r93.txt 2>/dev/null | tail -1 | cut -d' ' -f2); S=${S:-dysl2}
  if [ $S = feitu ]; then CAT=$DATA/catalog.duckdb; else CAT=$DATA/catalog_dysl2.duckdb; fi
  echo "  $D ($S): $(CRUCIBLE_SOURCE=$S python -u research/featgen/pred_ic.py --date $D --scores $f --target $DATA/pm_target_r22/date=$D.parquet --catalog $CAT --every 30 2>&1 | grep -E "ours|PM" | tr -s ' ' | cut -c1-200 | tr '\n' ' ')"
done
echo "[$(date +%T)] done"
