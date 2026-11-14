#!/usr/bin/env bash
# Round 84 on 97, foreground, long (about 1-1.5 hours). Writes data/backfill_3s and data/reports only.
# The backfill: every day with a PM state file (p1_intrading_dcp, then seed_probe), 3-second grid, the whole C++ chain,
# each day's engine state against PM's .jz; keeps only the scores (~11 MB a day, under 1 GB in all), then the folded
# state-engine table (which PM keys are still off, every day or some) and returns IC for every day.
#   bash research/r84.sh 2>&1 | tee data/reports/r84.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
B=/work/crucible/research/featgen/cpp/build/cmake
echo "[$(date +%T)] free space before: $(df -h /work | tail -1)"
python -u research/featgen/backfill_days.py --dates 20260101-20261231 --grid 3 --keep pred --pq $B/lgbpred_pq --out data/backfill_3s 2>&1 | cut -c1-260
echo "[$(date +%T)] state engine over the days"
python -u research/featgen/verify_days.py data/backfill_3s 2>&1 | cut -c1-200
echo "[$(date +%T)] returns IC per day (ours, and PM's prediction where pm_target_r22 has the day)"
for f in data/backfill_3s/days/*/pred/pred_*.parquet; do
  D=$(basename $f .parquet); D=${D#pred_}
  if [ $D -ge 20261104 ]; then S=dysl2; CAT=$DATA/catalog_dysl2.duckdb; else S=feitu; CAT=$DATA/catalog.duckdb; fi
  CRUCIBLE_SOURCE=$S python -u research/featgen/pred_ic.py --date $D --scores $f --target $DATA/pm_target_r22/date=$D.parquet --catalog $CAT --every 30 2>&1 | grep -E "ours|PM" | cut -c1-260
done
echo "[$(date +%T)] free space after: $(df -h /work | tail -1); kept $(du -sh data/backfill_3s | cut -f1)"
echo "[$(date +%T)] done"
