#!/usr/bin/env bash
# Round 205 on 97, foreground, read-only; prints only. ~5 min (one pass over the 1118 log, one join on the snapshots), light.
# R198 follow-up: rows with no logged block at their snapshot keep the old block state as it is (no order taken out); the
# quantity-only misses are classified (time of day, what set the quantity, what the SO quantity equals). 1118 then 1119, 1120.
#   bash research/r205.sh 2>&1 | tee data/reports/r205.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118 20261119 20261120}; do
  for S in ${STORES:-s224}; do
  echo "[$(date +%T)] $D book from $S"
  python -u research/pm_factors/runs_silent.py --logs $DATA/pm_224/$D/chrono_logs --date $D --catalog $DATA/catalog_$S.duckdb --show 14 2>&1 | grep -v -E "Warning|warn" | cut -c1-420
  done
done
echo "[$(date +%T)] r205 done"
