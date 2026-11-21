#!/usr/bin/env bash
# Round 197 on 97, foreground, read-only; prints only. ~5 min (one pass over the 1118 log, one join on the snapshots), light.
# R196 follow-up: each SO line is replayed on the block of its own snapshot as (a) the logged alpha-arrival run, (b) the unlogged
# snapshot-driven run just before it: previous alpha, position / sellable / room as they were before this order; (c) a if exact, else b.
#   bash research/r197.sh 2>&1 | tee data/reports/r197.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118}; do
  for S in ${STORES:-s224}; do
  echo "[$(date +%T)] $D book from $S"
  python -u research/pm_factors/runs_silent.py --logs $DATA/pm_224/$D/chrono_logs --date $D --catalog $DATA/catalog_$S.duckdb --show 14 2>&1 | grep -v -E "Warning|warn" | cut -c1-330
  done
done
echo "[$(date +%T)] r197 done"
