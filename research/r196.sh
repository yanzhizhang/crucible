#!/usr/bin/env bash
# Round 196 on 97, foreground, read-only; prints only. ~5 min (one pass over the 1118 log, one join on the snapshots), light.
# R195 follow-up: an SO line is replayed on the block of its own snapshot (SO field 2 == the block receive epoch) with (a) the
# block logged alpha, (b) the first signal chrono handled for the stock after that snapshot (trk SIG), (c) a, else b.
#   bash research/r196.sh 2>&1 | tee data/reports/r196.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118}; do
  for S in ${STORES:-s224}; do
  echo "[$(date +%T)] $D book from $S"
  python -u research/pm_factors/runs_silent.py --logs $DATA/pm_224/$D/chrono_logs --date $D --catalog $DATA/catalog_$S.duckdb --show 14 2>&1 | grep -v -E "Warning|warn" | cut -c1-330
  done
done
echo "[$(date +%T)] r196 done"
