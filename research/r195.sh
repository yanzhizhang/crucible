#!/usr/bin/env bash
# Round 195 on 97, foreground, read-only; prints only. ~5 min (one pass over the 1118 log, one join on the snapshots), light.
# The silent runs of runS: every SO line replayed from the stock's last LOGGED block (alpha arrival) carried to the
# snapshot that triggered the order (SO field 2 = its receive epoch), book from 224 own feed (s224, re-decoded in R194) and from DataYesSharesL2.
#   bash research/r195.sh 2>&1 | tee data/reports/r195.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118}; do
  for S in ${STORES:-s224 dysl2}; do
  echo "[$(date +%T)] $D book from $S"
  python -u research/pm_factors/runs_silent.py --logs $DATA/pm_224/$D/chrono_logs --date $D --catalog $DATA/catalog_$S.duckdb --show 14 2>&1 | grep -v -E "Warning|warn" | cut -c1-330
  done
done
echo "[$(date +%T)] r195 done"
