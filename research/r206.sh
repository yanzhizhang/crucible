#!/usr/bin/env bash
# Round 206 on 97, foreground, read-only; prints only. ~5 min (one pass over each day's log, one join on the snapshots), light.
# R205 follow-up: (1) the orders with no logged block at their snapshot, replayed on the old block's position under 12 input
# choices (alpha = old block's / signal before / signal after; spread = old block's / this snapshot's; threshold on / off);
# (2) own-block orders whose replay goes through zero, per 10 minutes: does PM stop at flat, and from when.
#   bash research/r206.sh 2>&1 | tee data/reports/r206.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118 20261119 20261120}; do
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/runs_silent.py --logs $DATA/pm_224/$D/chrono_logs --date $D --catalog $DATA/catalog_s224.duckdb --show 0 2>&1 | grep -v -E "Warning|warn" | grep -E "^==|R206|^   (old block|signal|[0-9][0-9]:[0-9]0 )|replay on|all  |own block|silent" | cut -c1-260
done
echo "[$(date +%T)] r206 done"
