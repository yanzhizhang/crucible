#!/usr/bin/env bash
# Round 201 on 97, foreground, read-only; prints only. ~3-6 min per day (python over 300 stocks' snapshots and trades), light, no engine.
# R200 follow-up: clock event-set variants (new exchange stamp / volume changed / from 09:30; trades from 09:30 / to 14:57),
# many more snapshot inputs, ratio spread per input (a tight PM / ours ratio = a scale factor), on s224 AND dysl2.
#   bash research/r201.sh 2>&1 | tee data/reports/r201.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118 20261119 20261120}; do
  for S in s224 dysl2; do
  echo "[$(date +%T)] $D $S"
  CRUCIBLE_SOURCE=$S python -u research/engine/ema_probe.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --catalog $DATA/catalog_$S.duckdb 2>&1 | grep -v Warning | cut -c1-220
  done
done
echo "[$(date +%T)] r201 done"
