#!/usr/bin/env bash
# Round 200 on 97, foreground, read-only; prints only. ~3-6 min per day (python over 300 stocks' snapshots and trades), light, no engine.
# Pins the clock and half-life of six PM averages from their WEIGHT alone (input-free), then scores input candidates for
# zzhj.p / zzui.r / zzvm.c.re: zzhj.p, zzui.r, zzvm.c.re, zzcb.r, zzeb.r, zztx.r on 224's own feed (s224).
#   bash research/r200.sh 2>&1 | tee data/reports/r200.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118 20261119 20261120}; do
  echo "[$(date +%T)] $D"
  CRUCIBLE_SOURCE=s224 python -u research/engine/ema_probe.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --catalog $DATA/catalog_s224.duckdb 2>&1 | grep -v Warning | cut -c1-200
done
echo "[$(date +%T)] r200 done"
