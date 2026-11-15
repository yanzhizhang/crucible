#!/usr/bin/env bash
# Round 124 on 97, foreground, read-only, writes only data/reports. ~5 min.
# R123: no exact input for zzmh.r (~1 s) / zzcb.r (~4.7 s) / zzeb.r -- counting exact matches is fragile for averages this
# fast. Instead: which last-snapshot quantity is PM's value PROPORTIONAL to (ratio spread, power-law slope)?
#   bash research/r124.sh 2>&1 | tee data/reports/r124.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/pm_factors/last_probe.py --date 20261106 --jz /data/prod/h/HS300/p1_intrading_dcp/20261106.jz --raw $DATA/dysl2_raw 2>&1 | cut -c1-200
python -u research/pm_factors/last_probe.py --date 20261113 --jz $DATA/seed_probe/HS300/20261113.jz --raw $DATA/dysl2_raw 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
