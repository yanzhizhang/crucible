#!/usr/bin/env bash
# Round 123 on 97, foreground, read-only, writes only data/reports. ~15-30 min.
# R122, PM's own weights at the close: zzmh.r 1.0007-1.1439 (steady 1.1439 on 3 s snapshots = half-life ~1.0 s), zzcb.r
# 1.37-2.81 (steady ~2.81 = ~4.7 s), zzeb.r 11-24 (~24.16 = ~49 s) and zzeb.r's value 0.121-0.159 on EVERY stock (structural:
# a snapshot-gap average?). So: half-lives down to 0.5 s, and inputs for the gap between snapshots, level-1 amounts, the
# range over price -- for the model-relevant zzmh.r / zzcb.r and for zzeb.r.
#   bash research/r123.sh 2>&1 | tee data/reports/r123.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
for D in 20261106 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/ema_fit.py --date $D --jz $JZ --prev-jz $PV --raw $DATA/dysl2_raw --by value --top 6 \
    --keys zzmh.r,zzcb.r,zzeb.r 2>&1 | grep -v "best clocks by wgt" | cut -c1-500
done
echo "[$(date +%T)] done"
