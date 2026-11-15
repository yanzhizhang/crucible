#!/usr/bin/env bash
# Round 112 on 97, foreground, read-only, writes only data/reports. ~10-20 min; parallel with R105b / R111 is fine.
# The averages in PM's .jz our engine does not build (zzvm.c.re, zzmh.r, zzhj.p, zzcb.r, zzeb.r, zztx.r, zzui.r,
# zzeg.c.rEma) and the built ones that miss on both days (zzup.ps, zzba.p): fit clock + half-life from the weight,
# then the input from the value (research/pm_factors/ema_fit.py), on 1106, 1112, 1113 (DataYes ticks, PM's .jz).
#   bash research/r112.sh 2>&1 | tee data/reports/r112.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261106 20261112 20261113; do
  if [ $D = 20261106 ]; then JZ=/data/prod/h/HS300/p1_intrading_dcp/$D.jz; else JZ=$DATA/seed_probe/HS300/$D.jz; fi
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/ema_fit.py --date $D --jz $JZ --raw $DATA/dysl2_raw 2>&1 | cut -c1-400
done
echo "[$(date +%T)] done"
