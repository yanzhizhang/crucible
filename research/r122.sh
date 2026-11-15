#!/usr/bin/env bash
# Round 122 on 97, foreground, read-only, writes only data/reports. ~20-40 min.
# R121: zzvm.c.re ~ trade-clock price average with a half-life at the top of the grid (172800 s: 0.77-0.86), so the grid
# now runs to 20 days; trade inputs gain the side (+1 / -1) and side x |logret| (zztx.r sits near 0 and can go negative).
# Every key also prints what PM holds (value, weight, last update time) to read the rule from.
#   bash research/r122.sh 2>&1 | tee data/reports/r122.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
for D in 20261106 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/ema_fit.py --date $D --jz $JZ --prev-jz $PV --raw $DATA/dysl2_raw --by value --top 5 \
    --keys zzvm.c.re,zzmh.r,zzcb.r,zzeb.r,zztx.r,zzhj.p 2>&1 | grep -v "best clocks by wgt" | cut -c1-400
done
echo "[$(date +%T)] done"
