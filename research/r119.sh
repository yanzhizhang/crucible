#!/usr/bin/env bash
# Round 119 on 97, foreground, read-only, writes only data/reports. ~20-40 min.
# R112 fitted the unbuilt .jz averages by their weight, but the weight follows the feed's update count (DataYes is not
# PM's colo feed), so most keys never got to the input step. Now rank clock x half by the VALUE match of the best of
# ~30 inputs (new: drawdown from zzmh.hm, spread in ticks, level-1 depth / imbalance, logs), rel 1e-3.
#   bash research/r119.sh 2>&1 | tee data/reports/r119.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261106 20261112 20261113; do
  if [ $D = 20261106 ]; then JZ=/data/prod/h/HS300/p1_intrading_dcp/$D.jz; else JZ=$DATA/seed_probe/HS300/$D.jz; fi
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/ema_fit.py --date $D --jz $JZ --raw $DATA/dysl2_raw --by value --top 5 \
    --keys zzmh.r,zzcb.r,zzeb.r,zztx.r,zzvm.c.re,zzhj.p,zzup.ps,zzba.p,zzui.r,zzqj.r,zzeg.c.rEma 2>&1 | grep -v "best clocks by wgt" | cut -c1-400
done
echo "[$(date +%T)] done"
