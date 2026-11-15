#!/usr/bin/env bash
# Round 116 on 97, foreground, read-only on inputs; rewrites data/side_r115/zzhj_*.parquet, writes data/reports. ~3 min.
# R115: zzhj.p is not today's mean quarter volume -- but its wgt is ~14.93 on EVERY stock = the steady state of an
# average updated once a quarter (16 a day) with half-life ~9000 s, carried over from yesterday. So: seed with yesterday's
# .jz zzhj.p (sum / wgt), add today's 16 sampled quarter volumes (or amounts), compare with today's .jz.
#   bash research/r116.sh 2>&1 | tee data/reports/r116.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
for D in 20261106 20261112 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261112) JZ=$SP/$D.jz; PV=$SP/20261111.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  echo "[$(date +%T)] $D (prev $PV)"
  python -u research/featgen/zzhj_quarter.py --date $D --x data/reports/stage6_r99_${D:4}/x.parquet --raw $DATA/dysl2_raw --jz $JZ --prev-jz $PV --halves 0 \
    --out data/side_r115 2>&1 | grep -E "carried|e.g." | cut -c1-300
done
echo "[$(date +%T)] done"
