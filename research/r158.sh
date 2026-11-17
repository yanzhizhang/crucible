#!/usr/bin/env bash
# Round 158 on 97, foreground, read-only, ~1 min: PM's own day-end state for the zzvx / zzxu / zzup / zzpk keys -- every key's
# level, which pairs of keys are ordered (K1 >= K2, often equal: the shape zzvx.a's thresholds just above 1 ask for), and the
# day-end ratios closest to the three suspect columns' training thresholds. Also lists keys we do not build (zzxu.upper /
# lower / dir were seen in R148's key list).
#   bash research/r158.sh 2>&1 | tee data/reports/r158.txt
set -u
cd /work/crucible
SEED=/work/crucible_data/seed_probe/HS300
DAYS=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | tail -8 | paste -sd, -)
python -u research/pm_factors/jz_pairs.py --jz-dir $SEED --days "$DAYS" --key-re '^zz(vx|xu|up|pk)\.' \
  --thr "zzvx.a=1.0000005,1.0044,1.139" --thr "zzxu.a=0.119,0.212,0.405" --thr "zzxu.b=0.848,0.981,1.261" --thr "zzup=0.99336,0.99999,1.00819" 2>&1 | cut -c1-220
echo "[$(date +%T)] r158 done"
