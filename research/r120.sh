#!/usr/bin/env bash
# Round 120 on 97, foreground, read-only on inputs; writes data/greedy_fill_r120* (scratch) and data/reports. ~30-60 min.
# R117 / R117b: from nothing the greedy cannot start (zzeg only pays inside a set) and from everything it gets stuck below
# 0430's baseline. So seed zzeg (two ways: span / rEma from the 4-day channel, R111b; R114's best) and grow forward on
# 0430 + 1112 + 1113 from the fingerprint options plus the family-grammar leads (R105b: zzjx3.b dm:-ret90, zzop.i
# |ret300s|*1e4, zzpe.e srank:-lpe50 / -rete30; R114: zzab2 ret120).
#   bash research/r120.sh 2>&1 | tee data/reports/r120.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
LEADS="zzab2:id:ret120,zzjx3.b:-dm:ret90,zzop.i:abs4:x.s.ret300s,zzpe.e:-srank:lpe50,zzpe.e:-srank:rete30,zzel.a:-dm:microprice_minus_mid_rel"
for SEED in "zzeg:1/b:x.rema/span" "zzeg:-id:x.(zzgh.nb|cnt-zzgh.nsnap)/zzgh.nsnap"; do
  echo "[$(date +%T)] seed $SEED"
  python -u research/featgen/greedy_fill.py --days 20260430,20261112,20261113 \
    --reports data/reports/r108.txt,data/reports/r113.txt,data/reports/r111b.txt,data/reports/r115.txt --options "$LEADS" --seed "$SEED" \
    --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
    --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet,data/side_r111/zzrema_{date}.parquet,data/side_r115/zzhj_{date}.parquet" \
    --models $DATA/models_gen/B --min-gain 0.0002 --steps 15 --work data/greedy_fill_r120 2>&1 | grep -v RuntimeWarning | cut -c1-400
done
echo "[$(date +%T)] done"
