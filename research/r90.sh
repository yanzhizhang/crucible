#!/usr/bin/env bash
# Round 90 on 97, foreground, after R89. Read-only on inputs; writes data/side_r90, data/implied_scan_work and data/reports.
# R85: for the open columns that move today's scores most (zzbt.b, zzaa.b, zzxo, zzpp.b) no C++ tick base helps -- every
# fill is worse than NaN. So the next candidates are not tick quantities: the CED daily table + PM's stsfile inputs (one
# value per stock and day) and the state engine's keys at the marks. Implied scan of the same columns with those only.
#   bash research/r90.sh 2>&1 | tee data/reports/r90.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
echo "[$(date +%T)] daily tables"
CRUCIBLE_SOURCE=feitu python -u research/featgen/side_day.py --date 20260430 --state $P1/20260430.jz --models $DATA/models_gen/B --out data/side_r90 --daily 2>&1 | tail -1 | cut -c1-200
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/side_day.py --date 20261112 --state $SEED/20261112.jz --models $DATA/models_gen/B --out data/side_r90 --daily 2>&1 | tail -1 | cut -c1-200
ln -sfn /work/crucible/data/engine_m7/marks_20260430.parquet data/side_r90/engine_20260430.parquet
ln -sfn /work/crucible/data/engine/marks_20261112.parquet data/side_r90/engine_20261112.parquet
echo "[$(date +%T)] implied scan, daily + state candidates"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target zzbt.b,zzaa.b,zzxo,zzpp.b,zzpe.e,zzop.i,zzhj \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --extra "data/side_r90/daily_{date}.parquet,data/side_r90/engine_{date}.parquet" \
  --only-extra --models $DATA/models_gen/B --top 8 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
