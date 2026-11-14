#!/usr/bin/env bash
# Round 82 on 97, foreground. Read-only on the catalogs and R80's scores; writes data/reports only.
# The model, judged on returns: cross-sectional rank IC of the C++ chain's scores against forward mid returns (30 s .. 900 s),
# next to PM's own prediction of the day where pm_target_r22 has it -- does the part of the model we reproduce predict,
# and how far is it from PM's live signal on the same marks. Needs R80.
#   bash research/r82.sh 2>&1 | tee data/reports/r82.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261009 20261110 20261111; do
  if [ $D -ge 20261104 ]; then S=dysl2; CAT=$DATA/catalog_dysl2.duckdb; else S=feitu; CAT=$DATA/catalog.duckdb; fi
  echo "[$(date +%T)] $D ($S)"
  CRUCIBLE_SOURCE=$S python -u research/featgen/pred_ic.py --date $D --scores data/backfill_r80/days/$D/pred/pred_$D.parquet \
    --target $DATA/pm_target_r22/date=$D.parquet --catalog $CAT --every 30 2>&1 | cut -c1-300
done
echo "[$(date +%T)] done"
