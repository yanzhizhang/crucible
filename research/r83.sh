#!/usr/bin/env bash
# Round 83 on 97, foreground. Read-only on R70's matrices and the model; writes data/reports and data/missing_damage_work only.
# Which missing model columns move today's scores most: each of the 30 heaviest open columns filled with random draws
# from PM's training distribution instead of NaN, through the C++ predictor; change of the scores and of the similarity.
#   bash research/r83.sh 2>&1 | tee data/reports/r83.txt
set -u
cd /work/crucible
R=data/reports
for D in 20261112 20260430; do
  O=$R/stage6_r70_${D:4}
  echo "[$(date +%T)] $D"
  python -u research/featgen/missing_damage.py --date $D --x $O/x.parquet --pred $O/pred.parquet --cal $O/cal_$D.json \
    --models /work/crucible_data/models_gen/B --top 30 2>&1 | cut -c1-220
done
echo "[$(date +%T)] done"
