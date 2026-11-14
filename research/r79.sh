#!/usr/bin/env bash
# Round 79 on 97, foreground. Read-only on inputs; writes data/side_r79 and data/reports only.
#   1 the top of model_status's worklist (R77 cut it with tail): the heaviest open columns and their closest candidates
#   2 side_day.py: the formula layer's side tables (side_<D>.npz, cal_<D>.json) without ic_score, for 0430 and 1112,
#     compared byte for byte with what ic_score --dump-x wrote in R70 -- the last piece the daily C++ chain took from ic_score
#   bash research/r79.sh 2>&1 | tee data/reports/r79.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
echo "[$(date +%T)] 1 model_status, top of the table"
python -u research/pm_factors/model_status.py --stage5 $R/stage5 --assign $R/stage6_r41c_gate/ic_assign_gated.json --top 30 2>&1 | cut -c1-200 | head -60
echo "[$(date +%T)] 2 side tables without ic_score"
CRUCIBLE_SOURCE=feitu python -u research/featgen/side_day.py --date 20260430 --state $P1/20260430.jz --models $DATA/models_gen/B --out data/side_r79 --check $R/stage6_r70_0430 2>&1 | cut -c1-260
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/side_day.py --date 20261112 --state $SEED/20261112.jz --models $DATA/models_gen/B --out data/side_r79 --check $R/stage6_r70_1112 2>&1 | cut -c1-260
echo "[$(date +%T)] done"
