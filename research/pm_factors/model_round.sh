#!/usr/bin/env bash
# Model line, one round on 97: probe every model month (the 202609 refit included), the worklist (gain covered by a
# formula, refit diff), then ic_score on the newest days starting from the last assignment. Foreground, CPUS-limited.
#   CPUS=16 DAYS="20261020 20261021 20261022 20261023 20261026 20261027" bash research/pm_factors/model_round.sh
set -u
CPUS=${CPUS:-16}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
DAYS=${DAYS:-"20261020 20261021 20261022 20261023 20261026 20261027"}
S5=data/reports/stage5; S6=data/reports/stage6
mkdir -p $S5 $S6

echo "### M1 probe every model month"
run research/pm_models_probe.py --models /data/prod/models --features /data/prod/models/lgb_fn_v1_209_cte.csv \
  --hstats /data/prod/hstats/HS300 --out $S5 > $S5/probe.txt 2>&1 || tail -5 $S5/probe.txt
grep -E "^[0-9]+ models, months|WARNING|Error" $S5/probe.txt | head -3

echo "### M2 ic_score on the newest days, starting from the last assignment"
START=""; [ -f $S6/ic_assign.json ] && { cp $S6/ic_assign.json $S6/ic_assign_prev.json; START="--assign $S6/ic_assign_prev.json"; }
run research/pm_factors/ic_score.py --dates $(echo $DAYS | tr ' ' ',') --models /data/prod/models/tk5m $START --out $S6 > $S6/ic.txt 2>&1 || tail -5 $S6/ic.txt
grep -E "IC |Error|Traceback" $S6/ic.txt | tail -4

echo
echo "########## SUMMARY (photograph from here) ##########"
run research/pm_factors/model_status.py --stage5 $S5 --assign $S6/ic_assign.json | grep -vE "^│ (---|str|f64|u32|i64|bool)|^shape"
