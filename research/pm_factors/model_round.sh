#!/usr/bin/env bash
# Model line, one round on 97: probe every model month, the engine's state at every 5-minute mark (C++), then ic_score on
# the newest days starting from the last assignment (engine state, samplerS profile cells, (x - y) / z state candidates,
# ablation evidence per assigned column), then the worklist. Foreground, CPUS-limited.
#   CPUS=16 DAYS="20261020 20261021 20261022 20261023 20261026 20261027" bash research/pm_factors/model_round.sh 2>&1 | tee data/reports/model_round2.txt
# MAXF = features scanned by model gain (default 60); FRESH=1 starts from the built-in assignment instead of the last one.
set -u
CPUS=${CPUS:-16}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
DAYS=${DAYS:-"20261020 20261021 20261022 20261023 20261026 20261027"}
MODELS=/data/prod/models
DCP=/data/prod/h/HS300/p1_intrading_dcp
S5=data/reports/stage5; S6=data/reports/stage6; RE=data/reports/engine
mkdir -p $S5 $S6 $RE

echo "### M1 probe every model month"
run research/pm_models_probe.py --models $MODELS --features $MODELS/lgb_fn_v1_209_cte.csv \
  --hstats /data/prod/hstats/HS300 --out $S5 > $S5/probe.txt 2>&1 || tail -5 $S5/probe.txt
grep -E "^[0-9]+ models, months|WARNING|Error" $S5/probe.txt | head -3

echo "### M2 engine state at the 5-minute marks (C++)"
for d in $DAYS; do
  [ -f $DCP/$d.jz ] || { echo "  $d: no state file"; continue; }
  run research/engine/run_cpp.py --date $d --universe pm:$DCP/$d.jz --threads $((CPUS / 2)) --marks five --out data/engine \
    > $RE/marks_$d.txt 2>&1 && grep -E "state at" $RE/marks_$d.txt | sed 's/^/  /' || { echo "  $d: engine FAILED"; tail -3 $RE/marks_$d.txt; }
done

echo "### M3 ic_score on the newest days"
START=""; [ -f $S6/ic_assign.json ] && [ "${FRESH:-0}" != 1 ] && { cp $S6/ic_assign.json $S6/ic_assign_prev.json; START="--assign $S6/ic_assign_prev.json"; }
run research/pm_factors/ic_score.py --dates $(echo $DAYS | tr ' ' ',') --models $MODELS/tk5m $START --max-features ${MAXF:-60} --out $S6 \
  > $S6/ic.txt 2>&1 || tail -5 $S6/ic.txt
grep -E "^20[0-9]{6}: |^drop |accept |IC: start|Error|Traceback" $S6/ic.txt | tail -16

echo
echo "########## SUMMARY (photograph from here) ##########"
echo "model folders (newest file date, folder):"
find $MODELS -name 'model_s1.txt' -printf '%TY%Tm%Td %h\n' 2>/dev/null | sort | awk '{print "  " $0}' | tail -8
sed -n '/^== evidence per assigned column/,/^└/p' $S6/ic.txt | grep -vE "^│ (---|str|f64|u32|i64|bool)|^shape"
run research/pm_factors/model_status.py --stage5 $S5 --assign $S6/ic_assign.json | grep -vE "^│ (---|str|f64|u32|i64|bool)|^shape"
