#!/usr/bin/env bash
# Our own T0 strategy, one round on 97 (foreground, CPUS-limited):
#   1. the dense prediction panel (own_model, 30 s marks, walk-forward, PM-replay days dropped), predictions saved;
#      an existing panel is reused (REBUILD=1 rebuilds it from the ticks);
#   2. the passive (maker) T0 backtest on those predictions (maker_t0).
#   CPUS=16 bash research/strategy/run_strategy.sh 2>&1 | tee data/reports/strategy_round1.txt
set -u
CPUS=${CPUS:-256}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
P=data/reports/stage7/dense; O=data/reports/strategy
mkdir -p $P $O

echo "### 1 prediction panel (30 s marks, walk-forward), predictions saved"
if [ -f $P/panel.parquet ] && [ "${REBUILD:-0}" != 1 ]; then
  run research/pm_factors/own_model.py --from-panel $P/panel.parquet --cv walk --drop-dates 20260429,20260430 --threads $((CPUS / 2)) \
    --save-preds --out $P > $P/model.txt 2>&1 || tail -5 $P/model.txt
else
  run research/pm_factors/own_model.py --dates all --step 30 --workers $((CPUS / 4)) --threads $((CPUS / 2)) --cv walk \
    --drop-dates 20260429,20260430 --save-preds --out $P > $P/model.txt 2>&1 || tail -5 $P/model.txt
fi
grep -E "out-of-sample IC|predictions ->|Error|Traceback" $P/model.txt | tail -4

echo "### 2 passive T0 backtest"
run research/strategy/maker_t0.py --preds $P/preds.parquet --targets 100,300,1000 --holds 300,900,1800 --queues 1.0,0.5 \
  --out $O > $O/maker.txt 2>&1 || tail -5 $O/maker.txt

echo
echo "########## SUMMARY (photograph from here) ##########"
grep -E "out-of-sample IC" $P/model.txt | tail -2
sed -n '/^== passive T0/,$p' $O/maker.txt | grep -vE "^│ (---|str|f64|u32|i64|bool)|^shape"
