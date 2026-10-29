#!/usr/bin/env bash
# One round on 97, on a fixed slice of the machine (CPUS cores via taskset; every library is told the same budget):
#   1. dense panel: every tick day, a row per (stock, 30 s) with the order-flow features (tick_flow) added, walk-forward
#      without 0429/0430 at the 5-minute label (the 5-minute-grid result of the last round is the baseline);
#   2. the same model scored on every snapshot (snap_model), round trips at the real costs.
# Foreground; logs under data/reports/stage7/dense and stage8.
set -u
cd /work/crucible
CPUS=${CPUS:-32}  # cores this round may use; everything else on the box stays free
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
R7=data/reports/stage7/dense
R8=data/reports/stage8
mkdir -p $R7 $R8
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
echo "cpus $CPUS, $(nproc) on the box, $(free -g | awk '/Mem:/ {print $7}') GB free"

echo "### 1/2 dense panel (30 s, order-flow features), walk-forward (one line per day as the panel builds)"
run research/pm_factors/own_model.py --dates all --step 30 --workers 8 --threads $CPUS --cv walk --min-train 5 \
  --drop-dates 20260429,20260430 --out $R7 2>&1 | tee $R7/own.txt \
  | grep -E "days in raw_quotation|rows,|skip|^panel:|cv walk|dropped|out-of-sample IC|Error|error:|^tables"

echo "### 2/2 snapshot model on the dense panel"
run research/pm_factors/snap_model.py --panel $R7/panel.parquet --workers 6 --threads 4 --out $R8 2>&1 \
  | tee $R8/snap.txt | grep -E "^panel|models fitted|snapshots of|Error|error:|^tables"

echo
echo "########## SUMMARY (photograph from here) ##########"
echo "== dense panel, walk-forward, 5-minute label (last round on the 5-minute grid: ridge 0.1067 / lgbm 0.0985)"
grep -E "^panel:|^dropped|out-of-sample IC|^tables" $R7/own.txt
sed -n '/^== out of sample/,/^== LightGBM/p' $R7/own.txt | grep -E "^│ (lgbm|ridge) +┆ [-0-9.]+ +┆" | grep -v "┆ 20"
sed -n '/^== LightGBM gain share/,/^== single-feature/p' $R7/own.txt | grep "^│" | grep -vE "^│ (---|str|f64)" | head -16
sed -n '/^== single-feature IC/,/^== T0 round/p' $R7/own.txt | grep "^│ flow" | head -12
sed -n '/^== T0 round trips/,/^== taker round trips/p' $R7/own.txt | grep -E "^│ lgbm +┆ (300|900) +┆ 0\.0[12]00 "
echo "== snapshot model"
grep -E "^panel|rank corr|^tables" $R8/snap.txt
sed -n '/^== snapshot-triggered/,$p' $R8/snap.txt | grep -vE "^│ (---|i64|f64|u32)|^shape"
