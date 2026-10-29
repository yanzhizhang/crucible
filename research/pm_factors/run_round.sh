#!/usr/bin/env bash
# One round on 97: our own model, walk-forward on the saved all-days panel without 0429/0430 (PM's replay days, label check
# first), on the 5- and the 15-minute label. No tick rebuild. Foreground; each step tees its log to data/reports/stage7.
set -u
cd /work/crucible
R=data/reports/stage7
P=$R/all/panel.parquet
DROP=20260429,20260430
mkdir -p $R/walk_clean $R/walk_h900

echo "### 1/2 walk-forward, 5-minute label, without $DROP"
python -u research/pm_factors/own_model.py --from-panel $P --cv walk --min-train 5 --drop-dates $DROP --out $R/walk_clean 2>&1 \
  | tee $R/walk_clean/own.txt | grep -E "cv walk|dropped|out-of-sample IC|Error|error:"

echo "### 2/2 walk-forward, 15-minute label, without $DROP"
python -u research/pm_factors/own_model.py --from-panel $P --cv walk --min-train 5 --drop-dates $DROP --horizon 900 --out $R/walk_h900 2>&1 \
  | tee $R/walk_h900/own.txt | grep -E "cv walk|dropped|out-of-sample IC|Error|error:"

echo
echo "########## SUMMARY (photograph from here) ##########"
sed -n '/^label check/,/^dropped/p' $R/walk_clean/own.txt | grep -vE "^│ (---|i64|f64|u32)|^shape|^dropped"
for cv in walk_clean walk_h900; do
  f=$R/$cv/own.txt
  echo "== $cv"
  grep -E "^dropped|out-of-sample IC" $f
  sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ lgbm +┆ 20" | cut -c1-110
  sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ (lgbm|ridge) +┆ [-0-9.]+ +┆" | grep -v "┆ 20"
  sed -n '/^== T0 round trips/,/^== taker round trips/p' $f | grep -E "^│ lgbm +┆ (300|900) +┆ 0\.0[125]00 "
  sed -n '/^== taker round trips by quoted spread/,$p' $f | grep -E "^│ lgbm .*┆ top5% "
done
