#!/usr/bin/env bash
# One round on 97: rebuild the all-days panel (now with quotes 30 and 60 minutes ahead), then our own model walk-forward without
# 0429/0430 (PM's replay days; the label check runs before they are dropped) on 5, 15, 30 and 60-minute labels.
# Foreground; each step tees its log to data/reports/stage7.
set -u
cd /work/crucible
R=data/reports/stage7
DROP=20260429,20260430
COMMON="--cv walk --min-train 5 --drop-dates $DROP"
mkdir -p $R/h300 $R/h900 $R/h1800 $R/h3600

echo "### 1/4 rebuild the panel, walk-forward, 5-minute label (prints one line per day)"
python -u research/pm_factors/own_model.py --dates all $COMMON --out $R/h300 2>&1 | tee $R/h300/own.txt \
  | grep -E "days in raw_quotation|rows,|skip|cv walk|dropped|out-of-sample IC|Error|error:"
for h in 900 1800 3600; do
  echo "### horizon $h s"
  python -u research/pm_factors/own_model.py --from-panel $R/h300/panel.parquet $COMMON --horizon $h --out $R/h$h 2>&1 \
    | tee $R/h$h/own.txt | grep -E "cv walk|out-of-sample IC|Error|error:"
done

echo
echo "########## SUMMARY (photograph from here) ##########"
sed -n '/^label check/,/^dropped/p' $R/h300/own.txt | grep -vE "^│ (---|i64|i32|f64|u32)|^shape|^dropped"
for h in 300 900 1800 3600; do
  f=$R/h$h/own.txt
  echo "== label $h s"
  grep -E "^dropped|out-of-sample IC" $f
  [ $h = 300 ] && sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ lgbm +┆ 20" | cut -c1-110
  sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ (lgbm|ridge) +┆ [-0-9.]+ +┆" | grep -v "┆ 20"
  holds=$h; [ $h = 300 ] && holds="300|900|1800|3600"
  sed -n '/^== T0 round trips/,/^== taker round trips/p' $f | grep -E "^│ lgbm +┆ ($holds) +┆ 0\.0[125]00 "
  sed -n '/^== taker round trips by quoted spread/,$p' $f | grep -E "^│ lgbm +┆ $h +┆ top5% "
done
