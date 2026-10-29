#!/usr/bin/env bash
# One round on 97: our 5-minute model scored on every snapshot (PM-style, a decision per snapshot), trades picked by a
# threshold aimed at 250-2000 a day, held 1 / 5 / 15 minutes, at the real costs (9 bp a round trip). Uses the panel of the
# last round (stage7/h300). Foreground; the log is teed to data/reports/stage8.
set -u
cd /work/crucible
R=data/reports/stage8
mkdir -p $R

echo "### snapshot model (fits 20 models, then scores the test days in parallel; one line per day)"
python -u research/pm_factors/snap_model.py --panel data/reports/stage7/h300/panel.parquet --workers 10 --threads 4 --out $R 2>&1 \
  | tee $R/snap.txt | grep -E "^panel|models fitted|snapshots|Error|error:"

echo
echo "########## SUMMARY (photograph from here) ##########"
grep -E "^panel|rank corr" $R/snap.txt
sed -n '/^== snapshot-triggered/,$p' $R/snap.txt | grep -vE "^│ (---|i64|f64|u32)|^shape"
