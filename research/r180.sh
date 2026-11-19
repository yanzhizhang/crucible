#!/usr/bin/env bash
# Round 180 on 97, foreground, read-only, light (~1-2 min, one process): the decoded alg.cpp (sht-demo feat/run-s 07b0cd0)
# checked against the live log of 20261118: A the 15% participation gate and winsdQty = (0.15 mktVol - ourVol) / 0.85;
# B lthr = the close relaxation -w (cr + msr); C the order condition |fa| asf >= cr + hs + lthr as a classifier of
# "order made"; D the no-trade band against the logged executp. Reads pm_orders/date=20261118 and sts_224 only.
#   bash research/r180.sh 2>&1 | tee data/reports/r180.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_alg.py --orders /work/crucible_data/pm_orders --date 20261118 --sts /work/crucible_data/sts_224 2>&1 | grep -v -E "Warning|warn" | cut -c1-240
echo "[$(date +%T)] r180 done"
