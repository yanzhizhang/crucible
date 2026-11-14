#!/usr/bin/env bash
# Round 78 on 97, foreground. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only.
# runS on 20261113: a one-step replay of the send decision (candidate rules scored as classifiers of "made an order"),
# morning and afternoon apart, and the sent qty against candidate sizes.
#   bash research/r78.sh 2>&1 | tee data/reports/r78.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] runs_replay $D"
python -u research/pm_factors/runs_replay.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|join_asof" | cut -c1-200
echo "[$(date +%T)] done"
