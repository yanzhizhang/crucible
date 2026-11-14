#!/usr/bin/env bash
# Round 81 on 97, foreground. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only.
# runS on 20261113: 2,389 decisions made an order but 5,948 orders were sent -- are the rest the execution service
# re-placing after its own cancel? Each sent order against the stock's last made decision, cancel confirm and fill.
#   bash research/r81.sh 2>&1 | tee data/reports/r81.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] runs_es $D"
python -u research/pm_factors/runs_es.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|join_asof" | cut -c1-320
echo "[$(date +%T)] done"
