#!/usr/bin/env bash
# Round 74 on 97, foreground. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only.
# runS on 20261113, third pass: executp and lthr as formulas (log-space least squares), and what still holds orders back
# once |trade| >= 100 and the side's threshold are met (a live order of the stock, a cooldown, an account-wide pacer).
#   bash research/r74.sh 2>&1 | tee data/reports/r74.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] runs_rule $D"
python -u research/pm_factors/runs_rule.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|warn|_methods|join_asof" | cut -c1-260
echo "[$(date +%T)] done"
