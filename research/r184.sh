#!/usr/bin/env bash
# Round 184 on 97, foreground, read-only, light (~1 min): the runS order rule's false positives on 20261118 by board, stock,
# price and time, and the rule with "a STAR order needs 200 shares" added. Reads pm_orders/date=20261118 only.
#   bash research/r184.sh 2>&1 | tee data/reports/r184.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_alg4.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-320
echo "[$(date +%T)] r184 done"
