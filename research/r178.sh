#!/usr/bin/env bash
# Round 178 on 97, foreground, read-only, light (~1-2 min, one process): three questions from the user about runS on the live
# day 20261118, from the parsed log (pm_orders): (1) what turns a decision into an order (lot rounding, the side threshold,
# a live order); (2) cross or join, and whether any cap binds; (3) does the position go back to the base in the afternoon.
#   bash research/r178.sh 2>&1 | tee data/reports/r178.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_gate.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-230
echo "[$(date +%T)] r178 done"
