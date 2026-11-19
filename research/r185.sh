#!/usr/bin/env bash
# Round 185 on 97, foreground, read-only, light (~30 s): PM's order trace (SO lines) against chrono's ORD / SEND lines on
# 20261118: is price / quantity swapped between PM's library and chrono? Reads pm_orders/date=20261118 (po + trk) only.
#   bash research/r185.sh 2>&1 | tee data/reports/r185.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_po.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-260
echo "[$(date +%T)] r185 done"
