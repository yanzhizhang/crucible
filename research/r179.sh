#!/usr/bin/env bash
# Round 179 on 97, foreground, read-only, light (~1-2 min, one process): what STARTS an order chain in runS on 20261118.
# R178: orders cluster within seconds of the stock's previous order (re-places); only "quiet" decisions matter for the gate.
# For those: the best single cut of ~30 features as a classifier of "chain start", the best pairs, and feature quantiles.
#   bash research/r179.sh 2>&1 | tee data/reports/r179.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_gate2.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-230
echo "[$(date +%T)] r179 done"
