#!/usr/bin/env bash
# Round 89 on 97, foreground, after R88. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only.
# runS on 20261113: what decides "made an order" among decisions passing the visible gate -- a small LightGBM classifier
# (the env's lightgbm, native API) on the decision row, its changes since the stock's previous decision and the order state;
# held-out AUC, importance, made rate by bin for the top features. About a minute.
#   bash research/r89.sh 2>&1 | tee data/reports/r89.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] runs_gate_fit $D"
python -u research/pm_factors/runs_gate_fit.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|join_asof" | cut -c1-400
echo "[$(date +%T)] done"
