#!/usr/bin/env bash
# Round 73 on 97, foreground. Reads logs already on 97; writes data/reports and /work/crucible_data/pm_orders only.
# runS on 20261113, second pass (R72 settled the units: pos = qty - base, asq = yesterday_qty, trade = executp - cpos):
#   1 pm_orders again: the Po lines now take their side from catchRun's place_order line, and the order life
#     (cancel requests, OnECM cancel confirms, OnEOM fills) is saved as life.parquet
#   2 runs_exec: the formula of executp, the order gate, the order price against the touch, sent -> cancel / fill
#   bash research/r73.sh 2>&1 | tee data/reports/r73.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
D=${1:-20261113}
say() { echo "[$(date +%T)] $*"; }
say "1 pm_orders $D"
python -u research/pm_factors/pm_orders.py --logs $DATA/chrono_logs --dates $D --out $DATA/pm_orders > $R/r73_orders_$D.txt 2>&1 || echo "  pm_orders FAILED: $(tail -3 $R/r73_orders_$D.txt | cut -c1-200)"
grep -E "^place_order|^-- 7|^   (buy|sell|SO):" $R/r73_orders_$D.txt | cut -c1-260
say "2 runs_exec $D"
python -u research/pm_factors/runs_exec.py --orders $DATA/pm_orders --poslog $DATA/chrono_poslog --date $D 2>&1 | cut -c1-260
say "done"
