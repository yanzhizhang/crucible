#!/usr/bin/env bash
# Round 33 on 97, foreground: translate runS (PM's order rules) on 1112, the first day whose plugin lines go through chrono's
# logger in one format (pm_signals: order match 0.990, corr 0.993). Read-only on everything but data/reports and pm_orders.
#   1 pm_orders: census of every plugin line shape, snapshot -> decision, thresholds, decision -> order, timing
#   2 pm_decisions: the constants, executp against its caps and least squares on candidate terms
#   3 pm_poslog: pos / executp against chrono's position log (holding, sellable, base)
#   bash research/r33.sh 2>&1 | tee data/reports/r33.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
D=${1:-20261112}
say() { echo "[$(date +%T)] $*"; }

say "1 pm_orders $D"
python -u research/pm_factors/pm_orders.py --logs $DATA/chrono_logs --dates $D --out $DATA/pm_orders > $R/r33_orders_$D.txt 2>&1 \
  || say "  pm_orders FAILED: $(tail -3 $R/r33_orders_$D.txt)"
cut -c1-230 $R/r33_orders_$D.txt | head -250

say "2 pm_decisions $D"
python -u research/pm_factors/pm_decisions.py --state $DATA/pm_signals/state --dates $D > $R/r33_decisions_$D.txt 2>&1 \
  || say "  pm_decisions FAILED: $(tail -3 $R/r33_decisions_$D.txt)"
cut -c1-230 $R/r33_decisions_$D.txt | head -120

say "3 pm_poslog $D"
python -u research/pm_factors/pm_poslog.py --poslog $DATA/chrono_poslog --state $DATA/pm_signals/state --dates $D > $R/r33_poslog_$D.txt 2>&1 \
  || say "  pm_poslog FAILED: $(tail -3 $R/r33_poslog_$D.txt)"
cut -c1-230 $R/r33_poslog_$D.txt | head -60
say "done"
