#!/usr/bin/env bash
# Round 72 on 97, foreground. Reads logs already on 97; writes data/reports, pm_signals and pm_orders under /work only.
# runS (PM's order rule) on the two normal days 20261113 and 20261114, with the daily target in view: PM steers each stock
# from yesterday's holding toward the day's base position.
#   0 what is on 97 for the two days (chrono logs, position logs, base position csvs) -- nothing is pulled here
#   1 pm_signals (decision lines -> state parquet) and pm_orders (census, thresholds, decision -> order, timing)
#   2 runs_sample / runs_fit: executp as a target, the order gate
#   3 pm_poslog: pos / executp against chrono's position log
#   4 runs_target: start-of-day gap, share closed in the day, day-to-day recalibration, both sides' thresholds
#   bash research/r72.sh 2>&1 | tee data/reports/r72.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
DAYS="${1:-20261113,20261114}"
say() { echo "[$(date +%T)] $*"; }
say "0 inventory"
for D in ${DAYS//,/ }; do
  echo "  $D: log files $(ls $DATA/chrono_logs 2>/dev/null | grep -cE "(${D:0:4}-${D:4:2}-${D:6:2}_|_${D}_)"), position-log csvs $(ls $DATA/chrono_poslog/$D/position_log/*.csv 2>/dev/null | wc -l), base position csvs: $(ls $DATA/chrono_poslog/$D/base_position_*.csv 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"
done
for D in ${DAYS//,/ }; do
  say "1 $D pm_signals, pm_orders"
  python -u research/pm_factors/pm_signals.py --logs $DATA/chrono_logs --dates $D --out $DATA/pm_signals > $R/r72_signals_$D.txt 2>&1 || echo "  pm_signals FAILED: $(tail -2 $R/r72_signals_$D.txt | cut -c1-200)"
  python -u research/pm_factors/pm_orders.py --logs $DATA/chrono_logs --dates $D --out $DATA/pm_orders > $R/r72_orders_$D.txt 2>&1 || echo "  pm_orders FAILED: $(tail -2 $R/r72_orders_$D.txt | cut -c1-200)"
  cut -c1-230 $R/r72_orders_$D.txt | head -150
  say "2 $D runs_sample, runs_fit"
  python -u research/pm_factors/runs_sample.py --orders $DATA/pm_orders --date $D > $R/r72_sample_$D.txt 2>&1; head -60 $R/r72_sample_$D.txt | cut -c1-230
  python -u research/pm_factors/runs_fit.py --orders $DATA/pm_orders --date $D 2>&1 | cut -c1-230
done
say "3 pm_poslog"
python -u research/pm_factors/pm_poslog.py --poslog $DATA/chrono_poslog --state $DATA/pm_signals/state --dates $DAYS 2>&1 | cut -c1-230
say "4 runs_target"
python -u research/pm_factors/runs_target.py --poslog $DATA/chrono_poslog --orders $DATA/pm_orders --dates $DAYS 2>&1 | cut -c1-260
say "done"
