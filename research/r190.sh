#!/usr/bin/env bash
# Round 190 on 97, foreground, read-only; prints only. Light (a join of ~40k decisions against one day's snapshots).
# runS order replay on 20261118: decoded cOrd (side, price at the touch, quantity with room / board / 5e6 / top-of-book / lot
# rules) against PM's SO lines, with the book taken from 224's own feed (s224); then the day's order life for live-vs-sim.
#   bash research/r190.sh 2>&1 | tee data/reports/r190.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118}; do
  [ -e $DATA/pm_orders/date=$D/decisions.parquet ] || { echo "== $D: no pm_orders tables (run pm_orders.py for the day first)"; continue; }
  C=$DATA/catalog_s224.duckdb; [ -d $DATA/s224_raw/kind=quotation/date=$D ] || C=$DATA/catalog_dysl2.duckdb
  echo "[$(date +%T)] $D book from $C"
  python -u research/pm_factors/runs_ord.py --orders $DATA/pm_orders --date $D --catalog $C --show 15 2>&1 | grep -v -E "Warning|warn" | cut -c1-330
done
echo "[$(date +%T)] r190 done"
