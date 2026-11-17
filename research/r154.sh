#!/usr/bin/env bash
# Round 154 on 97, foreground, read-only, ~1 min: the cap behind poscst's low tail (R152: poscst = min(lot(5 x sod.mb), 1e7)
# on ~95% of the stocks). Candidate second caps (net-market-value cap over the price, msr / vr multiples) and the stocks
# the best formula misses, with their numbers.
#   bash research/r154.sh 2>&1 | tee data/reports/r154.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_poscst2.py --orders /work/crucible_data/pm_orders --date 20261113 --sts /work/crucible_data/pm_stsfile 2>&1 | grep -v -E "Warning|warn" | cut -c1-230
echo "[$(date +%T)] r154 done"
