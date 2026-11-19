#!/usr/bin/env bash
# Round 182 on 97, foreground, read-only, light (~1-2 min, one process): three follow-ups on the decoded alg.cpp against the
# 20261118 log: (1) which snapshots get a decision line (by time of day; in-band before 13:00?), (2) the 6.5% of decisions
# where executp is not the band target, by side (cFtp cuts buys; the sellable cap on sells), (3) what the order rule's
# false positives have in common. Reads pm_orders/date=20261118 only.
#   bash research/r182.sh 2>&1 | tee data/reports/r182.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_alg2.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-250
echo "[$(date +%T)] r182 done"
