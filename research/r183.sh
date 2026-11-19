#!/usr/bin/env bash
# Round 183 on 97, foreground, read-only, light (~1 min): the 700 false positives of the runS order rule on 20261118 against
# the decoded cOrd terms that ARE in the log (the pending leg via pos / pp / cpos, the sellable quantity asq, the room):
# their values on orders vs false positives, then the rule with each term added. Reads pm_orders/date=20261118 only.
#   bash research/r183.sh 2>&1 | tee data/reports/r183.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_alg3.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-320
echo "[$(date +%T)] r183 done"
