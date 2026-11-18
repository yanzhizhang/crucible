#!/usr/bin/env bash
# Round 174 on 97, foreground, read-only, light (~1-2 min, one process): exact relations in runS's decision log of 20261118:
# lthr and executp as one- or two-term identities of the line's own values and the config constants, per time segment.
# Truth table for the run_s rebuild (tier 1 of the equivalence round). Reads pm_orders/date=20261118/decisions.parquet only.
#   bash research/r174.sh 2>&1 | tee data/reports/r174.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_ident.py --orders /work/crucible_data/pm_orders --date 20261118 2>&1 | grep -v -E "Warning|warn" | cut -c1-200
echo "[$(date +%T)] r174 done"
