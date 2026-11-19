#!/usr/bin/env bash
# Round 181 on 97, foreground, read-only, light (~1 min): the clock behind lthr = -w (cr + msr / 2) and behind the vpen ramp
# on 20261118: each implied weight regressed on the receive time (start, end, residual in seconds), and whether the implied
# time is a whole exchange second. Tells which timestamp and which end time the rebuild must use.
#   bash research/r181.sh 2>&1 | tee data/reports/r181.txt
set -u
cd /work/crucible
python -u research/pm_factors/runs_w.py --orders /work/crucible_data/pm_orders --date 20261118 --sts /work/crucible_data/sts_224 2>&1 | grep -v -E "Warning|warn|stsfile" | cut -c1-240
echo "[$(date +%T)] r181 done"
