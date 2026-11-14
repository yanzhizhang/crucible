#!/usr/bin/env bash
# Round 86 on 97, foreground. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only. Seconds.
# runS on 20261113: the re-quote loop -- timing of a re-quote against the previous order's send / cancel request / cancel
# confirm, its qty against the unfilled rest, and the chains a made decision starts.
#   bash research/r86.sh 2>&1 | tee data/reports/r86.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] runs_requote $D"
python -u research/pm_factors/runs_requote.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|warn|ret = ret|join_asof" | cut -c1-300
echo "[$(date +%T)] done"
