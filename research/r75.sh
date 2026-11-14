#!/usr/bin/env bash
# Round 75 on 97, foreground. Read-only on /work/crucible_data/pm_orders (R73's parquet); writes data/reports only.
# runS on 20261113, fourth pass: the morning / afternoon split (lthr == 0 all morning in R74) and what opens a burst of
# orders (episodes of orders sent within 10 s of the previous one's end), plus which chrono log files cover which hours.
#   bash research/r75.sh 2>&1 | tee data/reports/r75.txt
set -u
cd /work/crucible
D=${1:-20261113}
echo "[$(date +%T)] log files of $D: first and last timestamped line, size"
for f in /work/crucible_data/chrono_logs/${D:0:4}-${D:4:2}-${D:6:2}_*; do
  echo "  $(basename $f) $(du -h $f | cut -f1) first $(grep -m1 -oE '^\[20[0-9-]+ [0-9:.]+' $f) last $(tac $f | grep -m1 -oE '^\[20[0-9-]+ [0-9:.]+')"
done
echo "[$(date +%T)] runs_episode $D"
python -u research/pm_factors/runs_episode.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | grep -v -E "Warning|join_asof" | cut -c1-280
echo "[$(date +%T)] done"
