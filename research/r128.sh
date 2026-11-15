#!/usr/bin/env bash
# Round 128 on 97, foreground, read-only, writes only data/reports. ~5-15 min.
# R127: even on 1.4M rows single fills move 0430 by ~0.001 (64 other columns NaN), so ask the residual itself: which of our
# quantities does PM - ours follow, per time-of-day / drawdown slice (research/featgen/residual_attrib.py).
#   bash research/r128.sh 2>&1 | tee data/reports/r128.txt
set -u
cd /work/crucible
python -u research/featgen/residual_attrib.py --date 20260430 --day data/day_r126 --full data/full_0430 --top 12 2>&1 | cut -c1-400
echo "[$(date +%T)] done"
