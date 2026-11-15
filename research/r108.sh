#!/usr/bin/env bash
# Round 108 on 97, foreground, read-only, writes only data/reports. ~10-20 min, no model scoring.
# R105: single-column similarity scans are too weak to tell candidates apart (best fills move a day by ~0.001). A second,
# target-free ruler: the units fingerprint (research/featgen/fingerprint.py) -- does a raw candidate's distribution sit
# where PM's trees put the column's thresholds and root CDF points? Every open column x every candidate we have (tick
# bases, R105 stock / index ratios, R107 state-key ratios, engine keys) x raw transforms, on 0430 and 1112.
# Needs data/side_r107/state_*.parquet (R107 step 1) and data/side_r105/engine_*.parquet (R105b's links).
#   bash research/r108.sh 2>&1 | tee data/reports/r108.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
ls data/side_r107/state_20260430.parquet data/side_r107/state_20261112.parquet data/side_r105/engine_20260430.parquet data/side_r105/engine_20261112.parquet 2>&1 | sed 's/^/  /'
python -u research/featgen/fingerprint.py --days 20260430,20261112 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" \
  --models $DATA/models_gen/B --top 6 2>&1 | grep -v RuntimeWarning | cut -c1-200
echo "[$(date +%T)] done"
