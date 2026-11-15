#!/usr/bin/env bash
# Round 113 on 97, foreground, read-only, writes only data/reports. ~10-20 min, no model scoring; parallel-safe.
# R108's units fingerprint without logs. Some open columns read like logs (zzxo: thresholds -15.62..-15.41, half the
# training rows <= -15.497 -- e^-15.5 ~ 1.9e-7), so the same fingerprint with the log / log|x| transforms, every open column.
#   bash research/r113.sh 2>&1 | tee data/reports/r113.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/featgen/fingerprint.py --days 20260430,20261112 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" \
  --transforms "log,log|b|" --models $DATA/models_gen/B --top 4 2>&1 | grep -v RuntimeWarning | cut -c1-200
echo "[$(date +%T)] done"
