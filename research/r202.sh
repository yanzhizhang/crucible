#!/usr/bin/env bash
# Round 202 on 97, foreground, read-only; prints only, seconds. Is PM's day-end state float32 or float64? Per key of the
# .jz: share of non-integer values that are exactly a float32, digits as written, three examples.
#   bash research/r202.sh 2>&1 | tee data/reports/r202.txt
set -u
cd /work/crucible
for D in ${DAYS:-20261120}; do
  python -u research/engine/jz_precision.py /work/crucible_data/seed_probe/HS300/$D.jz 2>&1 | cut -c1-200
done
