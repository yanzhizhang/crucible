#!/usr/bin/env bash
# Round 144 on 97, foreground, read-only on inputs: the conditional-count test (featgen/cond_counts.py). Every internal node
# of PM's trees stores how many training rows went left, so a node whose path splits only on columns we hold exactly gives
# P(x_k <= t | path) on PM's own rows; a proxy can match the marginal but not these. Synthetic check: true input 0.002,
# corr-0.85 proxy 0.009, same-marginal decoy 0.042 (root share alone: all ~0.001).
#  1 calibrate: hide each of the 4 PM-state columns in turn, rank its true values among the day's bases -> the test's power
#  2 zzro.a: dmi / dm / neu vfrac10 x 0.01, both signs; zzqf5.q.c1 / zzqf8.q.c1: zr60 x 0.01 both signs vs the stand-ins
# 0430 full grid (the trusted day). ~20-40 min.
#   bash research/r144.sh 2>&1 | tee data/reports/r144.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
A="--day 20260430 --x-dir data/full_0430 --base data/full_0430/base_{date}.parquet --extra data/full_0430/extra_{date}.parquet --models $DATA/models_gen/B"
echo "== 1 calibrate"
python -u research/featgen/cond_counts.py $A --calibrate --max-bases 300 2>&1 | grep -v Warning
echo "== 2 zzro.a"
python -u research/featgen/cond_counts.py $A --target zzro.a --cands "dmi:vfrac10@0.01,-dmi:vfrac10@0.01,dm:vfrac10@0.01,-dm:vfrac10@0.01,neu:vfrac10@0.01,-neu:vfrac10@0.01" 2>&1 | grep -v Warning
for C in zzqf5.q.c1 zzqf8.q.c1; do
  echo "== 2 $C"
  python -u research/featgen/cond_counts.py $A --target $C --cands "dm:zr60@0.01,-dm:zr60@0.01,dmi:zr60@0.01,-dmi:zr60@0.01,-id:bid_depth_weighted_distance_5lvl_rel,id:overnight_gap_log" 2>&1 | grep -v Warning
done
echo "[$(date +%T)] r144 done"
