#!/usr/bin/env bash
# Round 140 on 97, foreground, read-only on inputs: the veto test (4) for R139's root-matched passes and the best leads.
# Each fill REPLACES the column in the baseline matrix (settled stand-in or NaN) and is scored alone against PM:
# 0430 on the full 3 s grid, plus 1112 / 1113 on the scoring matrices. A pin survives if it is not worse than the
# current formula on 0430 and on both other days. ~10-20 min.
#   bash research/r140.sh 2>&1 | tee data/reports/r140.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
F="zzqf5.q.c1:-dm:zr60@0.01,zzqf5.q.c1:dm:zr60@0.01,zzqf5.q.c1:-id:bid_depth_weighted_distance_5lvl_rel"
F="$F,zzqf8.q.c1:dm:zr60@0.01,zzqf8.q.c1:-dm:zr60@0.01,zzqf8.q.c1:id:overnight_gap_log"
F="$F,zzro.a:dm:vfrac10@0.01,zzro.a:-dm:vfrac10@0.01"
F="$F,zzrg:abs4:depth_weighted_price_5lvl_minus_mid_rel@0.001,zzrg:abs4:mret1800@0.0001"
echo "== 0430 full grid"
python -u research/featgen/single_fill.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" \
  --extra "data/full_0430/extra_{date}.parquet" --fills "$F" --models $DATA/models_gen/B --subs s1,s2,s3,s4 --work data/single_fill_r140 2>&1 | grep -v Warning
for D in 20261112 20261113; do
  echo "== $D scoring matrix"
  python -u research/featgen/single_fill.py --day $D --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
    --fills "$F" --models $DATA/models_gen/B --subs s1,s2,s3,s4 --work data/single_fill_r140 2>&1 | grep -v Warning
done
echo "[$(date +%T)] r140 done"
