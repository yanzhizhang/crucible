#!/usr/bin/env bash
# Round 142 on 97, foreground, read-only on inputs: veto test for R141's best zzro.a pin, industry-demeaned vfrac10 / 100
# (fp 0.034, 13 root points), against dm / neu and both signs; each fill replaces zzro.a alone. 0430 full grid + 1112 / 1113.
#   bash research/r142.sh 2>&1 | tee data/reports/r142.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
F="zzro.a:dmi:vfrac10@0.01,zzro.a:-dmi:vfrac10@0.01,zzro.a:dm:vfrac10@0.01,zzro.a:-dm:vfrac10@0.01,zzro.a:neu:vfrac10@0.01,zzro.a:-neu:vfrac10@0.01"
echo "== 0430 full grid"
python -u research/featgen/single_fill.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" \
  --extra "data/full_0430/extra_{date}.parquet" --fills "$F" --models $DATA/models_gen/B --subs s1,s2,s3,s4 --work data/single_fill_r142 2>&1 | grep -v Warning
for D in 20261112 20261113; do
  echo "== $D scoring matrix"
  python -u research/featgen/single_fill.py --day $D --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
    --fills "$F" --models $DATA/models_gen/B --subs s1,s2,s3,s4 --work data/single_fill_r142 2>&1 | grep -v Warning
done
echo "[$(date +%T)] r142 done"
