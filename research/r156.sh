#!/usr/bin/env bash
# Round 156 on 97, foreground, read-only on inputs. R155's calibration put only 2 of the 4 exact columns first (zzvx.a 280 / 394,
# zzxu.a 181 / 224) on 20261112 with generation C. Is that the test, or the day / model? Same calibration on:
#  apr  20260430 (a day inside generation B's training window, full 3 s grid) with generation B: no period shift
#  nov  20261112 with generation B (the model our columns were tuned on): separates the model from the day
# Only the 4 exact columns as targets. If apr also fails, the conditional test is not usable on our matrix and R155 a / b are void.
#   bash research/r156.sh apr 2>&1 | tee data/reports/r156_apr.txt     (same for nov)
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?apr|nov}" in
apr) python -u research/featgen/cond_counts.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" --extra "data/full_0430/extra_{date}.parquet" \
       --target zzup,zzvx.a,zzxu.a,zzxu.b --trusted all --pool --pool-tr id,-id,dm,-dm,abs4,srank --models $DATA/models_gen/B --rows 200000 --max-nodes 2500 --workers 32 --top 6 2>&1 \
       | grep --line-buffered -v Warning | stdbuf -oL cut -c1-200 ;;
nov) python -u research/featgen/cond_counts.py --day 20261112 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
       --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" \
       --target zzup,zzvx.a,zzxu.a,zzxu.b --trusted all --pool --pool-tr id,-id,dm,-dm,abs4,srank --models $DATA/models_gen/B --rows 150000 --max-nodes 2500 --workers 32 --top 6 2>&1 \
       | grep --line-buffered -v Warning | stdbuf -oL cut -c1-200 ;;
esac
echo "[$(date +%T)] r156 $1 done"
