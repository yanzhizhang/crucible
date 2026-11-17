#!/usr/bin/env bash
# Round 167 on 97, foreground, read-only on inputs, ~5 min: R166's part 3 with the right candidate names (the marks' averages
# are read as "x.<key>", without ".sum"). Our zzvx.a replaced by zzvx.rEma (and controls) in the scoring matrix: change in
# agreement with PM's logged prediction, generations C and B, 1112 and 1113. Uses data/engine_r166/marks_*.parquet from R166.
#   bash research/r167.sh 2>&1 | tee data/reports/r167.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for G in C B; do
  for D in 20261112 20261113; do
    echo "######## single_fill $D, generation $G"
    python -u research/featgen/single_fill.py --day $D --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "data/engine_r166/marks_{date}.parquet" \
      --fills "zzvx.a:id:x.zzvx.rEmaD,zzvx.a:1/b:x.zzvx.rEmaD,zzvx.a:sqrt:x.zzvx.rEmaD,zzvx.a:id:x.zzvx.trDEmaD,zzvx.a:id:x.zzvx.trEmaD,zzvx.a:id:x.zzvx.high" \
      --models $DATA/models_gen/$G --subs s1,s2,s3,s4 --work data/single_fill_r167 2>&1 | grep -v Warning | cut -c1-220
  done
done
echo "[$(date +%T)] r167 done"
