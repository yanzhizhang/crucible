#!/usr/bin/env bash
# Round 61 on 97, foreground, local only (writes under /work). After R60: 85 of 89 base quantities agreed on real ticks;
# lpe5 was rounding noise near zero (now an absolute tolerance), and the snapshot-signed flows (simb*) differ on ~1.5 % of
# cells because the Python definition compares a trade price with exp(log(mid)): on a trade exactly at the mid the last
# bit of exp / log decides the sign. CRUCIBLE_SIMB_EXACT=1 compares with the mid itself (Python and C++ alike).
#   0 self-tests: featgen (default and exact), featvec (the formula layer, all 144 columns)
#   1 featgen against Python on 20261112 and 20260430 (8 stocks each), default and exact
#   2 what the exact variant does to the similarity: r41c on 0430 and on 1112, default against exact
#     NOTE ic_score.py changed inside generic(), so every cached build is rebuilt once (a few minutes a day).
#   bash research/r61.sh 2>&1 | tee data/reports/r61.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
C1=$R/stage6_r41c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
say "0 self-tests"
python -u research/featgen/selftest.py --work data/featgen_selftest > $R/r61_self0.txt 2>&1; echo "  featgen default: exit $?, $(tail -1 $R/r61_self0.txt)"
CRUCIBLE_SIMB_EXACT=1 python -u research/featgen/selftest.py --work data/featgen_selftest > $R/r61_self1.txt 2>&1; echo "  featgen exact: exit $?, $(tail -1 $R/r61_self1.txt)"
grep -hE "simb" $R/r61_self0.txt $R/r61_self1.txt | cut -c1-175
python -u research/featgen/selftest_vec.py --work data/featvec_selftest > $R/r61_selfv.txt 2>&1; echo "  featvec: exit $?, $(tail -1 $R/r61_selfv.txt)"
grep -E "OFF|Error" $R/r61_selfv.txt | head -10 | cut -c1-200
say "1 featgen against Python on real days"
for X in 0 1; do
  echo "  -- exact=$X"
  CRUCIBLE_SIMB_EXACT=$X CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED --marks rand:240 --check 8 --out data/featgen_x$X 2>&1 | grep -vE "^  ok .*cells differing +0 " | cut -c1-230
  CRUCIBLE_SIMB_EXACT=$X CRUCIBLE_SOURCE=feitu python -u research/featgen/run_day.py --date 20260430 --state $P1/20260430.jz --seed-dir $P1 --marks rand:240:7 --check 8 --out data/featgen_x$X 2>&1 | grep -vE "^  ok .*cells differing +0 " | cut -c1-230
done
say "2 r41c similarity, default against exact"
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99"
for X in 0 1; do
  CRUCIBLE_SIMB_EXACT=$X CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $C1 $SCORE --models $DATA/models_gen/B \
    --marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir $P1 --out $R/stage6_r61_0430_x$X > $R/r61_0430_x$X.txt 2>&1
  echo "  0430 exact=$X similarity $(ic $R/r61_0430_x$X.txt)   (r41c before: .6282)"; grep -E "Error|Traceback" $R/r61_0430_x$X.txt | head -2
  CRUCIBLE_SIMB_EXACT=$X CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261112 --assign $C1 $SCORE --models $DATA/models_gen/B \
    --marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $SEED --out $R/stage6_r61_1112_x$X > $R/r61_1112_x$X.txt 2>&1
  echo "  1112 exact=$X similarity $(ic $R/r61_1112_x$X.txt)   (r41c before: .6558)"; grep -E "Error|Traceback" $R/r61_1112_x$X.txt | head -2
done
say "done"
