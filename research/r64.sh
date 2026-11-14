#!/usr/bin/env bash
# Round 64 on 97, foreground. Local only (writes under /work).
# The exact-mid tie rule of the snapshot-signed flows is now the default (the user's decision; CRUCIBLE_SIMB_EXACT=0 = the
# old way). ic_score.generic changed again, so every day's cached build is rebuilt once on its next use.
# Two switches are measured here, each on 0430 and 1112, r41c similarity:
#   CRUCIBLE_LPE_SNAP=1   a log price over its own EMA below 1e-12 in size is 0 (the rounding left once the price stood still)
#   CRUCIBLE_BASE64=1     ic_score keeps its bases in float64 instead of float32 (float32 ties the rpe* values near 1)
# Then, with the snap on: the C++ columns against ic_score's matrix again (R63: 140 / 141 of 144), with the stocks and
# times of the differing cells for any column still off (gen.stoch on 0430 is the open one).
#   bash research/r64.sh 2>&1 | tee data/reports/r64.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
C1=$R/stage6_r41c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99 --models $DATA/models_gen/B --dump-x"
args_of() { if [ $1 = 20260430 ]; then echo "feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz"; else echo "dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/$1.jz"; fi; }
score() {  # score <day> <tag> [VAR=1 ...]
  local D=$1 tag=$2; shift 2
  set -- $(args_of $D) "$@"
  local src=$1 marks=$2 eng=$3 fb=$4 seed=$5; shift 6
  env "$@" CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks $marks --engine-dir $eng --full-book $fb --seed-dir $seed \
    --out $R/stage6_r64_${D:4}_$tag > $R/r64_${D}_$tag.txt 2>&1
  echo "  $D $tag: similarity $(ic $R/r64_${D}_$tag.txt)  $(grep -E 'Error|Traceback' $R/r64_${D}_$tag.txt | head -1 | cut -c1-120)"
}
say "1 similarity (r41c before: 0430 .6282, 1112 .6558)"
for D in 20260430 20261112; do
  score $D default
  score $D snap CRUCIBLE_LPE_SNAP=1
  score $D f64 CRUCIBLE_BASE64=1
  score $D snap_f64 CRUCIBLE_LPE_SNAP=1 CRUCIBLE_BASE64=1
done
say "2 C++ columns against ic_score's matrix, snap on"
for D in 20260430 20261112; do
  set -- $(args_of $D)
  O=$R/stage6_r64_${D:4}_snap
  CRUCIBLE_LPE_SNAP=1 CRUCIBLE_SOURCE=$1 python -u research/featgen/run_day.py --date $D --state $6 --seed-dir $5 --marks $2 --out data/featgen_r64 2>&1 | tail -1 | cut -c1-200
  python -u research/featgen/vec_day.py --date $D --base data/featgen_r64/base_$D.parquet --full-book $4/date=$D.parquet --engine $3/marks_$D.parquet \
    --side $O/side_$D.npz --cal $O/cal_$D.json --x $O/x.parquet --float32 --out data/featvec_r64 2>&1 | grep -vE "assemble|names nobody" | cut -c1-260
done
say "done"
