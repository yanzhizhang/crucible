#!/usr/bin/env bash
# Round 63 on 97, foreground. Local only (writes under /work). R63 again with what it showed:
#   - ic_score caches its bases as float32 (base_vec), so its matrix is float32-precise: vec_day --float32 rounds the same way
#   - a stem's "px" is the state's last price (zzxu.last), not the snapshot price: vec_day now feeds that
#   - the snapshot-signed flows tie-break on the last bit: both sides run with CRUCIBLE_SIMB_EXACT=1
# Then the 3-second day in double precision, and how far float32 moves the columns (double against float32, same day).
# The C++ pipeline end to end: base quantities (featgen) + whole-book quantity (full_book parquet, still Python) + state
# engine marks + CED side tables + training-CDF points -> the 144 model columns (featvec), compared column by column with
# the matrix ic_score scores (ic_score --dump-x), on 0430 and 1112. Then one whole day on a 3-second grid, for the time.
#   bash research/r63.sh 2>&1 | tee data/reports/r63.txt
set -u
export CRUCIBLE_SIMB_EXACT=1
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
C1=$R/stage6_r41c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99 --models $DATA/models_gen/B --dump-x"
one() {  # one <day> <source> <marks> <engine dir> <full book dir> <seed dir> <state file>
  local D=$1 src=$2 marks=$3 eng=$4 fb=$5 seed=$6 st=$7 O=$R/stage6_r63_${1:4}
  say "$D: ic_score --dump-x"
  CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks $marks --engine-dir $eng --full-book $fb --seed-dir $seed --out $O > $R/r63_${D}_ic.txt 2>&1
  grep -E "Error|Traceback|feature matrix|^drop " $R/r63_${D}_ic.txt | cut -c1-200 | head -8
  echo "  similarity $(ic $R/r63_${D}_ic.txt)"
  say "$D: featgen base"
  CRUCIBLE_SOURCE=$src python -u research/featgen/run_day.py --date $D --state $st --seed-dir $seed --marks $marks --out data/featgen_r63 2>&1 | cut -c1-220
  say "$D: featvec, against ic_score's matrix"
  python -u research/featgen/vec_day.py --date $D --base data/featgen_r63/base_$D.parquet --full-book $fb/date=$D.parquet --engine $eng/marks_$D.parquet \
    --side $O/side_$D.npz --cal $O/cal_$D.json --x $O/x.parquet --float32 --out data/featvec_r63 2>&1 | cut -c1-240
}
one 20260430 feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz
one 20261112 dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/20261112.jz

say "float32 against double: the same 1112 marks without --float32, compared with ic_score's matrix (how much the rounding moves each column)"
python -u research/featgen/vec_day.py --date 20261112 --base data/featgen_r63/base_20261112.parquet --full-book $DATA/store/pm_factors/full_book_dysl2/date=20261112.parquet   --engine data/engine/marks_20261112.parquet --side $R/stage6_r63_1112/side_20261112.npz --cal $R/stage6_r63_1112/cal_20261112.json   --x $R/stage6_r63_1112/x.parquet --out data/featvec_r63_double 2>&1 | grep -E "corr 0\.|columns agree" | cut -c1-240
say "done"
