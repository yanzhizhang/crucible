#!/usr/bin/env bash
# Round 62 on 97, foreground, AFTER R61. Local only (writes under /work).
# The C++ pipeline end to end: base quantities (featgen) + whole-book quantity (full_book parquet, still Python) + state
# engine marks + CED side tables + training-CDF points -> the 144 model columns (featvec), compared column by column with
# the matrix ic_score scores (ic_score --dump-x), on 0430 and 1112. Then one whole day on a 3-second grid, for the time.
#   bash research/r62.sh 2>&1 | tee data/reports/r62.txt
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
one() {  # one <day> <source> <marks> <engine dir> <full book dir> <seed dir> <state file>
  local D=$1 src=$2 marks=$3 eng=$4 fb=$5 seed=$6 st=$7 O=$R/stage6_r62_${1:4}
  say "$D: ic_score --dump-x"
  CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks $marks --engine-dir $eng --full-book $fb --seed-dir $seed --out $O > $R/r62_${D}_ic.txt 2>&1
  grep -E "Error|Traceback|feature matrix|^drop " $R/r62_${D}_ic.txt | cut -c1-200 | head -8
  echo "  similarity $(ic $R/r62_${D}_ic.txt)"
  say "$D: featgen base"
  CRUCIBLE_SOURCE=$src python -u research/featgen/run_day.py --date $D --state $st --seed-dir $seed --marks $marks --out data/featgen_r62 2>&1 | cut -c1-220
  say "$D: featvec, against ic_score's matrix"
  python -u research/featgen/vec_day.py --date $D --base data/featgen_r62/base_$D.parquet --full-book $fb/date=$D.parquet --engine $eng/marks_$D.parquet \
    --side $O/side_$D.npz --cal $O/cal_$D.json --x $O/x.parquet --out data/featvec_r62 2>&1 | cut -c1-240
}
one 20260430 feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz
one 20261112 dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/20261112.jz

say "20261112 on a 3-second grid: state engine, base, formula layer (no whole-book quantity: 8 columns stay empty)"
G=$(python -c "import numpy as np; print(','.join(repr(float(x)) for x in np.arange(3.0, 14220.0 + 1e-9, 3.0)))")
t0=$(date +%s)
python -u research/engine/run_cpp.py --date 20261112 --universe pm:$SEED/20261112.jz --prev $SEED/20261111.jz --catalog $DATA/catalog_dysl2.duckdb \
  --marks "$G" --threads 32 --out data/engine_3s --work data/engine_3s/work > $R/r62_engine3s.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/r62_engine3s.txt | cut -c1-200)"
tail -2 $R/r62_engine3s.txt | cut -c1-200; t1=$(date +%s)
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED --grid 3 --out data/featgen_3s 2>&1 | tail -2 | cut -c1-220; t2=$(date +%s)
python -u research/featgen/vec_day.py --date 20261112 --base data/featgen_3s/base_20261112.parquet --engine data/engine_3s/marks_20261112.parquet \
  --side $R/stage6_r62_1112/side_20261112.npz --cal $R/stage6_r62_1112/cal_20261112.json --out data/featvec_3s 2>&1 | cut -c1-240; t3=$(date +%s)
echo "  seconds: state engine $((t1 - t0)), base $((t2 - t1)), formula layer $((t3 - t2)), total $((t3 - t0))"
ls -la data/featvec_3s/*.parquet data/engine_3s/*.parquet 2>&1 | sed 's/^/  /'
say "done"
