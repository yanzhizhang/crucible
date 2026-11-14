#!/usr/bin/env bash
# Round 70 on 97, foreground. Local only (writes under /work; nothing the baseline reads is overwritten).
# PM's LightGBM model in C++ (research/featgen/cpp/lgbpred.cpp): the model we score with (generation B, livev1_fit1, s1).
#   0 its self-test on made-up trees against ic_score.TreeModel
#   1 0430 and 1112: ic_score's own feature matrix (--dump-x) through the C++ predictor, against ic_score's scores
#     (--dump-pred): the same matrix and the same model, so every score should be the same double
#   2 the whole chain in C++: the C++ feature columns (R69) through the C++ predictor -> similarity to PM's prediction,
#     with the bases rounded to float32 as ic_score does, and in full double precision
#   3 20261112 every 3 s: scores for the whole day from R69's 3-second vectors (the time it takes)
#   4 0430 scored with the deterministic state-engine marks of R68 (data/engine_r68a) instead of data/engine_m7
#   bash research/r70.sh 2>&1 | tee data/reports/r70.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
C1=$R/stage6_r41c_gate/ic_assign_gated.json
G=$DATA/models_gen/B
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99 --models $G --dump-x --dump-pred"
args_of() { if [ $1 = 20260430 ]; then echo "feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz"; else echo "dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/$1.jz"; fi; }
say "0 self-test"
python -u research/featgen/pred_day.py --selftest --work data/lgbpred_work 2>&1 | tail -2 | cut -c1-200
for D in 20260430 20261112; do
  set -- $(args_of $D)
  O=$R/stage6_r70_${D:4}
  say "1 $D: ic_score's matrix and scores"
  CRUCIBLE_SOURCE=$1 python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks $2 --engine-dir $3 --full-book $4 --seed-dir $5 --out $O > $R/r70_${D}_ic.txt 2>&1
  echo "  similarity $(ic $R/r70_${D}_ic.txt)  $(grep -E 'Error|Traceback' $R/r70_${D}_ic.txt | head -1 | cut -c1-120)"
  python -u research/featgen/pred_day.py --date $D --x $O/x.parquet --models $G --check $O/pred.parquet --out data/featpred_r70/same_matrix 2>&1 | cut -c1-300
  say "2 $D: C++ columns -> C++ model"
  for mode in float32 double; do
    python -u research/featgen/vec_day.py --date $D --base data/featgen_r67/base_$D.parquet --full-book data/featgen_book/date=$D.parquet --engine $3/marks_$D.parquet \
      --side $O/side_$D.npz --cal $O/cal_$D.json $([ $mode = float32 ] && echo --float32) --out data/featvec_r70_$mode > $R/r70_${D}_vec_$mode.txt 2>&1 || tail -3 $R/r70_${D}_vec_$mode.txt
    echo "  -- $mode bases"
    python -u research/featgen/pred_day.py --date $D --x data/featvec_r70_$mode/vec_$D.parquet --models $G --check $O/pred.parquet --out data/featpred_r70/$mode 2>&1 | cut -c1-300
  done
done
say "3 20261112 every 3 s: scores"
python -u research/featgen/pred_day.py --date 20261112 --x data/featvec_3s/vec_20261112.parquet --models $G --out data/featpred_3s 2>&1 | cut -c1-300
ls -la data/featpred_3s/ 2>&1 | sed 's/^/  /'
say "4 0430 with the deterministic engine marks (r41c: .6282 with data/engine_m7)"
if [ -f data/engine_r68a/marks_20260430.parquet ]; then
  CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $C1 ${SCORE% --dump-x --dump-pred} --marks rand:240:7 --engine-dir data/engine_r68a \
    --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir $P1 --out $R/stage6_r70_0430_eng > $R/r70_0430_eng.txt 2>&1
  echo "  similarity $(ic $R/r70_0430_eng.txt)  $(grep -E 'Error|Traceback' $R/r70_0430_eng.txt | head -1 | cut -c1-120)"
else
  echo "  no data/engine_r68a/marks_20260430.parquet: skipped"
fi
say "done"
