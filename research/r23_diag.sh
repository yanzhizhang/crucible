#!/usr/bin/env bash
# The diagnostic half of R23, which never ran (its scan / gate / held-out steps are superseded by R24). Run AFTER r24.sh,
# which builds the hstats view and the 1109 / 1110 engine state this needs. Read-only outside /work.
#   1 do the model generations mean the same by each column (gen_compare)
#   3 invert PM's model for the 15 open columns with the most gain (--implied) + residual breakdown, train days and 224 days
#   bash research/r23_diag.sh 2>&1 | tee data/reports/r23_diag.txt
set -u
cd /work/crucible
R=data/reports
D=/work/crucible_data
TGT=$D/pm_target_r22
HV=$D/hstats_view/HS300
FBF=$D/store/pm_factors/full_book
FBD=$D/store/pm_factors/full_book_dysl2
R17=$R/stage6_r17/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
[ -d $HV ] || { say "no $HV: run research/r24.sh first"; exit 1; }

say "1 model generations: does each column mean the same in every generation"
python -u research/pm_factors/gen_compare.py --models /data/prod/models/tk5m --fit livev1_fit1 --out $R/gen_compare.csv 2>&1 | head -60

say "3 implied values of the 15 open columns with the most gain (R17), plus the residual breakdown"
python -u research/pm_factors/ic_score.py --dates 20261104,20261105,20261106 --assign $R17 $SCORE --full-book $FBF $M --implied 15 --dump-pred \
  --out $R/stage6_r23_implied > $R/r23_implied.txt 2>&1
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261109,20261110 --assign $R17 $SCORE --full-book $FBD $M --implied 15 --dump-pred \
  --out $R/stage6_r23_implied_224 > $R/r23_implied_224.txt 2>&1
grep -hE "Error|Traceback|stocks x .* marks|IC: start" $R/r23_implied.txt $R/r23_implied_224.txt | cut -c1-200
for f in $R/r23_implied.txt $R/r23_implied_224.txt; do echo "  -- $f"; sed -n '/== implied values/,$p' $f | grep -vE "^time " | cut -c1-260 | head -80; done
python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r23_implied/pred.parquet 2>&1 | cut -c1-200
python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r23_implied_224/pred.parquet 2>&1 | cut -c1-200
say "done"
