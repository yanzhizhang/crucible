#!/usr/bin/env bash
# Round 35 on 97, foreground: the diagnostics redone on 0430, the one day with a correct target (PM's production alpha; every
# 224 signal is unreliable). Writes under /work only.
#   1 r27_both on 0430: implied values of the 15 open columns with the most gain + the residual breakdown (time, board, stock)
#   2 the per-stock residual against every daily candidate (CED returns / vol / range / caps / tick / band / Barra industry
#     and style exposures): a missing static input shows as a strong rank correlation
#   bash research/r35.sh 2>&1 | tee data/reports/r35.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
M="--models /data/prod/models/tk5m --marks rand:240 --target $DATA/pm_target_r22 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $DATA/hstats_view/HS300"
say() { echo "[$(date +%T)] $*"; }

say "1 implied values + residual breakdown on 0430 (r27_both)"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/r27_both.json --no-drop-harmful --max-features 0 \
  --implied 15 --dump-pred --full-book $DATA/store/pm_factors/full_book $M --out $R/stage6_r35_20260430 > $R/r35_implied.txt 2>&1
grep -E "Error|Traceback|IC: start" $R/r35_implied.txt | cut -c1-200
sed -n '/== implied values/,$p' $R/r35_implied.txt | grep -vE "^time |Column_[0-9]+ span" | cut -c1-300 | head -120
python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r35_20260430/pred.parquet 2>&1 | cut -c1-300

say "2 per-stock residual against daily candidates"
python -u research/pm_factors/static_resid.py --pred $R/stage6_r35_20260430/pred.parquet 2>&1 | cut -c1-220
say "done"
