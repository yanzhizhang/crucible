#!/usr/bin/env bash
# Round 42 on 97, foreground: is the 0430 tuning real or fitted to that day? Every set scored on 1113 (224 target, dysl2) with
# the SAME options the 0430 runs use -- the earlier 1113 scores of r37b / r40g / r41c ran without --neutral and --seed-dir, so
# their neu: and gen.sd_* columns were empty there. Then r41c column by column on the two days. Writes under /work only.
#   bash research/r42.sh 2>&1 | tee data/reports/r42.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
D=${D:-20261113}
SEED=${SEED:-/data/prod/h/HS300/p1_intrading_dcp}
C1=$R/stage6_r41c_gate/ic_assign_gated.json
M="--models /data/prod/models/tk5m --marks rand:240 --target $DATA/pm_target_r22 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --seed-dir $SEED"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
say() { echo "[$(date +%T)] $*"; }
say "1 every set on $D with --neutral --seed-dir"
for tag in r27both r34g r37b r40g r41c; do
  case $tag in r27both) A=$R/r27_both.json ;; r34g) A=$R/r34g_baseline.json ;; r37b) A=$R/stage6_r37_gate/ic_assign_gated.json ;; r40g) A=$R/r40g_baseline.json ;; r41c) A=$C1 ;; esac
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $D --assign $A $SCORE --dump-pred --full-book $DATA/store/pm_factors/full_book_dysl2 $M \
    --out $R/stage6_r42_$tag > $R/r42_$tag.txt 2>&1
  echo "  $tag: $(grep -E 'IC: start' $R/r42_$tag.txt | cut -c1-110) $(grep -cE '^drop ' $R/r42_$tag.txt) dropped at load"
  grep -E "Error|Traceback|seed state" $R/r42_$tag.txt | head -3
done
say "2 r41c column by column: 0430 (2.23 target) against $D (224 target)"
python -u research/pm_factors/col_days.py --a $R/stage6_r41_audit2 --b $R/stage6_r42_r41c --names 0430,${D:4} 2>&1 | cut -c1-230
say "3 tails on $D"
python -u research/pm_factors/audit.py --run $R/stage6_r42_r41c --out $R/r42_audit.json 2>&1 | sed -n '1,12p;/== tails/,$p' | cut -c1-200
grep -lE "Error|Traceback" $R/r42*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
