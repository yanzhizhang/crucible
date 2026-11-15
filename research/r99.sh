#!/usr/bin/env bash
# Round 99 on 97, foreground. Writes data/reports/stage6_r99_<mmdd>, data/featgen_r99, and (only when missing)
# full_book_dysl2 / data/engine marks for a day. Perhaps 1-2 hours. Read-only on PM data and the 224 logs.
# The scoring matrices R70 built for 1112 only, now for every 224 day on DataYes: ic_score's matrix + PM's live
# prediction (pm_target_r22 = pm_signals links) + our bases at the same marks. Same arguments as r70.sh's dysl2 branch,
# so later scans (zzeg, invert_cols) can be judged on 10 days instead of 2.
#   bash research/r99.sh 2>&1 | tee data/reports/r99.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
FBD=$DATA/store/pm_factors/full_book_dysl2
TGT=$DATA/pm_target_r22
C1=$R/stage6_r41c_gate/ic_assign_gated.json
G=$DATA/models_gen/B
DAYS=${DAYS:-20261102 20261103 20261104 20261105 20261106 20261109 20261110 20261111 20261112 20261113}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $TGT --target-stale 4 --max-cdf 1 --max-score 99 --models $G --dump-x --dump-pred"
for D in $DAYS; do
  say "$D"
  [ -e $DATA/pm_signals/date=$D.parquet ] || { echo "  no pm_signals for $D: skipped"; continue; }
  [ -e $TGT/date=$D.parquet ] || ln -sf $DATA/pm_signals/date=$D.parquet $TGT/
  P=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
  [ -n "$P" ] || { echo "  no PM state before $D: skipped"; continue; }
  if [ ! -e $FBD/date=$D.parquet ]; then
    CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/full_book.py --dates $D --marks rand:240 --workers 1 > $R/r99_${D}_fullbook.txt 2>&1 \
      || { echo "  full_book FAILED: $(tail -2 $R/r99_${D}_fullbook.txt)"; continue; }
  fi
  if [ ! -e data/engine/marks_$D.parquet ]; then
    python -u research/engine/run_cpp.py --date $D --universe pm:$SEED/$P.jz --prev $SEED/$P.jz --catalog $DATA/catalog_dysl2.duckdb \
      --marks rand:240 --threads 32 > $R/r99_${D}_engine.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $R/r99_${D}_engine.txt)"; continue; }
  fi
  ST=$SEED/$D.jz; [ -e $ST ] || ST=$SEED/$P.jz
  O=$R/stage6_r99_${D:4}
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks rand:240 --engine-dir data/engine \
    --full-book $FBD --seed-dir $SEED --out $O > $R/r99_${D}_ic.txt 2>&1
  echo "  similarity $(ic $R/r99_${D}_ic.txt)  $(grep -E 'Error|Traceback' $R/r99_${D}_ic.txt | head -1 | cut -c1-120)"
  CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date $D --state $ST --seed-dir $SEED --marks rand:240 --out data/featgen_r99 2>&1 | tail -1 | cut -c1-200
  ls $O/x.parquet $O/pred.parquet $O/cal_$D.json data/featgen_r99/base_$D.parquet 2>&1 | sed 's/^/  /'
done
say "done"
