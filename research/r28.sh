#!/usr/bin/env bash
# Round 28 on 97, foreground, after R27. Our engine starts every EMA from zero each morning; PM's live process starts from a
# seed .jz (production: the true D-1; 224: whatever 2.23's sod link pointed to at 09:00, often days old). Seed EVERY engine
# ema (run_cpp --seed-all) and score R17 against the same targets:
#   zero = as so far; A = true D-1, c0 (carried in full to the first tick); B = 224's stale seed (map B), c0 and c600
#   per day: engine run per seed into data/engine_seed/<cfg>, then ic_score R17 --engine-dir, with the half-hour IC table
#   bash research/r28.sh 2>&1 | tee data/reports/r28.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
R17=$R/stage6_r17/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }
cat_of() { [ "$(src $1)" = feitu ] && echo $DATA/catalog.duckdb || echo $DATA/catalog_dysl2.duckdb; }
true_prev() { case $1 in 20261104) echo 20261103 ;; 20261105) echo 20261104 ;; 20261106) echo 20261105 ;; 20261109) echo 20261106 ;;
  20261110) echo 20261109 ;; 20261111) echo 20261110 ;; 20260430) echo 20260429 ;; esac; }
stale() { case $1 in 20261104) echo 20260914 ;; 20261105) echo 20261030 ;; 20261106) echo 20261102 ;; 20261109|20261110) echo 20261104 ;;
  20261111) echo 20261110 ;; 20260430) echo 20260429 ;; esac; }
hh() { python -u research/pm_factors/residual_breakdown.py --pred $1 2>&1 | sed -n '/per-mark IC by half hour/,/by board/p' | head -16; }

ls $SEED/20260914.jz $SEED/20261030.jz $SEED/20261102.jz $SEED/20260429.jz >/dev/null || say "a seed file is missing in $SEED"
printf "%-10s %8s %8s %8s %8s\n" day zero A_c0 B_c0 B_c600
for d in 20261104 20261105 20261106 20261109 20261110 20261111 20260430; do
  P=$(true_prev $d); S=$(stale $d)
  line="$(printf '%-10s' $d)"
  for cfg in zero A_c0 B_c0 B_c600; do
    case $cfg in zero) ED=data/engine ;; *) ED=data/engine_seed/$cfg ;; esac
    if [ $cfg != zero ]; then
      case $cfg in A_c0) SF=$SEED/$P.jz; T=c0 ;; B_c0) SF=$SEED/$S.jz; T=c0 ;; B_c600) SF=$SEED/$S.jz; T=c600 ;; esac
      if [ $d = 20260430 ] && [ $cfg = B_c600 ]; then line="$line $(printf '%8s' -)"; continue; fi
      python -u research/engine/run_cpp.py --date $d --universe pm:$SEED/$P.jz --prev $SF --seed-all $T --catalog $(cat_of $d) \
        --marks rand:240 --threads 32 --out $ED --work data/engine_seed/work_$cfg > $R/r28_engine_${d}_$cfg.txt 2>&1 \
        || { line="$line $(printf '%8s' FAIL)"; continue; }
    fi
    CRUCIBLE_SOURCE=$(src $d) python -u research/pm_factors/ic_score.py --dates $d --assign $R17 $SCORE --full-book $(fb $d) $M \
      --engine-dir $ED --dump-pred --out $R/stage6_r28_${d}_$cfg > $R/r28_${d}_$cfg.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r28_${d}_$cfg.txt)")"
  done
  echo "$line"
done
say "half-hour IC tables: zero / A_c0 / B_c0"
for d in 20261104 20261105 20261106 20261109 20261110 20261111 20260430; do
  for cfg in zero A_c0 B_c0; do
    [ -e $R/stage6_r28_${d}_$cfg/pred.parquet ] && { echo "  -- $d $cfg"; hh $R/stage6_r28_${d}_$cfg/pred.parquet; }
  done
done
grep -h "carry seeds exported" $R/r28_engine_*.txt | cut -c1-160 | head -8
grep -lE "Error|Traceback" $R/r28_*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
