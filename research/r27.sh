#!/usr/bin/env bash
# Round 27 on 97, foreground, after R25 (uses its r25_var / r25v_* runs). Reads 2.23 only; writes under /work.
# 224's live predictions (the target of 1104-1111) carry a stale seed (2.23's sod links were made late), so:
#   0 PM's production predictions on 2.23 (livev1 .../alpha.parquet, true seed) for the November days -> a clean target
#   1 R17 against the production target vs against 224's, on the days that have both
#   2 pick R25's one-change variants that lose on no selection day (1110, 0430) -> one combined change set
#   3 R17 + ONE of R24's high-t new formulas each, scored on the selection days; pick the same way
#   4 R17 / picked R25 changes / picked additions / both, on every day (224 target) and every production day
#   bash research/r27.sh 2>&1 | tee data/reports/r27.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
T224=$DATA/pm_target_r22
TP=$DATA/pm_target_prod
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH23="ssh -p $P23 -o BatchMode=yes"
R17=$R/stage6_r17/ic_assign.json
R24=$R/stage6_r24/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
MM="--models /data/prod/models/tk5m --marks rand:240 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }
run() { # run <assign> <day> <target dir> <out tag>
  CRUCIBLE_SOURCE=$(src $2) python -u research/pm_factors/ic_score.py --dates $2 --assign $1 $SCORE --full-book $(fb $2) $MM --target $3 \
    --out $R/stage6_$4 > $R/$4.txt 2>&1
}

say "0 PM production predictions on 2.23 (livev1)"
$SSH23 $H23 'find /data/share/data/prod/h/HS300/livev1 -maxdepth 5 -name "alpha*.parquet" -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort -k4 | tail -40; echo ==; ls -la /data/share/data/prod/h/HS300/livev1 | head -30' \
  > $R/r27_livev1.txt 2>&1 || say "  livev1 listing failed: $(tail -1 $R/r27_livev1.txt)"
cut -c1-200 $R/r27_livev1.txt
mkdir -p $TP
[ -e $DATA/pm_alpha/date=20260430.parquet ] && ln -sf $DATA/pm_alpha/date=20260430.parquet $TP/
PROD=""
for p in $(grep -E 'alpha[^ ]*\.parquet$' $R/r27_livev1.txt | awk '{print $4}'); do
  d=$(echo $p | grep -oE '2026(10|11)[0-9]{2}' | tail -1)
  [ -n "$d" ] && [ "$d" -ge 20261101 ] || continue
  mkdir -p $DATA/pm_livev1/rec10_p/$d
  rsync -aL -e "$SSH23" $H23:$p $DATA/pm_livev1/rec10_p/$d/alpha.parquet >> $R/r27_rsync.txt 2>&1 || { say "  copy $p failed"; continue; }
  python -u research/pm_factors/alpha_target.py --alpha $DATA/pm_livev1/rec10_p/$d/alpha.parquet --out $DATA/pm_alpha > $R/r27_alpha_$d.txt 2>&1 \
    && ln -sf $DATA/pm_alpha/date=$d.parquet $TP/ && PROD="$PROD $d"
done
PROD=$(echo $PROD | tr ' ' '\n' | sort -u | tr '\n' ' ')
echo "  production target days (besides 0430): ${PROD:-none}"

say "1 R17: production target vs 224 target"
for d in $PROD; do
  run $R17 $d $TP r27_prod_r17_$d; run $R17 $d $T224 r27_224_r17_$d
  echo "  $d: production $(ic $R/r27_prod_r17_$d.txt), 224 $(ic $R/r27_224_r17_$d.txt)"
done

say "2 pick R25's variants (1110, 0430)"
python -u research/pm_factors/r27_pick.py --base $R/r25_var/base.json --var-dir $R/r25_var --prefix $R/r25v_ --days 20261110,20260430 \
  --out $R/r27_pick_r25.json 2>&1 | cut -c1-200

say "3 R17 + one of R24's new formulas, on the selection days"
SELD="20260430 20261110"
python -u research/pm_factors/r27_adds.py --r17 $R17 --r24 $R24 --out $R/r27_add
for v in $(ls $R/r27_add/*.json | xargs -n1 basename | sed 's/\.json$//'); do
  for d in $SELD; do run $R/r27_add/$v.json $d $T224 r27a_${v}_$d; done
done
python -u research/pm_factors/r27_pick.py --base $R/r27_add/base.json --var-dir $R/r27_add --prefix $R/r27a_ --days $(echo $SELD | tr ' ' ',') \
  --exclude base --out $R/r27_pick_add.json 2>&1 | cut -c1-200

say "4 R17 / R25 picks / additions / both, every day"
python - <<'EOF'
import json
r17 = json.load(open("data/reports/stage6_r17/ic_assign.json"))
a = json.load(open("data/reports/r27_pick_r25.json"))
b = json.load(open("data/reports/r27_pick_add.json"))
both = dict(a)
for k in set(r17) | set(b):
    if r17.get(k) != b.get(k):
        if b.get(k) is None:
            both.pop(k, None)
        else:
            both[k] = b[k]
json.dump(both, open("data/reports/r27_both.json", "w"), indent=1)
print(f"  r17 {len(r17)}, r25 picks {len(a)}, additions {len(b)}, both {len(both)} columns")
EOF
printf "%-12s %-5s %8s %8s %8s %8s\n" day tgt r17 r25pick addpick both
for dt in 20261104:224 20261105:224 20261106:224 20261109:224 20261110:224 20261111:224 20260430:prod $(for x in $PROD; do echo $x:prod; done); do
  d=${dt%:*}; tg=${dt#*:}; [ $tg = prod ] && TG=$TP || TG=$T224
  line="$(printf '%-12s %-5s' $d $tg)"
  for c in r17 r25pick addpick both; do
    case $c in r17) A=$R17 ;; r25pick) A=$R/r27_pick_r25.json ;; addpick) A=$R/r27_pick_add.json ;; both) A=$R/r27_both.json ;; esac
    run $A $d $TG r27f_${c}_${tg}_$d
    line="$line $(printf '%8s' "$(ic $R/r27f_${c}_${tg}_$d.txt)")"
  done
  echo "$line"
done
grep -lE "Error|Traceback" $R/r27*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
