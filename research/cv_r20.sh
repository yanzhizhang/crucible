#!/usr/bin/env bash
# Round 21 on 97, foreground. A: let the whole-book candidates compete for the columns already mapped to snapshot-book
# formulas. C: leave-one-day-out check of the R19 + R20 picks: scan on two days from the R17 assignment, score the third
# day with that fold's assignment, with R17's (baseline) and with R20's (in-sample reference).
#   bash research/cv_r20.sh 2>&1 | tee data/reports/r21.txt
set -u
cd /work/crucible
export CRUCIBLE_SOURCE=dysl2
R=data/reports
FB=/work/crucible_data/store/pm_factors/full_book_dysl2
DAYS="20261104 20261105 20261106"
COMMON="--models /data/prod/models/tk5m --marks rand:240 --target /work/crucible_data/pm_signals --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --implied 0 --full-book $FB"
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1 \2/'; }

say "A: book-mapped columns against the whole-book candidates"
BOOK=$(python -c "import json, re; a=json.load(open('$R/stage6_r20/ic_assign.json')); rx=re.compile(r'imb|depth|ofi|simb|book|microprice|level|spread|bv\d|av\d|wp\d'); f=[k for k,v in a.items() if rx.search(v)]; print(','.join(f))")
echo "  book-mapped columns: $(echo $BOOK | tr , '\n' | wc -l)"
python -u research/pm_factors/ic_score.py --dates 20261104,20261105,20261106 --assign $R/stage6_r20/ic_assign.json --calibrate --scan-feats "$BOOK" $COMMON --out $R/stage6_r21 > $R/r21_a.txt 2>&1
grep -E "Error|Traceback|^dropped|IC: start" $R/r21_a.txt | cut -c1-300
python -c "import json; a=json.load(open('$R/stage6_r21/ic_assign.json')); b=json.load(open('$R/stage6_r20/ic_assign.json')); print('  ', len(b), '->', len(a)); [print('  ', k, ':', b.get(k), '=>', v) for k, v in a.items() if b.get(k) != v]; [print('  ', k, ':', v, '=> dropped') for k, v in b.items() if k not in a]"
grep -E "full_book|resting_|price_levels|best_level|beyond_level10" $R/r21_a.txt | grep "t_gain" | cut -c1-240 | head -30

say "C: leave one day out (scan from the R17 assignment on two days, score the third)"
U=$(python -c "import json, polars as pl; a=json.load(open('$R/stage6_r17/ic_assign.json')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  columns unmapped at R17: $(echo $U | tr , '\n' | wc -l)"
for H in $DAYS; do
  T=$(echo $DAYS | tr ' ' '\n' | grep -v $H | paste -sd,)
  python -u research/pm_factors/ic_score.py --dates $T --assign $R/stage6_r17/ic_assign.json --calibrate --scan-feats "$U" $COMMON --out $R/stage6_cv_$H > $R/r21_cv_$H.txt 2>&1
  grep -E "Error|Traceback" $R/r21_cv_$H.txt | head -3
  for tag in fold r17 r20; do
    case $tag in fold) AS=$R/stage6_cv_$H/ic_assign.json ;; r17) AS=$R/stage6_r17/ic_assign.json ;; r20) AS=$R/stage6_r20/ic_assign.json ;; esac
    python -u research/pm_factors/ic_score.py --dates $H --assign $AS --no-drop-harmful --max-features 0 $COMMON --out $R/stage6_cv_${H}_$tag > $R/r21_cv_${H}_$tag.txt 2>&1
  done
  n=$(python -c "import json; print(len(json.load(open('$R/stage6_cv_$H/ic_assign.json'))))")
  say "  held out $H (scan on $T): train IC start/end $(ic $R/r21_cv_$H.txt), $n columns; on $H: R17 $(ic $R/r21_cv_${H}_r17.txt | cut -d' ' -f1), fold $(ic $R/r21_cv_${H}_fold.txt | cut -d' ' -f1), R20 in-sample $(ic $R/r21_cv_${H}_r20.txt | cut -d' ' -f1)"
done
python -c "
import json
R='$R'; days='$DAYS'.split()
a={d: json.load(open(f'{R}/stage6_cv_{d}/ic_assign.json')) for d in days}; r17=json.load(open(f'{R}/stage6_r17/ic_assign.json')); r20=json.load(open(f'{R}/stage6_r20/ic_assign.json'))
new=sorted({k for d in days for k in a[d] if r17.get(k)!=a[d][k]} | {k for k in r20 if r17.get(k)!=r20[k]})
same=[k for k in new if len({a[d].get(k) for d in days})==1 and a[days[0]].get(k)==r20.get(k)]
two=[k for k in new if k not in same and max(sum(a[d].get(k)==a[e].get(k) and a[d].get(k) is not None for e in days) for d in days)>=2]
print(f'  columns new since R17: {len(new)}; same formula in all three folds and R20: {len(same)}; same in two folds: {len(two)}; unstable: {len(new)-len(same)-len(two)}')
print('  stable:', ', '.join(f'{k}={r20[k]}' for k in same)[:3000])
"
say "done"
