#!/usr/bin/env bash
# Round 25 on 97, foreground. 224's PM process starts every day from ONE old state file soft-linked in place of yesterday's
# (user, 2026-11-12): find it, and test whether its staleness explains the per-stock residual against 224's live prediction.
# Plus the read-only state probes, 1111 follow-ups and one-change variants of R17 on held-out days. Reads 224 / 2.23 only.
#   0 224: where PM's state files are, which are links, to what; copy them to /work; which 2.23 day the seed is (md5)
#   1 jz_probe: zzro keys, 0429 vs 1106 key sets, weight carry, 2.23 sod/p1_intrading_dcp against dcp
#   2 224 logs of 1109 / 1111: PM start-up lines (seed, restart), line formats; pm_signals summaries 1104-1111; pm_poslog 1111
#   3 gen_compare columns flagged as moved; state_keys groups by base family (R17, R24)
#   4 R17 on 1110 / 1111 with --dump-pred: per half hour, board, persistence; then fake_seed on 1109-1111
#   5 1111 again: hstats / PM 1111.jz from 2.23 if landed (rescore R17 with --rebuild, day-end state check)
#   6 one-change variants of R17 scored on 1110, 1111 (224, stale seed) and 0430 (PM production alpha, true seed)
#   bash research/r25.sh 2>&1 | tee data/reports/r25.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
DCP=/data/prod/h/HS300/p1_intrading_dcp
SEED=$DATA/seed_probe/HS300
S224=$DATA/seed_224
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
H224=${H224:-zhangyanzhi@10.12.1.224}
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH23="ssh -p $P23 -o BatchMode=yes"
R17=$R/stage6_r17/ic_assign.json
R24=$R/stage6_r24/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }
mkdir -p $S224

say "0 224: PM state files and links"
ssh -o BatchMode=yes $H224 'find /tmp /home/zhangyanzhi -maxdepth 7 \( -name "*.jz" -o -name "*intrading_dcp*" \) -printf "%y %TY-%Tm-%Td %TH:%TM %s %p -> %l\n" 2>/dev/null | sort -k5 | head -80' \
  > $R/r25_224_jz.txt 2>&1 || say "  224 find failed: $(tail -1 $R/r25_224_jz.txt)"
cat $R/r25_224_jz.txt | cut -c1-220
for p in $(grep -E '\.jz' $R/r25_224_jz.txt | awk '{print $5}' | sort -u | head -40); do
  rsync -aL -e "ssh -o BatchMode=yes" "$H224:$p" "$S224/$(echo $p | tr / _)" >> $R/r25_224_rsync.txt 2>&1
done
echo "  copied: $(ls $S224 | wc -l) files"
md5sum $SEED/*.jz 2>/dev/null | awk '{n=split($2,a,"/"); print $1, a[n]}' > $R/r25_md5_23.txt
for f in $S224/*; do
  [ -f "$f" ] || continue
  m=$(md5sum "$f" | cut -c1-32); hit=$(grep "^$m " $R/r25_md5_23.txt | awk '{print $2}' | tr '\n' ' ')
  echo "  $(basename $f): md5 ${m:0:8} = 2.23 ${hit:-none}"
done
FAKE=$(ls $S224/* 2>/dev/null | sort | tail -1)  # the latest-dated name = the file 224 loaded last (a link to the old seed)
FAKE=${FAKE_OVERRIDE:-$FAKE}
echo "  seed used below: ${FAKE:-none} (FAKE_OVERRIDE=<path> to pick another)"

say "1 jz_probe: zzro, 0429 vs 1106, weight carry, sod"
mkdir -p $DATA/sod_dcp
$SSH23 $H23 'ls -la --time-style=full-iso /data/share/data/prod/h/HS300/sod/p1_intrading_dcp | tail -12; ls -la /data/share/data/prod/h/HS300/sod | head -20' \
  > $R/r25_sod_list.txt 2>&1 || say "  sod listing failed: $(tail -1 $R/r25_sod_list.txt)"
cat $R/r25_sod_list.txt | cut -c1-200
for f in $($SSH23 $H23 'ls /data/share/data/prod/h/HS300/sod/p1_intrading_dcp | tail -4' 2>/dev/null); do
  rsync -aL -e "$SSH23" $H23:/data/share/data/prod/h/HS300/sod/p1_intrading_dcp/$f $DATA/sod_dcp/ >> $R/r25_sod_rsync.txt 2>&1
done
python -u research/pm_factors/jz_probe.py --dcp $SEED --stem zzro --a 20260429 --b 20261106 --days 6 --sod $DATA/sod_dcp > $R/r25_jz_probe.txt 2>&1 \
  || say "  jz_probe FAILED: $(tail -3 $R/r25_jz_probe.txt)"
cut -c1-260 $R/r25_jz_probe.txt

say "2 224 logs: PM start-up, restarts, line formats; pm_signals summaries; pm_poslog 1111"
for f in $DATA/chrono_logs/2026-11-09_* $DATA/chrono_logs/2026-11-11_*; do
  [ -f "$f" ] || continue
  echo "  -- $(basename $f): $(wc -l < $f) lines, chrono-format $(grep -c '^\[20' $f), other $(grep -vc '^\[20' $f)"
  grep -n -iE '\.jz|dcp|stsfile|samplerS|hstats|restore|seed|load(ed|ing)? ' $f | head -25 | cut -c1-220
done
for d in 20261104 20261105 20261106 20261109 20261110 20261111; do
  echo "  -- $d signals:"; grep -E "signals,|factor lines|order match|factor program" $R/daily/${d}_signals.txt 2>/dev/null | cut -c1-220
done
python -u research/pm_factors/pm_poslog.py --dates 20261111 > $R/daily/20261111_poslog.txt 2>&1 || say "  pm_poslog FAILED: $(tail -2 $R/daily/20261111_poslog.txt)"
grep -E "WARNING|^==|decisions with|pos ==|executp ==|prediction vs" $R/daily/20261111_poslog.txt | cut -c1-230

say "3 gen_compare flags; state_keys by base family"
python - <<'EOF'
import polars as pl
d = pl.read_csv("data/reports/gen_compare.csv")
m = d.filter(pl.col("moved"))
print(f"  moved rows {m.height} of {d.height}; per column: months moved, max median-split shift, span ratio range")
s = m.group_by("col", "ref_name").agg(pl.col("month").n_unique().alias("months"), pl.col("t50_shift").max(), pl.col("span_ratio").min().alias("span_min"),
                                      pl.col("span_ratio").max().alias("span_max"), pl.col("month").min().alias("first"), pl.col("month").max().alias("last"))
with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=2):
    print(s.sort("months", descending=True))
EOF
HVS=$HV; [ -d $HVS ] || HVS=/data/prod/hstats/HS300
for tag in r17 r24; do
  A=$R/stage6_$tag/ic_assign.json
  python -u research/pm_factors/state_keys.py --state $DCP/20261106.jz --assign $A --hstats $HVS --out $R/r25_state_keys_$tag.csv > $R/r25_state_keys_$tag.txt 2>&1
  echo "  $tag:"; sed -n '/^open columns by where/,$p' $R/r25_state_keys_$tag.txt | cut -c1-240
done

say "4 R17 on 1110 / 1111 with --dump-pred; residual breakdown; fake seed"
for day in 20261110 20261111; do
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $day --assign $R17 $SCORE --full-book $FBD $M --dump-pred \
    --out $R/stage6_r25_ic_$day > $R/r25_ic_$day.txt 2>&1
  echo "  $day R17 $(ic $R/r25_ic_$day.txt)"
  python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r25_ic_$day/pred.parquet 2>&1 | cut -c1-300
done
if [ -n "${FAKE:-}" ] && [ -f "$FAKE" ]; then
  python -u research/pm_factors/fake_seed.py --seed $FAKE --dcp $SEED --pred $R/stage6_r23_implied_224/pred.parquet $R/stage6_r25_ic_20261110/pred.parquet \
    $R/stage6_r25_ic_20261111/pred.parquet > $R/r25_fake_seed.txt 2>&1 || say "  fake_seed FAILED: $(tail -3 $R/r25_fake_seed.txt)"
  cut -c1-220 $R/r25_fake_seed.txt
else
  say "  no 224 seed file found: fake_seed skipped"
fi

say "5 1111 follow-up from 2.23"
rsync -aLm -e "$SSH23" --include="*/" --include="20261111/***" --exclude="*" $H23:/data/share/data/prod/hstats/HS300/ $DATA/hstats_extra/HS300/ > $R/r25_rsync_1111.txt 2>&1
for fam in $(ls $DATA/hstats_extra/HS300); do mkdir -p $HV/$fam; for x in $DATA/hstats_extra/HS300/$fam/*; do [ -e $HV/$fam/$(basename $x) ] || ln -sfn $x $HV/$fam/; done; done
NH=$(ls -d $HV/*/20261111/samplerS.nc 2>/dev/null | wc -l); echo "  1111 hstats families: $NH"
rsync -aL -e "$SSH23" --include="*.jz" --exclude="*" $H23:/data/share/data/prod/h/HS300/p1_intrading_dcp/ $SEED/ >> $R/r25_rsync_1111.txt 2>&1
if [ "$NH" -gt 0 ]; then
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261111 --assign $R17 $SCORE --full-book $FBD $M --rebuild \
    --out $R/stage6_r25_ic_20261111_hs > $R/r25_ic_20261111_hs.txt 2>&1
  echo "  1111 R17 with hstats $(ic $R/r25_ic_20261111_hs.txt)"
fi
if [ -e $SEED/20261111.jz ]; then
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/dcp_build.py --dates 20261111 --dcp-dir $SEED > $R/r25_dcp_1111.txt 2>&1 || say "  dcp_build FAILED"
  tail -40 $R/r25_dcp_1111.txt | cut -c1-200
else
  say "  still no PM 20261111.jz on 2.23"
fi

say "6 one-change variants of R17 on 1110, 1111, 0430"
python -u research/pm_factors/r25_variants.py --r17 $R17 --r24 $R24 --out $R/r25_var
printf "%-22s %8s %8s %8s\n" variant 1110 1111 0430
for v in $(ls $R/r25_var/*.json | xargs -n1 basename | sed 's/\.json$//'); do
  line="$(printf '%-22s' $v)"
  for day in 20261110 20261111 20260430; do
    CRUCIBLE_SOURCE=$(src $day) python -u research/pm_factors/ic_score.py --dates $day --assign $R/r25_var/$v.json $SCORE --full-book $(fb $day) $M \
      --out $R/stage6_r25v_${v}_$day > $R/r25v_${v}_$day.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r25v_${v}_$day.txt)")"
  done
  echo "$line"
done
grep -lE "Error|Traceback" $R/r25v_*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
