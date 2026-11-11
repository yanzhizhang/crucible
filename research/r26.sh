#!/usr/bin/env bash
# Round 26 on 97, foreground, after R25. 224's PM loads no .jz at all; its only daily input from outside is /tmp/s/stsfile,
# and the day it loads is pinned / lagged (every 1111 run loaded 20261109, the first 1109 run 20261105). ic_score reads
# the stsfile day BEFORE the trading day. Score R17 three ways on every day:
#   before = the latest file before the day (as so far); on = the day's own file; map = the day 224's log shows it loaded
#   0 which stsfile day each 224 log loaded -> data/reports/sts_map_224.json; stsfile refreshed from 2.23
#   1 R17 on 1104-1106 (feitu), 1109-1111 (DataYes) and 0430 (PM production alpha; no 224 log, map = on) x before / on / map
#   bash research/r26.sh 2>&1 | tee data/reports/r26.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
R17=$R/stage6_r17/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }

say "0 stsfile days 224 loaded; stsfile from 2.23"
rsync -aL -e "ssh -p $P23 -o BatchMode=yes" $H23:/data/share/s/ $DATA/pm_stsfile/ > $R/r26_rsync.txt 2>&1 || say "  stsfile rsync failed: $(tail -1 $R/r26_rsync.txt)"
ls $DATA/pm_stsfile/stats/sod | tail -8 | tr '\n' ' '; echo
python -u research/pm_factors/sts_days.py --logs $DATA/chrono_logs --out $R/sts_map_224.json 2>&1 | cut -c1-240

say "1 R17 x before / on / map"
printf "%-10s %8s %8s %8s\n" day before on map
for day in 20261104 20261105 20261106 20261109 20261110 20261111 20260430; do
  line="$(printf '%-10s' $day)"
  for mode in before on map; do
    case $mode in before) F="--pm-sts-day before" ;; on) F="--pm-sts-day on" ;; map) F="--pm-sts-day on --pm-sts-map $R/sts_map_224.json" ;; esac
    CRUCIBLE_SOURCE=$(src $day) python -u research/pm_factors/ic_score.py --dates $day --assign $R17 $SCORE $F --full-book $(fb $day) $M \
      --out $R/stage6_r26_${day}_$mode > $R/r26_${day}_$mode.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r26_${day}_$mode.txt)")"
  done
  echo "$line"
done
grep -h "stsfile stats/sod" $R/r26_*_*.txt | sort | uniq -c | head -30
grep -lE "Error|Traceback" $R/r26_*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
