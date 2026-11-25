#!/usr/bin/env bash
# Cron scan of 224 and 2.23 on 97: read-only rsync pulls into /work/crucible_data, one summary line per source and day.
# Bare env (no conda, no python); ssh BatchMode; flock so two runs never overlap; no catalog rebuild, no engine run.
# Log: /work/crucible_data/cron/pull_<YYYYMMDD>.log. Never pulls *.json5 or holding / pnl / other portfolio day files.
#   bash research/cron_pull.sh 224                 # 224: chrono logs, position log, base positions, QD alpha.parquet (today + previous weekday)
#   bash research/cron_pull.sh 23                  # 2.23: hstats, PM .jz, stsfile, models
#   bash research/cron_pull.sh all 20261120        # both, one given day
# crontab (97, the user whose ssh keys reach 224 and 2.23):
#   40 16 * * 1-5 bash /work/crucible/research/cron_pull.sh 224
#   30 21 * * 1-5 bash /work/crucible/research/cron_pull.sh all
set -u
export PATH=/usr/local/bin:/usr/bin:/bin
WHAT=${1:-all}; shift || true
H224=${H224:-zhangyanzhi@10.12.1.224}
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
DATA=/work/crucible_data
C=$DATA/cron; mkdir -p $C
exec >> $C/pull_$(date +%Y%m%d).log 2>&1
exec 9> $C/.pull.lock
say() { echo "[$(date '+%F %T')] $*" | sed -e "s/${H224%@*}/<u>/g" -e "s/${H23%@*}@/<u>@/g"; }
flock -n 9 || { say "$WHAT: another pull still running, skipped"; exit 0; }

prev_wd() { local d; d=$(date -d "$1 -1 day" +%Y%m%d); while [ "$(date -d $d +%u)" -gt 5 ]; do d=$(date -d "$d -1 day" +%Y%m%d); done; echo $d; }
DAYS="$*"; [ -n "$DAYS" ] || DAYS="$(prev_wd $(date +%Y%m%d)) $(date +%Y%m%d)"
S224="ssh -o BatchMode=yes -o ConnectTimeout=20"
S23="ssh -p $P23 -o BatchMode=yes -o ConnectTimeout=20"
RS="rsync -a --timeout=600 --itemize-changes --exclude=*.json5"

# pull <label> <ssh> <src> <dst> [rsync filters...]: one line = files and MB at the destination, files new or changed this run
pull() {
  local label=$1 sh=$2 src=$3 dst=$4; shift 4
  mkdir -p "$dst"
  local out; out=$($RS -e "$sh" "$@" "$src" "$dst" 2>&1); local rc=$?
  local new; new=$(printf '%s\n' "$out" | grep -c '^>f')
  local nf; nf=$(find "$dst" -type f 2>/dev/null | wc -l)
  local mb; mb=$(du -sm "$dst" 2>/dev/null | cut -f1)
  if [ $rc -eq 0 ]; then say "  $label: files $nf, MB $mb, new $new"
  elif printf '%s\n' "$out" | grep -q 'No such file'; then say "  $label: not there (yet)"
  else say "  $label: rsync rc $rc: $(printf '%s\n' "$out" | grep -v '^[>.c]' | tail -1)"; fi
}

say "== $WHAT start, days $DAYS"
if [ "$WHAT" = 224 ] || [ "$WHAT" = all ]; then
  for D in $DAYS; do
    E=$H224:/tmp/o/executor/$D
    pull "224 $D chrono_logs" "$S224" $E/chrono_logs/ $DATA/pm_224/$D/chrono_logs/ --include='*.log' --exclude='*'
    pull "224 $D position_log" "$S224" $E/portfolio/ledger/ $DATA/chrono_poslog/$D/position_log/ --include='position_log*.csv' --exclude='*'
    pull "224 $D base_position" "$S224" $E/ $DATA/chrono_poslog/$D/ --include='base_position_*.csv' --exclude='*'
    pull "224 $D qd alpha" "$S224" $H224:/tmp/o/sig/$D/ $DATA/qd_alpha/$D/ --include='alpha.parquet' --exclude='*'
    [ "$D" -le 20261118 ] && pull "224 $D old sht-logs" "$S224" $H224:/tmp/zyq/sht-logs/ $DATA/chrono_logs/ --include="${D:0:4}-${D:4:2}-${D:6:2}_*" --exclude='*'
  done
fi
if [ "$WHAT" = 23 ] || [ "$WHAT" = all ]; then
  P=$H23:/data/share/data/prod
  for D in $DAYS; do
    pull "2.23 $D hstats" "$S23" $P/hstats/HS300/ $DATA/hstats_extra/HS300/ -m --include='*/' --include="$D/***" --exclude='*'
    for fam in $(ls $DATA/hstats_extra/HS300 2>/dev/null); do  # the view add_day / ic_score read: one link per family-day
      [ -e $DATA/hstats_extra/HS300/$fam/$D ] || continue
      mkdir -p $DATA/hstats_view/HS300/$fam; [ -e $DATA/hstats_view/HS300/$fam/$D ] || ln -sfn $DATA/hstats_extra/HS300/$fam/$D $DATA/hstats_view/HS300/$fam/
    done
    pull "2.23 $D PM .jz" "$S23" $P/h/HS300/p1_intrading_dcp/ $DATA/seed_probe/HS300/ --include="$D.jz" --exclude='*'
  done
  pull "2.23 stsfile" "$S23" $H23:/data/share/s/ $DATA/pm_stsfile/
  pull "2.23 models" "$S23" $P/models/ $DATA/prod_models/ --exclude='models.zip'
  for D in $DAYS; do [ -e $DATA/seed_probe/HS300/$D.jz ] || say "  2.23 $D: no .jz yet (lands ~19:30-21:00)"; done
fi
say "== $WHAT done"
