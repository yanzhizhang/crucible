#!/usr/bin/env bash
# The data pull of daily_pm.sh (its steps 1-2) on its own, for cron on 97: 224's chrono logs and each day's position log
# and base positions, read-only rsync from 224 into /work/crucible_data. No python, no conda (cron has a bare env).
# Days: the arguments, else today and yesterday (rsync is incremental, so a missed night is picked up the next day).
#   bash research/daily_pull.sh                 # today + yesterday
#   bash research/daily_pull.sh 20261114        # one day
# Cron (97, root, trading days Mon-Fri): 30 21 * * 1-5 bash /work/crucible/research/daily_pull.sh >> /work/crucible/data/reports/daily/cron_pull.log 2>&1
set -u
H=${H:-zhangyanzhi@10.12.1.224}
DATA=/work/crucible_data
O=/work/crucible/data/reports/daily; mkdir -p $O
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
DAYS="$*"; [ -n "$DAYS" ] || DAYS="$(date -d yesterday +%Y%m%d) $(date +%Y%m%d)"
say() { echo "[$(date '+%F %T')] $*"; }

say "pull start: $DAYS"
rsync -a -e "$SSH" $H:/tmp/zyq/sht-logs/ $DATA/chrono_logs/ > $O/pull_logs.txt 2>&1 || say "  rsync logs FAILED: $(tail -1 $O/pull_logs.txt)"
for D in $DAYS; do
  mkdir -p $DATA/chrono_poslog/$D
  rsync -a -e "$SSH" $H:/tmp/o/executor/$D/position_log/ $DATA/chrono_poslog/$D/position_log/ > $O/${D}_pull.txt 2>&1 || say "  $D: no position log"
  rsync -a -e "$SSH" "$H:/tmp/o/executor/$D/base_position_*.csv" $DATA/chrono_poslog/$D/ >> $O/${D}_pull.txt 2>&1 || say "  $D: no base positions"
  say "  $D: log files $(ls $DATA/chrono_logs | grep -c "^${D:0:4}-${D:4:2}-${D:6:2}_"), position-log csvs $(ls $DATA/chrono_poslog/$D/position_log/*.csv 2>/dev/null | wc -l), base position csvs $(ls $DATA/chrono_poslog/$D/base_position_*.csv 2>/dev/null | wc -l)"
done
say "pull done"
