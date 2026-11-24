#!/usr/bin/env bash
# The data pull of daily_pm.sh (its steps 1-2) on its own, for cron on 97: 224's chrono logs and each day's position log
# and base positions, read-only rsync from 224 into /work/crucible_data. No python, no conda (cron has a bare env).
# Logs of day D land in /work/crucible_data/pm_224/<D>/chrono_logs (what the runs_* tools and pm_signals --logs read).
# Days: the arguments, else today and yesterday (rsync is incremental, so a missed night is picked up the next day).
#   bash research/daily_pull.sh                 # today + yesterday
#   bash research/daily_pull.sh 20261114        # one day
# Cron (97, root, trading days Mon-Fri, 16:00 after the close): 0 16 * * 1-5 bash /work/crucible/research/daily_pull.sh >> /work/crucible/data/reports/daily/cron_pull.log 2>&1
set -u
H=${H:-zhangyanzhi@10.12.1.224}
DATA=/work/crucible_data
O=/work/crucible/data/reports/daily; mkdir -p $O
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
DAYS="$*"; [ -n "$DAYS" ] || DAYS="$(date -d yesterday +%Y%m%d) $(date +%Y%m%d)"
say() { echo "[$(date '+%F %T')] $*"; }

say "pull start: $DAYS"
# 224 layout since the daily-directory change: everything of a day under /tmp/o/executor/<D>/; chrono's spdlog in chrono_logs/
# (<name>_YYYYMMDD_HHMMSS.log), the 30 s position log in portfolio/ledger/ (older days: position_log/). The old tee'd
# /tmp/zyq/sht-logs is still pulled when it exists, for days before the change.
rsync -a -e "$SSH" $H:/tmp/zyq/sht-logs/ $DATA/chrono_logs/ > $O/pull_logs.txt 2>&1 || say "  old /tmp/zyq/sht-logs not pulled: $(tail -1 $O/pull_logs.txt)"
for D in $DAYS; do
  E=$H:/tmp/o/executor/$D
  mkdir -p $DATA/pm_224/$D/chrono_logs $DATA/chrono_poslog/$D/position_log
  rsync -a -e "$SSH" $E/chrono_logs/ $DATA/pm_224/$D/chrono_logs/ > $O/${D}_pull.txt 2>&1 || say "  $D: no chrono_logs/"
  rsync -a -e "$SSH" $E/portfolio/ledger/ $DATA/chrono_poslog/$D/position_log/ >> $O/${D}_pull.txt 2>&1     || rsync -a -e "$SSH" $E/position_log/ $DATA/chrono_poslog/$D/position_log/ >> $O/${D}_pull.txt 2>&1 || say "  $D: no position log"
  rsync -a -e "$SSH" "$E/base_position_*.csv" $DATA/chrono_poslog/$D/ >> $O/${D}_pull.txt 2>&1 || say "  $D: no base positions"
  say "  $D: chrono logs $(ls $DATA/pm_224/$D/chrono_logs/*_${D}_*.log 2>/dev/null | wc -l) (+ old tee'd $(ls $DATA/chrono_logs | grep -c "^${D:0:4}-${D:4:2}-${D:6:2}_")), position-log csvs $(ls $DATA/chrono_poslog/$D/position_log/*.csv 2>/dev/null | wc -l), base position csvs $(ls $DATA/chrono_poslog/$D/base_position_*.csv 2>/dev/null | wc -l)"
done
say "pull done"
