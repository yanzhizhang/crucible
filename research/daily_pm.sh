#!/usr/bin/env bash
# One trading day of the PM reproduction on 97, in order: 224's logs and position log (read-only rsync), PM's signals and
# states (pm_signals), the DataYesSharesL2 package into dysl2_raw + catalog_dysl2 (the source 224 runs on), then the
# decision and position checks. Foreground; each step logs to data/reports/daily/<D>_<step>.txt and prints one line.
#   bash research/daily_pm.sh 20261110          # a day whose package is not on 97 yet: decode is skipped, rerun later
set -u
D=${1:?trading day YYYYMMDD}
cd /work/crucible
H=${H:-zhangyanzhi@10.12.1.224}
DATA=/work/crucible_data
DY=${DY:-/mnt/TickDataRepo/DataYesSharesL2}
W=${W:-96}
O=data/reports/daily; mkdir -p $O
which python >/dev/null || { echo "no python: activate the crucible conda env"; exit 1; }
say() { echo "[$(date +%T)] $*"; }

say "1 logs: 224 /tmp/zyq/sht-logs -> $DATA/chrono_logs"
rsync -a $H:/tmp/zyq/sht-logs/ $DATA/chrono_logs/ > $O/${D}_rsync.txt 2>&1 || say "  rsync logs FAILED: $(tail -1 $O/${D}_rsync.txt)"
ls $DATA/chrono_logs | grep -c "^${D:0:4}-${D:4:2}-${D:6:2}_" | xargs echo "  log files for $D:"

say "2 position log + base positions: 224 /tmp/o/executor/$D"
mkdir -p $DATA/chrono_poslog/$D
rsync -a $H:/tmp/o/executor/$D/position_log/ $DATA/chrono_poslog/$D/position_log/ >> $O/${D}_rsync.txt 2>&1 || say "  no position log for $D"
rsync -a "$H:/tmp/o/executor/$D/base_position_*.csv" $DATA/chrono_poslog/$D/ >> $O/${D}_rsync.txt 2>&1 || say "  no base positions for $D"
find $DATA/chrono_poslog/$D -type f | wc -l | xargs echo "  files:"

say "3 pm_signals"
python -u research/pm_factors/pm_signals.py --logs $DATA/chrono_logs --dates $D --out $DATA/pm_signals > $O/${D}_signals.txt 2>&1 \
  || say "  pm_signals FAILED: $(tail -1 $O/${D}_signals.txt)"
grep -E "signals,|factor lines|order match|factor program|->" $O/${D}_signals.txt | sed 's/^/  /'

say "4 DataYesSharesL2 $D -> dysl2_raw, catalog_dysl2"
if [ -e $DY/$D.zip ] || [ -d $DY/$D ]; then  # a zip, or (recent days) the unzipped folder
  python -u research/decode_datayes_day.py --root $DY --date $D --out $DATA/dysl2_raw --workers $W --kinds quotation,order,transaction,index \
    > $O/${D}_decode.txt 2>&1 || say "  decode FAILED: $(tail -2 $O/${D}_decode.txt)"
  CRUCIBLE_SOURCE=dysl2 python -u research/catalog.py > $O/${D}_catalog.txt 2>&1 || say "  catalog FAILED: $(tail -1 $O/${D}_catalog.txt)"
  CRUCIBLE_SOURCE=dysl2 python -c "import sys; sys.path.insert(0,'research/pm_factors'); import duckdb, dcp_build as B; c=duckdb.connect(str(B.CATALOG), read_only=True); print('  quotes', c.sql('SELECT date, count(*), count(DISTINCT symbol_id) FROM raw_quotation WHERE date = $D GROUP BY 1').fetchall())"
else
  say "  no $DY/$D(.zip) yet: rerun this script later for steps 4-5"
fi

say "5 decisions + position log"
python -u research/pm_factors/pm_decisions.py --dates $D > $O/${D}_decisions.txt 2>&1 || say "  pm_decisions FAILED"
python -u research/pm_factors/pm_poslog.py --dates $D > $O/${D}_poslog.txt 2>&1 || say "  pm_poslog FAILED"
grep -E "^==|per stock" $O/${D}_decisions.txt | cut -c1-200 | sed 's/^/  /'
grep -E "^==|decisions with|pos ==|executp ==|prediction vs" $O/${D}_poslog.txt | cut -c1-230 | sed 's/^/  /'
say "done $D"
