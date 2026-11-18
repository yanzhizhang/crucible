#!/usr/bin/env bash
# Round 171 on 97, foreground: make 20261118 a truth day from the 224 pull already on 97 (no rsync, nothing from 2.23 or 224).
# Heavy (the decode uses 96 workers): run it alone. Writes dysl2_raw / catalog_dysl2 (the normal daily targets), pm_signals,
# pm_orders, data/reports/daily. Steps are daily_pm.sh's 3-6 with the log directory pointed at the pull.
#   1 pm_signals + pm_orders from /work/crucible_data/pm_224/<D>/chrono_logs (PM's predictions, states, decisions, orders)
#   2 DataYesSharesL2 <D> -> dysl2_raw, catalog_dysl2
#   3 decisions + order counts
#   bash research/r171.sh 2>&1 | tee data/reports/r171.txt
set -u
D=${1:-20261118}
cd /work/crucible
DATA=/work/crucible_data
LOGS=$DATA/pm_224/$D/chrono_logs
DY=${DY:-/mnt/TickDataRepo/DataYesSharesL2}
W=${W:-96}
O=data/reports/daily; mkdir -p $O
say() { echo "[$(date +%T)] $*"; }
ls -la $LOGS | cut -c1-160
[ -e $DATA/pm_signals/state/date=$D.parquet ] && say "pm_signals state for $D already exists: it will be rewritten from $LOGS (one source per day)"
say "1 pm_signals + pm_orders from $LOGS"
python -u research/pm_factors/pm_signals.py --logs $LOGS --dates $D --out $DATA/pm_signals > $O/${D}_signals.txt 2>&1 || say "  pm_signals FAILED: $(tail -2 $O/${D}_signals.txt)"
grep -E "signals,|factor lines|order match|factor program|->" $O/${D}_signals.txt | cut -c1-220 | sed 's/^/  /'
python -u research/pm_factors/pm_orders.py --logs $LOGS --dates $D > $O/${D}_orders.txt 2>&1 || say "  pm_orders FAILED: $(tail -2 $O/${D}_orders.txt)"
tail -12 $O/${D}_orders.txt | cut -c1-220 | sed 's/^/  /'
ls -la $DATA/pm_signals/state/date=$D.parquet $DATA/pm_orders/date=$D/ 2>&1 | cut -c1-160 | sed 's/^/  /'
say "2 DataYesSharesL2 $D -> dysl2_raw, catalog_dysl2"
if [ -e $DY/$D.zip ] || [ -d $DY/$D ]; then
  python -u research/decode_datayes_day.py --root $DY --date $D --out $DATA/dysl2_raw --workers $W --kinds quotation,order,transaction,index \
    > $O/${D}_decode.txt 2>&1 || say "  decode FAILED: $(tail -2 $O/${D}_decode.txt)"
  tail -3 $O/${D}_decode.txt | cut -c1-200 | sed 's/^/  /'
  CRUCIBLE_SOURCE=dysl2 python -u research/catalog.py > $O/${D}_catalog.txt 2>&1 || say "  catalog FAILED: $(tail -1 $O/${D}_catalog.txt)"
  CRUCIBLE_SOURCE=dysl2 python -c "import sys; sys.path.insert(0,'research/pm_factors'); import duckdb, dcp_build as B; c=duckdb.connect(str(B.CATALOG), read_only=True); print('  quotes on $D:', c.sql('SELECT count(*), count(DISTINCT symbol_id) FROM raw_quotation WHERE date = $D').fetchone())" 2>&1 | tail -1
else
  say "  no $DY/$D(.zip): skipped"
fi
say "3 decisions"
python -u research/pm_factors/pm_decisions.py --dates $D > $O/${D}_decisions.txt 2>&1 || say "  pm_decisions FAILED: $(tail -1 $O/${D}_decisions.txt)"
grep -E "^==|per stock|share of snapshots|constants" $O/${D}_decisions.txt | cut -c1-220 | sed 's/^/  /'
for f in $LOGS/*_${D}_*.log; do
  echo "  $(basename $f): nullptr $(grep -a -c 'order is nullptr' $f), ORD $(grep -a -c 'trk ORD' $f), SEND $(grep -a -c 'trk SEND' $f), REJ $(grep -a -c 'trk REJ' $f), TRD $(grep -a -c 'trk TRD' $f)"
done
say "r171 done $D"
