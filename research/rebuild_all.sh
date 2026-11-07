#!/usr/bin/env bash
# Rebuild everything from the raw archives, overwriting: ticks (feitu where /data/feitu has the day, else the DataYes
# XTP packages), index snapshots, pm basics, a level-1 sanity check, then the model round against PM's predictions.
#   CPUS=64 bash research/rebuild_all.sh 2>&1 | tee data/reports/rebuild_all.txt
# Logs per step in data/reports/rb_*.txt; photograph from the SUMMARY line down. SKIP_MODEL=1 stops before the model round.
set -u
FROM=${FROM:-20260901}; TO=${TO:-20261106}
CPUS=${CPUS:-64}
R=data/reports; mkdir -p $R
BP=/work/crucible_data/store/pm_factors/basics_pm
t0=$(date +%s)
stamp() { echo "[$(( ($(date +%s) - t0) / 60 )) min] $*"; }

stamp "1/5 ticks $FROM..$TO (overwrite)"
CPUS=$CPUS bash research/backfill.sh $FROM $TO > $R/rb_ticks.txt 2>&1

stamp "2/5 index snapshots"
KINDS=index CPUS=$CPUS bash research/backfill.sh $FROM $TO > $R/rb_index.txt 2>&1

stamp "3/5 pm basics"
D=$(ls /work/crucible_data/feitu_raw/kind=transaction | sed 's/date=//' | awk -v a=$FROM -v b=$TO '$1>=a && $1<=b' | paste -sd,)
nice -n 10 python -u research/pm_factors/basics.py --convention pm --dates $D --thr-dir /data/prod/hstats/HS300/ZZCR \
  --workers 24 --threads 4 --memory 24GB > $R/rb_basics.txt 2>&1

stamp "4/5 level-1 check"
python -c "
import polars as pl, glob
for f in sorted(glob.glob('$BP/date=2026*/m5.parquet')):
    d = f.split('date=')[1][:8]
    if not ('$FROM' <= d <= '$TO'): continue
    print(d, pl.read_parquet(f, columns=['symbol_id', 'bv1']).filter(pl.col('symbol_id') == 1)['bv1'].median())" > $R/rb_check.txt 2>&1

if [ "${SKIP_MODEL:-0}" != 1 ]; then
  stamp "5/5 model round against PM's predictions"
  CPUS=16 MARKS=rand:240 TARGET=/work/crucible_data/pm_signals MAXF=100 ICARGS="--passes 2 --max-cdf 0.08 --min-in-range 0.8" \
    DAYS="20261102 20261103 20261104 20261105 20261106" bash research/pm_factors/model_round.sh > $R/rb_model.txt 2>&1
fi
stamp "done"

echo
echo "########## SUMMARY rebuild_all (photograph from here) ##########"
echo "-- 1 ticks"
grep -cE "decoded in" $R/rb_ticks.txt | xargs echo "feitu days decoded:"; grep -E "^xtp:|no feitu" $R/rb_ticks.txt
grep -E "FAILED|holds parts" $R/rb_ticks.txt | head -5; grep -lE "holds parts|Traceback" $R/backfill_decode_*.txt $R/backfill_xtp.txt 2>/dev/null | head -5
echo "-- 2 index days on disk: $(ls -d /work/crucible_data/feitu_raw/kind=index/date=* 2>/dev/null | wc -l)"; grep -E "FAILED|Traceback" $R/rb_index.txt | head -3
echo "-- 3 pm basics: $(grep -c 'stock-minutes' $R/rb_basics.txt) days ok"; grep FAILED $R/rb_basics.txt | head -5
echo "-- 4 level-1 volume of 000001 under 1000 (bad days; empty = all fine):"; awk '$2 < 1000' $R/rb_check.txt
if [ "${SKIP_MODEL:-0}" != 1 ]; then
  echo "-- 5 model round"
  grep -E "PM daily inputs" $R/stage6/ic.txt | head -1
  grep -E "^IC: start|all-missing" $R/stage6/ic.txt | tail -2
  grep -E "Traceback|Error" $R/stage6/ic.txt | head -3
  sed -n '/########## SUMMARY/,$p' $R/rb_model.txt | tail -n +2
fi
