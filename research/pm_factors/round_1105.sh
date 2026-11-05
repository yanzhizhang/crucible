#!/usr/bin/env bash
# One batch on 97 (2026-11-05): ticks + pm basics for the days PM added (to LAST), snapshot flows into basics, index snapshots,
# the samplerS probe over every PM day with a full window, the fingerprint of the unknown model families, PM's livev1 tree,
# and the model round at random mid-bar times (index candidates; the 202609 model is now on 97) over the last 10 PM days.
#   CPUS=16 bash research/pm_factors/round_1105.sh 2>&1 | tee data/reports/round_1105.txt
# Each step logs to data/reports/r1105_*.txt; photograph from the SUMMARY line down.
set -u
CPUS=${CPUS:-16}
LAST=${LAST:-20261105}
R=data/reports; mkdir -p $R
BP=/work/crucible_data/store/pm_factors/basics_pm
DCP=/data/prod/h/HS300/p1_intrading_dcp

echo "### 0/6 ticks for the new PM days, pm basics for days not built yet, PM samplerS cache"
CPUS=$CPUS bash research/backfill.sh 20261028 $LAST > $R/r1105_backfill.txt 2>&1
NEW=$(ls /work/crucible_data/feitu_raw/kind=transaction | sed 's/date=//' | awk -v l=$LAST '$1>=20260910 && $1<=l' \
      | while read d; do [ -f $BP/date=$d/m1.parquet ] || echo $d; done | paste -sd,)
[ -z "$NEW" ] || nice -n 10 python -u research/pm_factors/basics.py --convention pm --dates $NEW --thr-dir /data/prod/hstats/HS300/ZZCR \
  --workers 9 --threads 4 --memory 24GB > $R/r1105_basics_new.txt 2>&1
python -u research/pm_factors/same_gap.py --families ZZAL,ZZMS,ZZQO,ZZTS,ZZXC --end $LAST > $R/r1105_cache.txt 2>&1
DATES=$(ls $BP | sed 's/date=//' | awk '$1>=20260910' | paste -sd,)
DAYSP=$(ls /data/prod/hstats/HS300/ZZQO | awk -v l=$LAST '$1>=20261016 && $1<=l' | paste -sd,)  # PM days whose 22-day window has basics
DAYSM=$(ls $DCP | sed 's/\.jz$//' | grep -E '^[0-9]{8}$' | awk -v l=$LAST '$1<=l' | sort | tail -10 | paste -sd' ')  # last 10 days with PM state

echo "### 1/6 basics: snapshot flows (--only snap)"
nice -n 10 python -u research/pm_factors/basics.py --convention pm --only snap --dates $DATES --thr-dir /data/prod/hstats/HS300/ZZCR \
  --workers 24 --threads 4 --memory 24GB > $R/r1105_basics.txt 2>&1

echo "### 2/6 index snapshots (KINDS=index)"
KINDS=index CPUS=$CPUS bash research/backfill.sh 20260910 $LAST > $R/r1105_index.txt 2>&1

echo "### 3/6 samplerS probe over $(echo $DAYSP | tr ',' ' ' | wc -w) PM days: snapshot flows against the trade flows"
python -u research/pm_factors/window_probe.py --unit --basics $BP --days $DAYSP \
  --combos "ZZXC:q:vol,ZZXC:q:snap_vol,ZZXC:n:amt,ZZXC:n:snap_amt,ZZTS:q:b_amt,ZZTS:q:snap_bamt,ZZQO:q:ordvolrc_sum,ZZMS:q:cum_vol_oa" \
  > $R/r1105_probe.txt 2>&1

echo "### 4/6 fingerprint of the unknown model families"
python -c "
import polars as pl
pl.Config.set_tbl_rows(90); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
f = pl.read_parquet('$R/stage5/fingerprint.parquet')
u = ['ZZQF','ZZOP','ZZEL','ZZAG','ZZBT','ZZSA','ZZPP','ZZRI','ZZHI','ZZHO','ZZEG','ZZUC','ZZRG','ZZAA','ZZYJ','ZZBO','ZZJA','ZZFQ','ZZPZ']
print(f.filter(pl.col('family').is_in(u)).select('feature','gain_share','kind','lo','hi','q_p10','q_med','q_p90','half_int','at_zero')
      .sort('gain_share', descending=True))
" > $R/r1105_fp.txt 2>&1

echo "### 5/6 PM's livev1 tree"
{ find /data/prod/h/HS300/livev1 -maxdepth 3 | head -15; du -sh /data/prod/h/HS300/livev1; } > $R/r1105_livev1.txt 2>&1

echo "### 6/6 model round at random mid-bar times (MARKS=rand:240) over: $DAYSM"
CPUS=$CPUS MARKS=rand:240 DAYS="$DAYSM" bash research/pm_factors/model_round.sh > $R/r1105_model.txt 2>&1

echo
echo "########## SUMMARY 1105 (photograph from here) ##########"
echo "-- 0 new days: backfill / pm basics / cache"
grep -E "decoded|FAILED|no feitu" $R/r1105_backfill.txt | tail -4
echo "new pm basics: ${NEW:-none} -> $(grep -c "stock-minutes" $R/r1105_basics_new.txt 2>/dev/null) ok"; grep FAILED $R/r1105_basics_new.txt 2>/dev/null | head -3
grep -E "days$" $R/r1105_cache.txt
echo "-- 1 basics snap: days ok / failed"
grep -c "stock-minutes" $R/r1105_basics.txt; grep FAILED $R/r1105_basics.txt | head -3
echo "-- 2 index snapshots"
grep -E "decoded|FAILED|no feitu" $R/r1105_index.txt | tail -4; ls -d /work/crucible_data/feitu_raw/kind=index/date=* 2>/dev/null | wc -l | xargs echo "index days on disk:"
echo "-- 3 samplerS probe ($DAYSP)"
grep -E "^==|exch|PM/ours|rank|nothing" $R/r1105_probe.txt
echo "-- 4 fingerprint"
cat $R/r1105_fp.txt
echo "-- 5 livev1"
cat $R/r1105_livev1.txt
echo "-- 6 model round ($DAYSM)"
grep -E "index snapshots for" $R/stage6/ic.txt 2>/dev/null | head -2
sed -n '/########## SUMMARY/,$p' $R/r1105_model.txt | tail -n +2
