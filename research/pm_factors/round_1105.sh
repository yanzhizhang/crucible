#!/usr/bin/env bash
# One batch on 97 (2026-11-05): snapshot flows into basics, index snapshots decoded, samplerS probe on the snapshot flows,
# the fingerprint of the unknown model families, and the model round at random mid-bar times (with index candidates).
#   CPUS=16 bash research/pm_factors/round_1105.sh 2>&1 | tee data/reports/round_1105.txt
# Each step logs to data/reports/r1105_*.txt; photograph from the SUMMARY line down.
set -u
CPUS=${CPUS:-16}
R=data/reports; mkdir -p $R
BP=/work/crucible_data/store/pm_factors/basics_pm
DATES=$(ls $BP | sed 's/date=//' | awk '$1>=20260910' | paste -sd,)
DAYS6=20261020,20261021,20261022,20261023,20261026,20261027

echo "### 1/5 basics: snapshot flows (--only snap)"
nice -n 10 python -u research/pm_factors/basics.py --convention pm --only snap --dates $DATES --thr-dir /data/prod/hstats/HS300/ZZCR \
  --workers 24 --threads 4 --memory 24GB > $R/r1105_basics.txt 2>&1

echo "### 2/5 index snapshots (KINDS=index)"
KINDS=index CPUS=$CPUS bash research/backfill.sh 20260910 20261027 > $R/r1105_index.txt 2>&1

echo "### 3/5 samplerS probe: snapshot flows against the trade flows"
python -u research/pm_factors/window_probe.py --unit --basics $BP --days $DAYS6 \
  --combos "ZZXC:q:vol,ZZXC:q:snap_vol,ZZXC:n:amt,ZZXC:n:snap_amt,ZZTS:q:b_amt,ZZTS:q:snap_bamt" > $R/r1105_probe.txt 2>&1

echo "### 4/5 fingerprint of the unknown model families"
python -c "
import polars as pl
pl.Config.set_tbl_rows(90); pl.Config.set_tbl_width_chars(220); pl.Config.set_float_precision(4)
f = pl.read_parquet('$R/stage5/fingerprint.parquet')
u = ['ZZQF','ZZOP','ZZEL','ZZAG','ZZBT','ZZSA','ZZPP','ZZRI','ZZHI','ZZHO','ZZEG','ZZUC','ZZRG','ZZAA','ZZYJ','ZZBO','ZZJA','ZZFQ','ZZPZ']
print(f.filter(pl.col('family').is_in(u)).select('feature','gain_share','kind','lo','hi','q_p10','q_med','q_p90','half_int','at_zero')
      .sort('gain_share', descending=True))
" > $R/r1105_fp.txt 2>&1

echo "### 5/5 model round at random mid-bar times (MARKS=rand:240)"
CPUS=$CPUS MARKS=rand:240 bash research/pm_factors/model_round.sh > $R/r1105_model.txt 2>&1

echo
echo "########## SUMMARY 1105 (photograph from here) ##########"
echo "-- 1 basics: days ok / failed"
grep -c "stock-minutes" $R/r1105_basics.txt; grep FAILED $R/r1105_basics.txt | head -3
echo "-- 2 index snapshots"
grep -E "decoded|FAILED|no feitu" $R/r1105_index.txt | tail -4; ls -d /work/crucible_data/feitu_raw/kind=index/date=* 2>/dev/null | wc -l | xargs echo "index days on disk:"
echo "-- 3 samplerS probe"
grep -E "^==|exch|PM/ours|rank|nothing" $R/r1105_probe.txt
echo "-- 4 fingerprint"
cat $R/r1105_fp.txt
echo "-- 5 model round"
grep -E "index snapshots for" $R/stage6/ic.txt 2>/dev/null | head -2
sed -n '/########## SUMMARY/,$p' $R/r1105_model.txt | tail -n +2
