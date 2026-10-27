#!/usr/bin/env bash
# One round on 97: the own-model spread table, PM's sod stats against tick candidates, and a look at PM's new files.
# Runs in the foreground; each step tees its full log to data/reports, and the end prints the short summaries.
set -u
cd /work/crucible
R=data/reports
mkdir -p $R/stage5 $R/stage7

echo "### 1/3 own3: taker round trips by spread (from the saved panel)"
python -u research/pm_factors/own_model.py --from-panel $R/stage7/panel.parquet 2>&1 | tee $R/stage7/own3.txt | tail -3

echo "### 2/3 sod stats (msr / mb / vr) against tick and CED candidates"
python -u research/pm_factors/sod_stats_match.py --dates 20260923,20260924,20260929,20260930,20261009 2>&1 | tee $R/stage5/sod_stats.txt | tail -3

echo "### 3/3 PM's new files"
{
  ls -la /data/prod/h/HS300/livev1/rec10/ /data/prod/h/HS300/livev1/rec10/20260430/ /data/prod/h/HS300/livev1/rec10_p/20260430/ 2>&1 | head -20
  ls /data/prod/h/HS300/p1_intrading_dcp | tail -12 | tr '\n' ' '; echo
  d=$(ls /data/prod/hstats/HS300/ZZAC | tail -1)
  python - "$d" <<'EOF'
import sys
import polars as pl
sys.path.insert(0, "research")
from compare_pm import load_pm
d = sys.argv[1]
for f in [f"/data/prod/hstats/HS300/ZZAC/{d}/samplerR.parquet", f"/data/prod/hstats/HS300/ZZCR/{d}/samplerR.parquet"]:
    s = pl.read_parquet_schema(f)
    print("==", f, pl.scan_parquet(f).select(pl.len()).collect().item(), "rows,", len(s), "columns:", list(s)[:25])
for f in [f"/data/prod/hstats/HS300/ZZUG/{d}/samplerSS.nc", f"/data/prod/hstats/HS300/ZZDS/{d}/sampler.nc"]:
    x = load_pm(__import__("pathlib").Path(f))
    print("==", f, x.height, "rows; columns:", x.columns[:12], "; distinct:", {c: x[c].n_unique() for c in x.columns[:6]})
EOF
} 2>&1 | tee $R/stage5/pm_new_files.txt

echo
echo "########## SUMMARY (photograph from here) ##########"
sed -n '/== taker round trips by quoted spread/,$p' $R/stage7/own3.txt | head -48
sed -n '/== PM sod stats vs candidates/,$p' $R/stage5/sod_stats.txt | head -34
cat $R/stage5/pm_new_files.txt | cut -c1-200
