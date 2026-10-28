#!/usr/bin/env bash
# One round on 97: our own model on every tick day (leave-one-day-out, then walk-forward on the same panel), PM's sod
# stats against tick candidates, and a look at PM's new files. Foreground; each step tees its log to data/reports.
set -u
cd /work/crucible
R=data/reports
mkdir -p $R/stage5 $R/stage7/all $R/stage7/walk

echo "### 1/4 own model, every day in the catalog, leave-one-day-out (builds the panel)"
python -u research/pm_factors/own_model.py --dates all --out $R/stage7/all 2>&1 | tee $R/stage7/all/own.txt \
  | grep -E "days in raw_quotation|universe only|skip|out-of-sample IC|Error|error:"

echo "### 2/4 own model, walk-forward on the same panel (each day from earlier days only)"
python -u research/pm_factors/own_model.py --from-panel $R/stage7/all/panel.parquet --cv walk --min-train 5 --out $R/stage7/walk 2>&1 \
  | tee $R/stage7/walk/own.txt | grep -E "cv walk|out-of-sample IC|Error|error:"

echo "### 3/4 sod stats (msr / mb / vr) against tick and CED candidates"
python -u research/pm_factors/sod_stats_match.py --dates 20260923,20260924,20260929,20260930,20261009 2>&1 | tee $R/stage5/sod_stats.txt | tail -3

echo "### 4/4 PM's new files"
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
for cv in all walk; do
  f=$R/stage7/$cv/own.txt
  echo "== own model, $cv"
  grep -E "^panel:|out-of-sample IC" $f
  sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ (lgbm|ridge) +┆ 20" | grep lgbm | cut -c1-110
  sed -n '/^== out of sample/,/^== LightGBM/p' $f | grep -E "^│ (lgbm|ridge) +┆ [-0-9.]+ +┆" | grep -v "┆ 20"
  sed -n '/^== T0 round trips/,/^== taker round trips/p' $f | grep -E "^│ (lgbm|ridge) +┆ 300 +┆ 0\.0[125]00 "
done
sed -n '/^== taker round trips by quoted spread/,$p' $R/stage7/walk/own.txt | grep -E "^│ lgbm .*┆ 300 " | head -12
sed -n '/^== LightGBM gain share/,/^== single-feature/p' $R/stage7/all/own.txt | grep "^│" | grep -vE "^│ (---|str|f64)" | head -12
sed -n '/== PM sod stats vs candidates/,$p' $R/stage5/sod_stats.txt | head -34
cut -c1-200 $R/stage5/pm_new_files.txt
