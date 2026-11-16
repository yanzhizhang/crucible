#!/usr/bin/env bash
# Round 138 on 97: five independent pinning experiments, one tab each, all read-only on inputs, outputs only under data/.
# Pass the experiment as $1:
#   atlas  E1-E4  model files only: segment ranges x generations, sibling ladders, split neighbours, leaf grammar (~2 min)
#   e5     the 8 unit-violating columns (R130): their bases + session clocks x transforms x scales vs per-segment ranges (~20 min)
#   e6     engine-sourced open / stand-in columns vs within-family state formulas of BIT-EXACT keys only (~30 min)
#   e7     hstats-sourced columns vs live quantity / profile cell ratios (~30-60 min)
#   e8     every unbuilt averaged .jz key vs EMA(q) at the half-life PM's own weight implies, 1112 + 1113 (~10 min)
#   bash research/r138.sh <exp> 2>&1 | tee data/reports/r138_<exp>.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
DAYS=20260430,20261112
SF="--x-dir data/reports/stage6_r99_{mmdd} --base data/featgen_r99/base_{date}.parquet --models $DATA/models_gen/B"
case "${1:?atlas|e5|e6|e7|e8}" in
atlas)
  python -u research/featgen/model_atlas.py all --gens $DATA/models_gen --date 20261113 --out data/reports/atlas_r138 ;;
e5)
  python -u research/featgen/seg_fit.py --days $DAYS --target zzqf.q.c3,zzrg,zzwa.g,zzag2.e,zzag2.a,zzro2.a,zzop2.b,zzqf7.q.c2 $SF --clocks \
    --extra "data/side_r105/ratio_{date}.parquet" --transforms=id,-id,1/b,dm,-dm,srank,log,abs4 --scales 1,1e4,1e-4,100,0.01,0.001 --top 6 2>&1 | grep -v Warning ;;
e6)
  ls -l data/side_r107/ data/side_r105/engine_* 2>&1 | sed 's/^/  /' | head
  python -u research/featgen/seg_fit.py --days $DAYS --target data/reports/audit/audit.csv:engine $SF --no-base \
    --extra "data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" --exact-csv data/reports/r125/verify_20261113.csv \
    --transforms=id,-id,1/b,dm,srank,log --scales 1,1e4,100 --top 6 2>&1 | grep -v Warning ;;
e7)
  python -u research/featgen/seg_fit.py --days $DAYS --target data/reports/audit/audit.csv:hstats $SF --extra "data/side_r95/hstats_prev_{date}.parquet" \
    --cand-re '^x\.ZZ' --ratio '^x\.ZZWA\..*@0$=spread|lpe|lee|rpe|half' --ratio '^x\.ZZQO\..*@0$=ord|flow|refill|vol' \
    --ratio '^x\.ZZAL\..*@0$=amount|depth|imbv|amihud|buy' --ratio '^x\.ZZ(XC|TS)\..*@0$=amount|vol|turn' --max-ratios 3000 \
    --transforms=id,-id,1/b,dm,srank,log --scales 1,1e4,100 --top 6 2>&1 | grep -v Warning ;;
e8)
  for D in 20261112 20261113; do
    python -u research/pm_factors/ema_sweep.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --raw $DATA/dysl2_raw --skip-verified data/reports/r125/verify_20261113.csv --top 4 2>&1 | grep -vE "Warning|polyfit"
  done ;;
esac
echo "[$(date +%T)] r138 $1 done"
