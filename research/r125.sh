#!/usr/bin/env bash
# Round 125 on 97, foreground, read-only on inputs; writes data/engine_r125, data/implied_scan_r125, data/reports. ~15 min.
# R124b: zzmh.r = 4 ((hm - p) / hm)^2 exactly (PM / q median 4.0000, IQR 0.0000, 294 stocks, both days), averaged over
# snapshots with half-life ~1 s (PM's weight 1.1439). 1: the rule in the C++ engine ("zzmh.r": snap, 1 s, mh_dd2), selftest,
# verify vs PM on 1106 / 1112 / 1113. 2: the zzmh column (11% of training rows <= 0.0045) as 2 x drawdown = sqrt(zzmh.r)
# or 4 dd^2: fingerprint + scan on 0430 / 1112 / 1113 (drawdown from R107's state ratios (px - zzmh.hm) / zzmh.hm).
#   bash research/r125.sh 2>&1 | tee data/reports/r125.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
O=data/reports/r125; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -1
for D in 20261106 20261112 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261112) JZ=$SP/$D.jz; PV=$SP/20261111.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --marks rand:240 --threads 32 \
    --out data/engine_r125 --work data/engine_r125/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r125/engine_$D.json.gz --pm $JZ --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python -c "import polars as pl; t=pl.read_csv('$O/verify_$D.csv'); print('$D', t.filter(pl.col('key').str.starts_with('zzmh')).rows())"
done
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzmh --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r107/state_{date}.parquet" --transforms "id,-id,-2b,2b,4b2,sqrt" --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzmh --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r107/state_{date}.parquet" --only-extra --transforms "-2b,4b2" --rank-day 20260430 --models $DATA/models_gen/B --top 8 \
  --work data/implied_scan_r125 2>&1 | grep -v RuntimeWarning | grep -iE "baseline|zzmh|hm|share|best on" | cut -c1-260
echo "[$(date +%T)] done"
