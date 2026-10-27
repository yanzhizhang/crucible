#!/usr/bin/env bash
# One XTP round on 97: probe the TDF layout, convert the overlap day into xtp_raw and compare it with feitu,
# convert the new days into feitu_raw, refresh the catalog. Foreground; every step tees its log to data/reports/stage8.
set -u
cd /work/crucible
R=data/reports/stage8
mkdir -p $R
ROOT=${ROOT:-/data/xtp_tdf}
OVERLAP=${OVERLAP:-20261009}
NEW=${NEW:-20261024,20261026}
T=$(nproc)

echo "### 0/4 probe the layout ($OVERLAP)"
python -u research/decode_tdf_day.py --root $ROOT --date $OVERLAP --probe 2>&1 | tee $R/probe.txt | grep -E "^==|UNMAPPED|left at 0|symbols under|rows [0-9]"

echo "### 1/4 overlap day $OVERLAP -> /work/crucible_data/xtp_raw"
python -u research/decode_tdf_day.py --root $ROOT --date $OVERLAP --out /work/crucible_data/xtp_raw --threads $T 2>&1 | tee $R/convert_$OVERLAP.txt | grep -E "rows in|market_id|^│|^\[" | tail -40

echo "### 2/4 compare $OVERLAP with feitu"
python -u research/compare_xtp_feitu.py --date $OVERLAP 2>&1 | tee $R/compare_$OVERLAP.txt | tail -5

echo "### 3/4 new days -> /work/crucible_data/feitu_raw"
for d in ${NEW//,/ }; do
  python -u research/decode_tdf_day.py --root $ROOT --date $d --threads $T 2>&1 | tee $R/convert_$d.txt | grep -E "rows in" 
done

echo "### 4/4 catalog"
python -u research/catalog.py 2>&1 | tail -3

echo
echo "########## SUMMARY (photograph from here) ##########"
grep -E "UNMAPPED|left at 0" $R/probe.txt | sort -u
sed -n '/totals per exchange/,$p' $R/compare_$OVERLAP.txt | grep -vE "^shape|^│ ---|^│ i16|^│ str|^│ f64|^│ u32|^│ i64|^│ decimal" | head -60
for d in $OVERLAP ${NEW//,/ }; do
  echo "-- $d coverage"
  grep -A4 "market_id ┆ records ┆ seq_span" $R/convert_$d.txt | grep -E "^│ 355"
  grep -A6 "market_id ┆ snap_median" $R/convert_$d.txt | grep -E "^│ 355"
done
