#!/usr/bin/env bash
# Full-day check of a copied feitu tree on 97, one day after another. First the feitu days already decoded in feitu_raw are
# checked as the known-good reference (read in place, nothing deleted); then each new day is decoded (a corrupt file is
# listed, not fatal), checked -- rows per file, arrival tiling, order/trade id holes, quotes against trades, index
# cadence -- and its decoded copy deleted; finally every day is compared with the reference and the bad files named.
# Foreground; logs in data/reports/feitu_day_check. ROOT / DAYS / BASE / T / STAGE can be overridden from the environment.
set -u
cd /work/crucible
ROOT=${ROOT:-/data/feitu_0901_0919}
STAGE=${STAGE:-/work/crucible_data/feitu_check_raw}  # not feitu_raw: these days stay out of the catalog until they pass
FEITU=/work/crucible_data/feitu_raw
BASE=${BASE:-20260922 20260923 20260924 20260928 20260929 20260930 20261008 20261009}
O=data/reports/feitu_day_check
T=${T:-32}
mkdir -p $O

echo "### layout"
python -u research/feitu_check.py $ROOT --layout-only 2>&1 | tee $O/layout.txt
DAYS=${DAYS:-$(find $ROOT -name 'v3_*.zst' | grep -o '20[0-9][0-9]_[0-9][0-9]_[0-9][0-9]' | sort -u | tr -d _ | tr '\n' ' ')}
echo "new days: $DAYS"
echo "reference days: $BASE"
df -h $(dirname $STAGE) | tail -1

for d in $BASE; do
  echo "### reference $d (from $FEITU)"
  python -u research/feitu_day_check.py --raw $FEITU --date $d --out $O 2>&1 | tee $O/check_$d.txt \
    | grep -E "files,|missing in|quotes market|index cadence|done|Error|error:"
done

for d in $DAYS; do
  echo "### $d"
  python -u research/decode_feitu_day.py --root $ROOT --date $d --out $STAGE --threads $T --keep-going --overwrite 2>&1 \
    | tee $O/decode_$d.txt | grep -E "no files|^done|Error|error:"
  python -u research/feitu_day_check.py --raw $STAGE --date $d --out $O 2>&1 | tee $O/check_$d.txt \
    | grep -E "files,|missing in|quotes market|index cadence|done|Error|error:"
  rm -rf $STAGE/kind=*/date=$d
done

echo
echo "########## SUMMARY (photograph from here) ##########"
python -u research/feitu_day_check.py --out $O --report --base "$(echo $BASE | tr ' ' ',')" 2>&1 | tee $O/report.txt \
  | grep -vE "^│ (---|i64|i32|i16|f64|str|u32)|^shape"
