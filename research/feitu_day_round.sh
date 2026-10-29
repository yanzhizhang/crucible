#!/usr/bin/env bash
# Full-day check of a copied feitu tree on 97, one day after another: decode the day (a corrupt file is listed, not fatal),
# count rows per file and holes in the order/trade id sequence, delete the decoded day, then flag files over all days.
# Foreground; logs in data/reports/feitu_day_check. ROOT / DAYS / T / STAGE can be overridden from the environment.
set -u
cd /work/crucible
ROOT=${ROOT:-/data/feitu_0901_0919}
STAGE=${STAGE:-/work/crucible_data/feitu_check_raw}  # not feitu_raw: these days stay out of the catalog until they pass
O=data/reports/feitu_day_check
T=${T:-32}
mkdir -p $O

echo "### layout"
python -u research/feitu_check.py $ROOT --layout-only 2>&1 | tee $O/layout.txt
DAYS=${DAYS:-$(find $ROOT -name 'v3_*.zst' | grep -o '20[0-9][0-9]_[0-9][0-9]_[0-9][0-9]' | sort -u | tr -d _ | tr '\n' ' ')}
echo "days: $DAYS"
df -h $(dirname $STAGE) | tail -1

for d in $DAYS; do
  echo "### $d"
  python -u research/decode_feitu_day.py --root $ROOT --date $d --out $STAGE --threads $T --keep-going --overwrite 2>&1 \
    | tee $O/decode_$d.txt | grep -E "no files|^done|Error|error:"
  python -u research/feitu_day_check.py --raw $STAGE --date $d --out $O 2>&1 | tee $O/check_$d.txt | grep -E "files,|missing in|done|Error|error:"
  rm -rf $STAGE/kind=*/date=$d
done

echo
echo "########## SUMMARY (photograph from here) ##########"
python -u research/feitu_day_check.py --out $O --report 2>&1 | tee $O/report.txt
