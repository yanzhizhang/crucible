#!/usr/bin/env bash
# Backfill tick history on 97: decode -> catalog -> basics, day by day. Serves both lines: samplerS windows longer than
# 20 days (window_match) and a longer training history (strategy).
#
# ARCH (default /data/feitu) may hold either layout, per day:
#   <ARCH>/YYYYMMDD/v3_<kind>_YYYY_MM_DD_<SH|SZ>.zst        TickRepository archive (2.23), one file per stream / exchange
#   <ARCH>/v3_<kind>/YYYY_MM_DD/v3_<kind>_..._HH_MM.zst     live minute dumps, one folder per stream and day
#   CPUS=16 bash research/backfill.sh 20260901 20261009 2>&1 | tee data/reports/backfill.txt
#
# Days already decoded are skipped (FORCE=1 redecodes). Only order / transaction / quotation are decoded.
set -u
FROM=${1:?first day YYYYMMDD}; TO=${2:?last day YYYYMMDD}
CPUS=${CPUS:-16}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
ARCH=${ARCH:-/data/feitu}
RAW=${RAW:-/work/crucible_data/feitu_raw}
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }

df -h $(dirname $RAW) | tail -1
# days of either layout: YYYYMMDD folders, or YYYY_MM_DD folders under v3_transaction
days=$( { ls $ARCH 2>/dev/null | grep -E '^[0-9]{8}$'; ls $ARCH/v3_transaction 2>/dev/null | grep -E '^[0-9]{4}_[0-9]{2}_[0-9]{2}$' | tr -d _; } \
        | sort -u | awk -v a=$FROM -v b=$TO '$1 >= a && $1 <= b')
[ -n "$days" ] || { echo "no archive days in $ARCH between $FROM and $TO"; exit 1; }
echo "days: $(echo $days | wc -w) ($(echo $days | cut -c1-8) .. $(echo $days | rev | cut -c1-8 | rev))"

done_days=
for d in $days; do
  if [ -d $RAW/kind=transaction/date=$d ] && [ "${FORCE:-0}" != 1 ]; then echo "$d: decoded already"; done_days="$done_days $d"; continue; fi
  root=$ARCH/$d; [ -d $root ] || root=$ARCH  # archive day folder, else the minute-dump tree (the decoder finds v3_<kind>/YYYY_MM_DD)
  t0=$(date +%s)
  run research/decode_feitu_day.py --root $root --date $d --out $RAW --kinds order,transaction,quotation --threads $CPUS --keep-going \
    $([ "${FORCE:-0}" = 1 ] && echo --overwrite) > data/reports/backfill_decode_$d.txt 2>&1 \
    && { echo "$d: decoded in $(( $(date +%s) - t0 ))s ($([ $root = $ARCH ] && echo minute dumps || echo archive)), $(du -sh $RAW/kind=*/date=$d 2>/dev/null | awk '{s=s" "$1} END {print s}')"; done_days="$done_days $d"; } \
    || { echo "$d: decode FAILED"; tail -3 data/reports/backfill_decode_$d.txt; }
done

echo "### catalog"
run research/catalog.py > data/reports/backfill_catalog.txt 2>&1 || tail -3 data/reports/backfill_catalog.txt

echo "### basics (universe = PM's ZZCR samplerR of the latest earlier PM day; days without one use the CED index members)"
list=$(echo $done_days | tr ' ' '\n' | grep . | paste -sd,)
[ -n "$list" ] && run research/pm_factors/basics.py --dates $list --thr-dir /data/prod/hstats/HS300/ZZCR --workers $((CPUS / 4)) \
  > data/reports/backfill_basics.txt 2>&1 || tail -3 data/reports/backfill_basics.txt
ls -d /work/crucible_data/store/pm_factors/basics/date=* | wc -l | xargs echo "basics days now:"
