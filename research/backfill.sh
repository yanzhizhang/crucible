#!/usr/bin/env bash
# Backfill tick history on 97: decode -> catalog -> basics, day by day. Serves both lines: samplerS windows longer than
# 20 days (window_match) and a longer training history (strategy).
#
# ARCH (default /data/feitu) may hold either layout, per day:
#   <ARCH>/YYYYMMDD/v3_<kind>_YYYY_MM_DD_<SH|SZ>.zst        TickRepository archive (2.23), one file per stream / exchange
#   <ARCH>/YYYYMMDD/v3_<kind>/v3_<kind>_YYYY_MM_DD_HH_MM.zst  live minute dumps, one folder per stream (the layout on 97)
#   <ARCH>/<FROM>_<TO>/v3_<kind>/YYYY_MM_DD/...              the same dumps grouped by date range (also read)
# Days with no feitu data come from the DataYes XTP L2 packages <XTP>/YYYYMMDD.zip (decode_tdf_day.py), same raw layout.
#   CPUS=16 bash research/backfill.sh 20260901 20261009 2>&1 | tee data/reports/backfill.txt
#
# Days already decoded are skipped (FORCE=1 redecodes). Only order / transaction / quotation are decoded.
set -u
FROM=${1:?first day YYYYMMDD}; TO=${2:?last day YYYYMMDD}
CPUS=${CPUS:-256}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
ARCH=${ARCH:-/data/feitu}
RAW=${RAW:-/work/crucible_data/feitu_raw}
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }

df -h $(dirname $RAW) | tail -1
# days of either layout: YYYYMMDD folders, or YYYY_MM_DD folders under <range>/v3_transaction
days=$( { ls $ARCH 2>/dev/null | grep -E '^[0-9]{8}$'; ls $ARCH/*_*/v3_transaction 2>/dev/null | grep -E '^[0-9]{4}_[0-9]{2}_[0-9]{2}$' | tr -d _; } \
        | sort -u | awk -v a=$FROM -v b=$TO '$1 >= a && $1 <= b')
[ -n "$days" ] || echo "no feitu days in $ARCH between $FROM and $TO"
# the decoder's --root for a day: its archive folder, else the range folder holding v3_transaction/YYYY_MM_DD
root_of() { [ -d $ARCH/$1 ] && { echo $ARCH/$1; return; }; ls -d $ARCH/*_*/v3_transaction/${1:0:4}_${1:4:2}_${1:6:2} 2>/dev/null | head -1 | xargs -r dirname | xargs -r dirname; }
echo "days: $(echo $days | wc -w) ($(echo $days | cut -c1-8) .. $(echo $days | rev | cut -c1-8 | rev))"

done_days=
for d in $days; do
  if ls $RAW/kind=transaction/date=$d/part-* $RAW/kind=order/date=$d/part-* $RAW/kind=quotation/date=$d/part-* >/dev/null 2>&1 && [ "${FORCE:-0}" != 1 ]; then echo "$d: decoded already"; done_days="$done_days $d"; continue; fi
  root=$(root_of $d)
  t0=$(date +%s)
  run research/decode_feitu_day.py --root $root --date $d --out $RAW --kinds order,transaction,quotation --threads $CPUS --keep-going \
    $([ "${FORCE:-0}" = 1 ] && echo --overwrite) > data/reports/backfill_decode_$d.txt 2>&1 \
    && { echo "$d: decoded in $(( $(date +%s) - t0 ))s ($([ $root = $ARCH/$d ] && echo archive || echo minute dumps)), $(du -sh $RAW/kind=*/date=$d 2>/dev/null | awk '{s=s" "$1} END {print s}')"; done_days="$done_days $d"; } \
    || { echo "$d: decode FAILED"; tail -3 data/reports/backfill_decode_$d.txt; }
done


# days feitu lacks but the DataYes XTP L2 packages have (<XTP>/YYYYMMDD.zip): one pool for all of them, same raw layout
XTP=${XTP:-/mnt/TickDataRepo/XTPSharesL2}
XW=${XW:-96}; [ "$XW" -gt "$CPUS" ] && XW=$CPUS  # worker processes; each holds one symbol's csv, so fewer than CPUS
xdays=$(ls $XTP 2>/dev/null | grep -oE '^[0-9]{8}' | sort -u | awk -v a=$FROM -v b=$TO '$1 >= a && $1 <= b' \
        | while read d; do echo " $(echo $days) " | grep -q " $d " || [ -d $RAW/kind=transaction/date=$d -a "${FORCE:-0}" != 1 ] || echo $d; done)
if [ -n "$xdays" ]; then
  xl=$(echo $xdays | tr ' ' ',')
  echo "### XTP L2 (DataYes) for days feitu lacks: $xl"
  t0=$(date +%s)
  taskset -c 0-$((CPUS - 1)) nice -n 5 python -u research/decode_tdf_day.py --root $XTP --date $xl --out $RAW --workers $XW \
    $([ "${FORCE:-0}" = 1 ] && echo --overwrite) > data/reports/backfill_xtp.txt 2>&1 \
    && { echo "xtp: $(echo $xdays | wc -w) days decoded in $(( $(date +%s) - t0 ))s"; done_days="$done_days $xdays"; } \
    || { echo "xtp: decode FAILED"; tail -5 data/reports/backfill_xtp.txt; }
fi

echo "### catalog"
run research/catalog.py > data/reports/backfill_catalog.txt 2>&1 || tail -3 data/reports/backfill_catalog.txt

echo "### basics (universe = PM's ZZCR samplerR of the latest earlier PM day; days without one use the CED index members)"
list=$(echo $done_days | tr ' ' '\n' | grep . | paste -sd,)
[ -n "$list" ] && run research/pm_factors/basics.py --dates $list --thr-dir /data/prod/hstats/HS300/ZZCR --workers $((CPUS / 4)) \
  > data/reports/backfill_basics.txt 2>&1 || { grep FAILED data/reports/backfill_basics.txt || tail -3 data/reports/backfill_basics.txt; }
ls -d /work/crucible_data/store/pm_factors/basics/date=* | wc -l | xargs echo "basics days now:"
