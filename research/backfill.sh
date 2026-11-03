#!/usr/bin/env bash
# Backfill tick history on 97 from the feitu archive copied over from 2.23: decode -> catalog -> basics, day by day.
# Serves both lines: samplerS windows longer than 20 days (window_match) and a longer training history (strategy).
#
# 1. copy the archive days onto 97 yourself (you type the 2.23 password), e.g. July-September:
#      rsync -av -e "ssh -p 60777" --include='2026/' --include='2026/202607*/***' --include='2026/202608*/***' \
#        --include='2026/202609*/***' --exclude='*' root@172.21.2.23:/mnt/nas/pica_data_hdd_1/TickRepository/hdfs_data/ /work/feitu_archive/
# 2. CPUS=16 bash research/backfill.sh 20260701 20260930 2>&1 | tee data/reports/backfill.txt
#
# Days already decoded are skipped (FORCE=1 redecodes). Only order / transaction / quotation are decoded.
set -u
FROM=${1:?first day YYYYMMDD}; TO=${2:?last day YYYYMMDD}
CPUS=${CPUS:-16}
ALL=$(nproc --all); [ "$CPUS" -gt "$ALL" ] && CPUS=$ALL
ARCH=${ARCH:-/work/feitu_archive/2026}
RAW=${RAW:-/work/crucible_data/feitu_raw}
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }

df -h $(dirname $RAW) | tail -1
days=$(ls $ARCH 2>/dev/null | grep -E '^[0-9]{8}$' | awk -v a=$FROM -v b=$TO '$1 >= a && $1 <= b')
[ -n "$days" ] || { echo "no archive days in $ARCH between $FROM and $TO"; exit 1; }
echo "days: $(echo $days | wc -w) ($(echo $days | cut -c1-8) .. $(echo $days | rev | cut -c1-8 | rev))"

done_days=
for d in $days; do
  if [ -d $RAW/kind=transaction/date=$d ] && [ "${FORCE:-0}" != 1 ]; then echo "$d: decoded already"; done_days="$done_days $d"; continue; fi
  t0=$(date +%s)
  run research/decode_feitu_day.py --root $ARCH/$d --date $d --out $RAW --kinds order,transaction,quotation --threads $CPUS --keep-going \
    $([ "${FORCE:-0}" = 1 ] && echo --overwrite) > data/reports/backfill_decode_$d.txt 2>&1 \
    && { echo "$d: decoded in $(( $(date +%s) - t0 ))s, $(du -sh $RAW/kind=*/date=$d 2>/dev/null | awk '{s=s" "$1} END {print s}')"; done_days="$done_days $d"; } \
    || { echo "$d: decode FAILED"; tail -3 data/reports/backfill_decode_$d.txt; }
done

echo "### catalog"
run research/catalog.py > data/reports/backfill_catalog.txt 2>&1 || tail -3 data/reports/backfill_catalog.txt

echo "### basics (universe = PM's ZZCR samplerR of the latest earlier PM day; days without one use the CED index members)"
list=$(echo $done_days | tr ' ' '\n' | grep . | paste -sd,)
[ -n "$list" ] && run research/pm_factors/basics.py --dates $list --thr-dir /data/prod/hstats/HS300/ZZCR --workers $((CPUS / 4)) \
  > data/reports/backfill_basics.txt 2>&1 || tail -3 data/reports/backfill_basics.txt
ls -d /work/crucible_data/store/pm_factors/basics/date=* | wc -l | xargs echo "basics days now:"
