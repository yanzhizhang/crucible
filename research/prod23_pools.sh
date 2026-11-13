#!/usr/bin/env bash
# Round 46, read-only on 2.23, prints only: which stock pools PM's production keeps (HS300 / ZZ500 / ZZ1000 ...), and whether
# its mp/<date>/ files are the same zstd + msgpack layout decode_pm_mp.py already reads from 97's /data/prod/mp.
# Nothing is copied or written on 2.23; the report is tee'd on the machine this is started from.
#   bash research/prod23_pools.sh 2>&1 | tee data/reports/prod23_pools.txt
set -u
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
ssh -p $P23 -o BatchMode=yes $H23 '
P=/data/share/data/prod
echo "== 1 top of $P, $P/h, $P/models (names only)"
for d in $P $P/h $P/models $P/models/*; do [ -d $d ] && echo "-- $d: $(ls $d | head -40 | tr "\n" " ")"; done
echo "== 2 per pool under $P/h: sub-directories, files, first / last name, newest mtime"
for pool in $P/h/*/; do
  for d in $(find $pool -mindepth 1 -maxdepth 1 -type d | sort | head -40); do
    n=$(find -L $d -maxdepth 1 -type f 2>/dev/null | wc -l)
    f=$(ls $d | head -1); l=$(ls $d | tail -1); t=$(find -L $d -maxdepth 1 -printf "%TY-%Tm-%Td %TH:%TM\n" 2>/dev/null | sort | tail -1)
    echo "  $d: $n files, $f .. $l, newest $t"
  done
done
echo "== 3 anything named after another pool (depth 5)"
find -L $P -maxdepth 5 \( -iname "*500*" -o -iname "*1000*" -o -iname "*zz*" -o -iname "*csi*" -o -iname "*2000*" \) -printf "%y %TY-%Tm-%Td %s %p\n" 2>/dev/null | sort -k4 | head -60
echo "== 4 mp directories"
for m in $(find -L $P /data/share -maxdepth 3 -type d -name mp 2>/dev/null | sort -u); do
  n=$(ls $m | wc -l); echo "-- $m: $n entries, $(ls $m | head -1) .. $(ls $m | tail -1)"
  last=$m/$(ls $m | tail -1)
  ls -l --time-style=+%Y-%m-%dT%H:%M $last | awk "{print \"   \", \$5, \$6, \$7}" | head -40
  echo "   total: $(du -sh $last 2>/dev/null | cut -f1)"
done
echo "== 5 head of the smallest snap / order / trade / index file of the newest mp day (first 160 decoded bytes)"
M=$(find -L $P /data/share -maxdepth 3 -type d -name mp 2>/dev/null | sort -u | head -1)
L=$M/$(ls $M | tail -1)
command -v zstd >/dev/null || echo "   no zstd binary on this machine"
for k in index snap order trade; do
  f=$(ls -S $L/*${k}* 2>/dev/null | tail -1)
  [ -n "$f" ] || { echo "-- no *${k}* file in $L"; continue; }
  echo "-- $f"
  head -c 8 $f | od -A n -t x1
  command -v zstd >/dev/null && zstd -dc $f 2>/dev/null | head -c 160 | od -A d -t x1 | head -11
done
'
echo "== 6 the same on this machine, for comparison (/data/prod/mp)"
if [ -d /data/prod/mp ]; then
  L=/data/prod/mp/$(ls /data/prod/mp | tail -1)
  echo "-- /data/prod/mp: $(ls /data/prod/mp | wc -l) entries, $(ls /data/prod/mp | head -1) .. $(ls /data/prod/mp | tail -1)"
  ls $L | tr "\n" " "; echo
  for k in index snap order trade; do
    f=$(ls -S $L/*${k}* 2>/dev/null | tail -1)
    [ -n "$f" ] || continue
    echo "-- $f"
    head -c 8 $f | od -A n -t x1
    command -v zstd >/dev/null && zstd -dc $f 2>/dev/null | head -c 160 | od -A d -t x1 | head -11
  done
else
  echo "   no /data/prod/mp here"
fi
