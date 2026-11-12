#!/usr/bin/env bash
# What PM's production on 2.23 writes (the only correct outputs: 224's live signals are all unreliable). Read-only on 2.23,
# prints only. Looks for anything that could serve as a target: per-day predictions, features, or state.
#   bash research/prod23_probe.sh 2>&1 | tee data/reports/prod23_probe.txt
set -u
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
ssh -p $P23 -o BatchMode=yes $H23 '
B=/data/share/data/prod/h/HS300
echo "== dirs under $B (depth 2): files, first / last name, newest mtime"
for d in $(find $B -mindepth 1 -maxdepth 2 -type d | sort); do
  n=$(find -L $d -maxdepth 1 -type f 2>/dev/null | wc -l)
  [ $n -gt 0 ] || { echo "  $d: no files"; continue; }
  f=$(ls $d | head -1); l=$(ls $d | tail -1); t=$(find -L $d -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM\n" 2>/dev/null | sort | tail -1)
  echo "  $d: $n files, $f .. $l, newest $t"
done
echo "== p1..p10 and livev1 in depth"
find -L $B/livev1 $B/p[0-9]* -maxdepth 4 -printf "%y %TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort -k5 | head -120
echo "== prediction / feature / alpha-like files anywhere under /data/share/data/prod (depth 7)"
find -L /data/share/data/prod -maxdepth 7 -type f \( -iname "*alpha*" -o -iname "*pred*" -o -iname "*feat*" -o -iname "*signal*" -o -iname "*fa_*" -o -iname "*factor*" \) \
  -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort -k4 | head -80
'
