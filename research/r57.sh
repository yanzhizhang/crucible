#!/usr/bin/env bash
# Round 57, started on 97, foreground. 2.23 is only read; the one write is under /work on 97.
#   1 the layout of 2.23:/data/share/s and /data/share/o (what 224 keeps under /tmp/s and /tmp/o): names, counts, sizes,
#     newest file per directory, three levels deep. Names only, no file content.
#   2 does 2.23 keep hstats of 20260430 (and which days at all)
#   3 if it does: copy HS300 hstats of 20260430 to /work/crucible_data/hstats_0430 (the user's go-ahead: copy it down)
#   bash research/r57.sh 2>&1 | tee data/reports/r57.txt
set -u
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH="ssh -p $P23 -o BatchMode=yes"
DST=/work/crucible_data/hstats_0430/HS300
$SSH $H23 '
for top in /data/share/s /data/share/o; do
  echo "== 1 $top"
  [ -e $top ] || { echo "   not there"; continue; }
  ls -ld $top | cut -c1-120
  find -L $top -maxdepth 3 -type d 2>/dev/null | sort | head -150 | while read d; do
    n=$(find -L "$d" -maxdepth 1 -type f 2>/dev/null | wc -l)
    s=$(ls "$d" 2>/dev/null | wc -l)
    t=$(find -L "$d" -maxdepth 1 -type f -printf "%TY-%Tm-%Td %TH:%TM %f\n" 2>/dev/null | sort | tail -1)
    echo "  $d: $s entries, $n files, first $(ls "$d" | head -1), last $(ls "$d" | tail -1), newest: $t"
  done
  echo "  -- file name patterns (digits -> N), with counts and total size:"
  find -L $top -maxdepth 4 -type f -printf "%s %P\n" 2>/dev/null | sed -E "s/[0-9]{6,8}/N/g" | awk "{c[\$2]++; b[\$2]+=\$1} END {for (k in c) printf \"   %6d files %12d bytes  %s\n\", c[k], b[k], k}" | sort -k5 | head -80
done
P=/data/share/data/prod/hstats/HS300
echo "== 2 $P"
for f in $(ls $P 2>/dev/null); do
  echo "  $f: $(ls $P/$f | wc -l) days, $(ls $P/$f | head -1) .. $(ls $P/$f | tail -1); 20260430: $(ls $P/$f/20260430 2>/dev/null | tr "\n" " ")"
done
echo "  other places with a 20260430 hstats directory:"
find -L /data/share -maxdepth 7 -type d -name 20260430 -path "*hstats*" 2>/dev/null | head -20 | sed "s/^/   /"
'
echo "== 3 copy hstats of 20260430 to $DST"
if $SSH $H23 'ls /data/share/data/prod/hstats/HS300/*/20260430/samplerS.nc >/dev/null 2>&1'; then
  mkdir -p $DST
  rsync -aL -e "$SSH" --include="*/" --include="20260430/**" --exclude="*" $H23:/data/share/data/prod/hstats/HS300/ $DST/ 2>&1 | tail -3
  find $DST -type d -empty -delete 2>/dev/null
  for f in $(ls $DST); do echo "  $f/20260430: $(ls -l $DST/$f/20260430 2>/dev/null | awk 'NR>1 {printf "%s (%s) ", $9, $5}')"; done
  du -sh $DST | sed 's/^/  total /'
else
  echo "  2.23 has no samplerS.nc of 20260430 under prod/hstats/HS300: nothing copied"
fi
