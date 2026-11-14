#!/usr/bin/env bash
# Round 57b, started on 97, foreground. 2.23 is only read; the one write is under /work on 97.
# R57: 2.23 has no 20260430 under prod/hstats/HS300, but prod/hstats/sod/HS300/<family>/20260430 exists for 15 families
# ("sod" = the files as loaded at the start of the day?). Here:
#   1 what prod/hstats/sod holds: pools, families, days per family, the files of 20260430 with sizes, and for one family
#     whether sod/<D> is byte-equal to HS300/<D> or to HS300/<day before D> (md5 on 2.23, read-only)
#   2 copy sod/HS300/*/20260430 to /work/crucible_data/hstats_0430/HS300 (the user's go-ahead for the 0430 hstats copy)
#   bash research/r57b.sh 2>&1 | tee data/reports/r57b.txt
set -u
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH="ssh -p $P23 -o BatchMode=yes"
DST=/work/crucible_data/hstats_0430/HS300
$SSH $H23 '
H=/data/share/data/prod/hstats
echo "== 1 $H: $(ls $H | tr "\n" " ")"
echo "   $H/sod: $(ls $H/sod | tr "\n" " ")"
S=$H/sod/HS300
for f in $(ls $S); do
  echo "  $f: $(ls $S/$f | wc -l) days, $(ls $S/$f | head -1) .. $(ls $S/$f | tail -1); 20260430: $(ls -l $S/$f/20260430 2>/dev/null | awk "NR>1 {printf \"%s (%s) \", \$9, \$5}")"
done
echo "   days of sod/HS300/ZZAL: $(ls $S/ZZAL | tr "\n" " " | cut -c1-600)"
echo "   is sod/<D> the same file as HS300/<D> or HS300/<day before>? (md5 of ZZAL samplerS.nc)"
for D in $(ls $S/ZZAL | tail -4); do
  P=$(ls $H/HS300/ZZAL | awk -v d=$D "\$1 < d" | tail -1)
  a=$(md5sum $S/ZZAL/$D/samplerS.nc 2>/dev/null | cut -c1-8); b=$(md5sum $H/HS300/ZZAL/$D/samplerS.nc 2>/dev/null | cut -c1-8); c=$(md5sum $H/HS300/ZZAL/$P/samplerS.nc 2>/dev/null | cut -c1-8)
  echo "     $D: sod $a, HS300/$D $b, HS300/$P $c"
done
echo "   HS300/ZZAL/20260429: $(ls -l $H/HS300/ZZAL/20260429 2>/dev/null | awk "NR>1 {printf \"%s (%s) \", \$9, \$5}"); HS300/ZZAL/20260430: $(ls -la $H/HS300/ZZAL/20260430 2>&1 | head -3 | tr "\n" " ")"
du -shL $S/*/20260430 2>/dev/null | awk "{print \"   \", \$1, \$2}" | head -20
'
echo "== 2 copy sod/HS300/*/20260430 to $DST"
if $SSH $H23 'ls /data/share/data/prod/hstats/sod/HS300/*/20260430/* >/dev/null 2>&1'; then
  mkdir -p $DST
  rsync -aL -e "$SSH" --include="*/" --include="20260430/**" --exclude="*" $H23:/data/share/data/prod/hstats/sod/HS300/ $DST/ 2>&1 | tail -3
  find $DST -type d -empty -delete 2>/dev/null
  for f in $(ls $DST); do echo "  $f/20260430: $(ls -l $DST/$f/20260430 2>/dev/null | awk 'NR>1 {printf "%s (%s) ", $9, $5}')"; done
  du -sh $DST | sed 's/^/  total /'
else
  echo "  no files under sod/HS300/*/20260430: nothing copied"
fi
