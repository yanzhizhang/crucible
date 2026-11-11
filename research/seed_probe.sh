#!/usr/bin/env bash
# Where PM's day-end state chain (.jz seeds) starts, on 97 and on 2.23. Read-only on both; copies go to /work.
#   1 97: /data/prod/h/HS300/p1_intrading_dcp read in place
#   2 2.23: listing (links, mtimes, sizes) of the dcp dirs + /data/share/s, md5 of every .jz there
#   3 2.23: copy the .jz files to /work (only when under 5 GB) and read them like 1
#   bash research/seed_probe.sh 2>&1 | tee data/reports/seed_probe.txt
set -u
cd /work/crucible
R=data/reports
D=/work/crucible_data/seed_probe
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
S="ssh -p $P23 -o BatchMode=yes $H23"
say() { echo "[$(date +%T)] $*"; }
mkdir -p $D $R

say "1 97: /data/prod/h/HS300/p1_intrading_dcp"
ls -la --time-style=full-iso /data/prod/h/HS300/p1_intrading_dcp | head -5
ls -la --time-style=full-iso /data/prod/h/HS300/p1_intrading_dcp | tail -5
python -u research/pm_factors/seed_probe.py --dir /data/prod/h/HS300/p1_intrading_dcp --out $R/seed_probe_97.parquet

say "2 2.23: where the dcp dirs are, links and mtimes"
$S 'ls -d /data/share/data/prod/h/*/* 2>/dev/null | head -40; find /data/share/data/prod/h /data/share/s -maxdepth 4 -type d -name "*dcp*" 2>/dev/null | head -20' \
  > $R/seed_probe_23_dirs.txt 2>&1 || say "  ssh failed: $(tail -1 $R/seed_probe_23_dirs.txt)"
cat $R/seed_probe_23_dirs.txt
DIR23=${DIR23:-/data/share/data/prod/h/HS300/p1_intrading_dcp}
$S "cd $DIR23 && find . -maxdepth 1 -printf '%y %TY-%Tm-%Td %TH:%TM %s %f -> %l\n' | sort -k5; echo; du -shL .; md5sum -- *.jz 2>/dev/null" \
  > $R/seed_probe_23_list.txt 2>&1 || say "  listing $DIR23 failed: $(tail -1 $R/seed_probe_23_list.txt)"
echo "  $(grep -c '\.jz' $R/seed_probe_23_list.txt) .jz lines; links:"
grep -E '^l ' $R/seed_probe_23_list.txt | head -40
echo "  first / last:"; grep -E '^[fl] ' $R/seed_probe_23_list.txt | head -5; grep -E '^[fl] ' $R/seed_probe_23_list.txt | tail -5
echo "  md5 shared by more than one date:"
grep -E '^[0-9a-f]{32}  ' $R/seed_probe_23_list.txt | awk '{n[$1]++; d[$1]=d[$1]" "$2} END {for (m in n) if (n[m] > 1) print "   ", m, d[m]}' | head -40
$S 'ls -la --time-style=full-iso /data/share/s | head -20; ls -la --time-style=full-iso /data/share/s | tail -5' > $R/seed_probe_23_s.txt 2>&1
cat $R/seed_probe_23_s.txt

say "3 2.23: copy and read the .jz files"
GB=$($S "du -sLk $DIR23 2>/dev/null | cut -f1" 2>/dev/null)
echo "  $DIR23: ${GB:-?} KB"
if [ -n "${GB:-}" ] && [ "$GB" -lt 5000000 ]; then
  mkdir -p $D/HS300
  rsync -aL -e "ssh -p $P23 -o BatchMode=yes" --include="*.jz" --exclude="*" $H23:$DIR23/ $D/HS300/ > $R/seed_probe_rsync.txt 2>&1 \
    || say "  rsync failed: $(tail -1 $R/seed_probe_rsync.txt)"
  python -u research/pm_factors/seed_probe.py --dir $D/HS300 --out $R/seed_probe_23.parquet
else
  say "  skipped the copy (size unknown or over 5 GB); the listing and md5s above still tell links and duplicates"
fi
say "done"
