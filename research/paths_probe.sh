#!/usr/bin/env bash
# What lives under 224's /tmp/o and /tmp/s (PM's production layout; /tmp/s is another user's and not listable, but known
# sub-paths like /tmp/s/stsfile are readable) and 2.23's /data/share/o and /data/share/s, and how far they correspond.
# Read-only on both machines; prints only.
#   bash research/paths_probe.sh 2>&1 | tee data/reports/paths_probe.txt
set -u
H224=${H224:-zhangyanzhi@10.12.1.224}
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
echo "== 224: /tmp/o and /tmp/s, readable parts"
ssh -o BatchMode=yes $H224 '
for d in /tmp/o /tmp/s; do echo "-- $d"; ls -la $d 2>&1 | head -30; done
for d in /tmp/o/* /tmp/s/stsfile /tmp/s/stsfile/* /tmp/s/h /tmp/s/data /tmp/s/prod /tmp/s/dcp /tmp/s/models /tmp/o/h /tmp/o/prod; do
  [ -e "$d" ] || continue
  if [ -d "$d" ]; then n=$(ls "$d" 2>/dev/null | wc -l); echo "  $d/: $n entries, first $(ls "$d" 2>/dev/null | head -2 | tr "\n" " ") last $(ls "$d" 2>/dev/null | tail -2 | tr "\n" " ")";
  else ls -la "$d"; fi
done
echo "-- paths the plugin / chrono logs name under /tmp/o or /tmp/s (latest log)"
f=$(ls -t /tmp/zyq/sht-logs/*.log 2>/dev/null | head -1); [ -n "$f" ] && grep -oE "/tmp/[os]/[A-Za-z0-9_./{}-]+" "$f" | sort | uniq -c | sort -rn | head -40
'
echo
echo "== 2.23: /data/share/o and /data/share/s (depth 3)"
ssh -p $P23 -o BatchMode=yes $H23 '
for d in /data/share/o /data/share/s; do
  echo "-- $d"; ls -la --time-style=long-iso $d 2>&1 | head -30
  find $d -mindepth 1 -maxdepth 3 -type d 2>/dev/null | head -80 | while read x; do
    n=$(ls "$x" 2>/dev/null | wc -l); echo "  $x/: $n entries, first $(ls "$x" | head -1) last $(ls "$x" | tail -1)"; done
done
du -sh /data/share/o /data/share/s 2>/dev/null
'
