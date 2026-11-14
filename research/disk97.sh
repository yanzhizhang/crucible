#!/usr/bin/env bash
# Read-only disk survey of 97 for the user's cleanup: what takes the space under /work (and the filesystems around it).
# Deletes nothing; prints sizes, biggest files, and which crucible outputs are scratch from past rounds.
#   bash research/disk97.sh 2>&1 | tee data/reports/disk97.txt
set -u
say() { echo "[$(date +%T)] $*"; }
say "filesystems"
df -h / /work /data /tmp 2>/dev/null | awk '!seen[$0]++'
say "/work, two levels (largest 40)"
du -xh --max-depth=2 /work 2>/dev/null | sort -h | tail -40
say "/work/crucible/data, one level"
du -xh --max-depth=1 /work/crucible/data 2>/dev/null | sort -h | tail -60
say "/work/crucible_data, one level"
du -xh --max-depth=1 /work/crucible_data 2>/dev/null | sort -h | tail -60
say "files over 2 GB under /work (size, modified, path)"
find /work -xdev -type f -size +2G -printf '%s %TY-%Tm-%Td %p\n' 2>/dev/null | sort -rn | head -60 | awk '{printf "%7.1f GB  %s  %s\n", $1/2^30, $2, $3}'
say "round outputs under /work/crucible/data (data/<name> written by r<NN>.sh), with last change"
for d in /work/crucible/data/*/; do printf "%8s  %s  %s\n" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$(date -r "$d" +%F)" "$d"; done | sort -h | tail -60
say "conda pkgs cache and pip/uv caches"
du -sh /opt/anaconda3/pkgs ~/.cache/pip ~/.cache/uv 2>/dev/null
say "done"
