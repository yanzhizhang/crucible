#!/usr/bin/env bash
# Round 97 on 97, foreground, read-only (deletes nothing, writes only data/reports). A few minutes.
# 1 which days exist where: DataYesSharesL2 packages, dysl2_raw, feitu_raw, 224 logs.
# 2 space and inodes of every local intermediate (for the cleanup list).
# 3 dy_exact_probe on the last two DataYes days: how big index volume / amount get, and whether dysl2_raw kept them.
#   bash research/r97.sh 2>&1 | tee data/reports/r97.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
DY=/mnt/TickDataRepo/DataYesSharesL2
say() { echo "[$(date +%T)] $*"; }

say "1 days"
echo "  DataYesSharesL2 (last 30): $(ls $DY | grep -E '^[0-9]{8}(\.zip)?$' | tail -30 | tr '\n' ' ')"
echo "  ReadyFlag (last 10): $(ls $DY/ReadyFlag 2>/dev/null | tail -10 | tr '\n' ' ')"
for s in dysl2_raw feitu_raw; do
  for k in quotation order transaction index; do
    echo "  $s/$k: $(ls $DATA/$s/kind=$k 2>/dev/null | sed 's/date=//' | tr '\n' ' ')"
  done
done
echo "  224 log days: $(ls $DATA/chrono_logs 2>/dev/null | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}' | sort -u | tr '\n' ' ')"

say "2 space and inodes"
df -h /work; df -i /work
for top in $DATA /work/crucible/data; do
  echo "  -- $top (size, inodes, path)"
  for d in $top/*; do
    printf "  %8s %10s  %s\n" "$(du -sh "$d" 2>/dev/null | cut -f1)" "$(du -s --inodes "$d" 2>/dev/null | cut -f1)" "$d"
  done | sort -k2 -n | tail -60
done

say "3 dy_exact_probe"
LAST=$(ls $DY | grep -E '^[0-9]{8}(\.zip)?$' | cut -c1-8 | sort -u | tail -2 | paste -sd,)
python -u research/dy_exact_probe.py --dates $LAST 2>&1 | cut -c1-230
say "done"
