#!/usr/bin/env bash
# Round 133 on 97, foreground, read-only: so_probe stage 2 (order rule out of libcatchRun.so via objdump / gdb -batch,
# nothing executed) + PM's strategy config next to it. Writes only /work/crucible_data/so_probe/stage2. ~2-5 min.
#   bash research/r133.sh 2>&1 | tee /work/crucible_data/so_probe/stage2.txt
set -u
cd /work/crucible
D=/data/sht-stg-lib
echo "== config2_HS300.json"; cat $D/config2_HS300.json; echo
echo "== chrono_config.json5: strategy entry only"; grep -n -B2 -A25 -iE "runS|catchRun" $D/chrono_config.json5 | grep -viE "password|passwd|key\s*:|secret|token" | head -60
echo "== describe<runS> type name"; strings -n 3 $D/plugins/libcatchRun.so | grep -nE "^runS|^catchRun|^cts_runS|runS_a" | head
python3 -u research/so_probe2.py $D/plugins/libcatchRun.so --out /work/crucible_data/so_probe/stage2
echo "[$(date +%T)] done"
