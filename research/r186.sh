#!/usr/bin/env bash
# Round 186 on 97, foreground, read-only, light (~1 min): the 140 orders chrono rejected on 20261118 -- does PM's library keep
# trying to cancel them (decoded: a rejected order is not erased, so the cancel sweep retries it with a back-off), and what
# does chrono log for those attempts. Parsed tables first, then a grep of the raw chrono log for the rejected idx.
#   bash research/r186.sh 2>&1 | tee data/reports/r186.txt
set -u
cd /work/crucible
D=20261118
O=data/reports/r186; mkdir -p $O
L=/work/crucible_data/pm_224/$D/chrono_logs/chrono_${D}_092335.log
python -u research/pm_factors/runs_rej.py --orders /work/crucible_data/pm_orders --date $D --out $O 2>&1 | grep -v -E "Warning|warn" | cut -c1-300
echo "== raw log: lines mentioning a cancel for the rejected idx (chrono's own lines carry idx=<n>)"
PAT=$(sed 's/^/idx=/; s/$/\\b/' $O/rej_idx.txt | paste -sd'|' -)
grep -a -E "$PAT" $L | grep -a -i -E "cxl|cancel" | sed -E 's/^[^]]*\] *//; s/[0-9]+/#/g' | cut -c1-150 | sort | uniq -c | sort -rn | head -12
echo "== raw log: PM's own cancel lines of the whole day, by text (digits squeezed)"
grep -a -i -E "cxl ords|cancelExOrd|OnFCM|OnFOM" $L | sed -E 's/^[^]]*\] *//; s/^\[[^]]*\] *//; s/-?[0-9]+(\.[0-9]+)?/#/g' | cut -c1-150 | sort | uniq -c | sort -rn | head -12
echo "== raw log: chrono WARNING-or-worse lines about cancels, by text"
grep -a -E "WARN|ERROR|CRITICAL" $L | grep -a -i -E "cancel|cxl" | sed -E 's/^[^]]*\] *//; s/-?[0-9]+(\.[0-9]+)?/#/g' | cut -c1-170 | sort | uniq -c | sort -rn | head -10
echo "== three rejected orders: every raw line with their idx (first 8 lines each)"
for i in $(head -3 $O/rej_idx.txt); do echo "-- idx $i"; grep -a -E "idx=$i\b" $L | cut -c1-260 | head -8; done
echo "[$(date +%T)] r186 done"
