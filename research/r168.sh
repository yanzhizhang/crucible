#!/usr/bin/env bash
# Round 168 on 97, foreground, read-only: first look at the 224 pull of a live day (/work/crucible_data/pm_224/<day>/, copied
# by the supervisor with the user's approval). Inventory only: what is there, which strategies ran, what the logs contain.
# The *.json5 files (accounts / logins) are NOT opened; lines containing passw / secret / token are dropped from every print.
#   bash research/r168.sh [day] 2>&1 | tee data/reports/r168.txt
set -u
D=${1:-20261118}
R=/work/crucible_data/pm_224/$D
safe() { grep -a -v -i -E "passw|secret|token" | cut -c1-240; }
echo "== tree (size, time, path)"
find $R -type f -printf '%10s  %TY-%Tm-%Td %TH:%TM  %P\n' | sort -k4 | safe
echo "== config2_HS300.json (PM's strategy config, no logins expected)"
safe < $R/config2_HS300.json
echo "== run_s_hs300.csv: $(wc -l < $R/run_s_hs300.csv) lines; head"
head -5 $R/run_s_hs300.csv | safe
echo "== sod.csv: $(wc -l < $R/sod.csv) lines; header + 2 rows"
head -3 $R/sod.csv | safe
for sub in pnl qx_trade_info position_log ledger portfolio cron; do
  echo "== $sub"
  find $R/$sub -type f | sort | while read -r f; do
    echo "-- ${f#$R/}: $(wc -l < "$f") lines"
    head -3 "$f" | safe
  done
done
echo "== chrono_logs"
for f in $(find $R/chrono_logs -type f | sort); do
  echo "-- ${f#$R/}: $(wc -l < $f) lines, $(du -h $f | cut -f1); first / last line:"
  head -1 $f | safe; tail -1 $f | safe
  echo "   levels: $(grep -a -o -E "\[(trace|debug|info|warning|warn|error|critical)\]|\b(TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL)\b" $f | sort | uniq -c | sort -rn | head -8 | tr '\n' ' ')"
  echo "   trk kinds: $(grep -a -o -E "trk [A-Z_]+" $f | sort | uniq -c | sort -rn | tr '\n' ' ')"
  echo "   source files (top 25 of 'name.cpp:line' / 'name.hpp:line'):"
  grep -a -o -E "[A-Za-z_0-9]+\.(cpp|hpp|h|cc):[0-9]+" $f | sort | uniq -c | sort -rn | head -25 | sed 's/^/     /'
  echo "   strategy plugins / models named:"
  grep -a -i -E "plugin|dlopen|\.so\b|model_s[0-9]|tk5m|livev1|/lgb/|models/|runS|run_s|catchRun" $f | grep -a -i -E "load|open|path|plugin|model|init|start" | safe | head -40 | sed 's/^/     /'
  echo "   decision lines (vpen / poscst): $(grep -a -c -E "vpen|poscst" $f); 3 samples:"
  grep -a -E "vpen|poscst" $f | head -3 | safe | sed 's/^/     /'
  echo "   first 5 WARN-or-worse kinds (message text with digits squeezed):"
  grep -a -E "warn|WARN|error|ERROR|critical|CRITICAL" $f | sed -E 's/^[^]]*\]//; s/[0-9]+/#/g' | cut -c1-140 | sort | uniq -c | sort -rn | head -12 | safe | sed 's/^/     /'
done
echo "[$(date +%T)] r168 done"
