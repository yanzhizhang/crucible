#!/usr/bin/env bash
# Read-only inventory of PM's strategy binaries on 97 (libcatchRun.so, libshtst.a; symbols + DWARF present, user 2026-11-16).
# Strategy side only (runS: prediction -> orders); the model factors are not in them. Stage 1 = what is there: sections,
# source files, functions of the strategy sources, log format strings, tools available. Everything is written under
# OUT on 97 (default /work/crucible_data/so_probe); only the short summary at the end is meant to be relayed. Nothing
# is installed, executed from the libraries, or copied anywhere else.
#   bash research/so_probe.sh /path/to/libcatchRun.so /path/to/libshtst.a 2>&1 | tee /work/crucible_data/so_probe/stage1.txt
set -u
SO=${1:?path of libcatchRun.so}
AR=${2:-}
OUT=${OUT:-/work/crucible_data/so_probe}
mkdir -p $OUT
say() { echo "== $*"; }
say "tools"; for t in gdb readelf objdump nm strings addr2line pahole c++filt ar; do printf "  %-10s %s\n" $t "$(command -v $t || echo MISSING)"; done
for F in $SO $AR; do
  [ -e "$F" ] || { echo "missing $F"; continue; }
  B=$(basename $F)
  say "$B"; ls -l $F | awk '{print "  size", $5}'; file $F | cut -c1-200
  readelf -S $F 2>/dev/null | grep -E "debug_|symtab|\.rodata|\.text" | awk '{print "  ", $0}' | head -20
  case $F in *.a) ar t $F > $OUT/$B.members.txt; echo "  members: $(wc -l < $OUT/$B.members.txt)  $(head -30 $OUT/$B.members.txt | tr '\n' ' ')" ;; esac
  readelf --debug-dump=decodedline $F 2>/dev/null | grep -oE '[^ ]+\.(cpp|cc|c|h|hpp)' | sort | uniq -c | sort -rn > $OUT/$B.sources.txt
  echo "  source files in line table: $(wc -l < $OUT/$B.sources.txt); top:"; head -25 $OUT/$B.sources.txt | sed 's/^/    /'
  readelf --debug-dump=info $F 2>/dev/null | grep -E "DW_AT_comp_dir|DW_AT_producer" | sed 's/.*: //' | sort | uniq -c | sort -rn | head -6 | sed 's/^/  /'
  nm -C --defined-only $F 2>/dev/null | grep -E " [TtWw] " > $OUT/$B.functions.txt
  echo "  defined functions: $(wc -l < $OUT/$B.functions.txt)"
  grep -iE "alg|runS|catchRun|\bes::|order|cancel|vpen|lthr|winsd|executp|emsMkt|thr" $OUT/$B.functions.txt | awk '{$1=""; $2=""; print "   ", $0}' | sort -u | head -80
  strings -n 4 $F > $OUT/$B.strings.txt
  echo "  log / config strings:"
  grep -nE "fa=|hs=|lthr|winsdQty|asf|emsMktVol|emsOurVol|executp|vpen|asq|liid|place_order|order is nullptr|config|\.json" $OUT/$B.strings.txt | cut -c1-200 | head -60 | sed 's/^/    /'
done
say "stage 1 done; full lists in $OUT (members / sources / functions / strings)"
