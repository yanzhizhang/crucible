#!/usr/bin/env bash
# Round 150 on 97, foreground, read-only. Two independent parts (user, 2026-11-18: probe how the libraries use the alg
# config, and check vr / poscst against the logs).
#  inv <dir>  inventory of a directory of PM libraries ON 97 (default /data/sht-stg-lib; pass the 97 copy of 224's
#             /tmp/zyq/sht-stg-lib once the user has copied it). Per *.so / *.a / *.json: size, mtime, md5, whether the same
#             basename under /data/sht-stg-lib has the same md5; sections (debug info / symbol table); for files with debug
#             info, the structures that hold the alg-config member names with offsets; whether the config key strings and
#             the decision-log format string are present. Nothing is executed from the libraries, nothing is copied, lists
#             stay under /work/crucible_data/so_probe. This part does NOT touch 224.
#  logs       runs_cfg: the decision log's vpen / vr / poscst / hs / lthr against products of the config values with cpos, mp,
#             winsdQty, asq, per time segment (before 13:00 = rvts, 13:00-14:45, after). ~1 min.
#  sts        which model did 224 use in November: the 97 copy of PM's stsfile tree (/work/crucible_data/pm_stsfile, synced
#             from 2.23:/data/share/s) -- top-level layout, the newest files, and every text / json file naming a model
#             (model_s / tk5m / lgb / livev1 / models). Listing and grep only, on the 97 copy.
#   bash research/r150.sh inv [dir] 2>&1 | tee data/reports/r150_inv.txt
#   bash research/r150.sh logs      2>&1 | tee data/reports/r150_logs.txt
#   bash research/r150.sh sts       2>&1 | tee data/reports/r150_sts.txt
set -u
cd /work/crucible
KEYS=tqub,nmvcst,tpub,adur,vpen,rvts,asf,neg_nmvcst,ahubs
case "${1:?inv|logs|sts}" in
inv)
  DIR=${2:-/data/sht-stg-lib}
  REF=/data/sht-stg-lib
  OUT=/work/crucible_data/so_probe/r150
  mkdir -p $OUT
  echo "== $DIR"
  find $DIR -type f \( -name '*.so' -o -name '*.so.*' -o -name '*.a' -o -name '*.json' \) 2>/dev/null | sort | while read -r F; do
    B=$(basename $F)
    M=$(md5sum $F | cut -c1-12)
    R=$(find $REF -type f -name "$B" 2>/dev/null | head -1)
    SAME="no file of that name under $REF"
    [ -n "$R" ] && { [ "$(md5sum $R | cut -c1-12)" = "$M" ] && SAME="same as $R" || SAME="DIFFERS from $R"; }
    [ "$DIR" = "$REF" ] && SAME=""
    echo "-- ${F#$DIR/}  $(stat -c '%s bytes  %y' $F | cut -c1-40)  md5 $M  $SAME"
    case $B in
    *.json) ;;
    *)
      SEC=$(readelf -S $F 2>/dev/null | grep -oE "\.debug_info|\.symtab|\.debug_line" | sort | uniq -c | tr '\n' ' ')
      echo "   sections: ${SEC:-none}"
      strings -n 4 $F > $OUT/$B.strings.txt
      echo "   config keys present as strings: $(for k in $(echo $KEYS | tr ',' ' '); do grep -qx "$k" $OUT/$B.strings.txt && printf '%s ' $k; done)"
      echo "   decision-log format: $(grep -m1 -E 'vpen.*poscst|poscst.*cpos' $OUT/$B.strings.txt | cut -c1-160)"
      if readelf -S $F 2>/dev/null | grep -q "\.debug_info"; then
        readelf --debug-dump=info $F 2>/dev/null | python research/dwarf_members.py --names $KEYS --max 12 | cut -c1-160
      fi ;;
    esac
  done
  echo "(string lists in $OUT)" ;;
logs)
  DAYS=$(ls -d /work/crucible_data/pm_orders/date=* 2>/dev/null | sed 's/.*date=//' | tail -3 | paste -sd, -)
  echo "== decision days: $DAYS"
  python -u research/pm_factors/runs_cfg.py --orders /work/crucible_data/pm_orders --dates "$DAYS" 2>&1 | grep -v -E "Warning|warn" | cut -c1-220 ;;
sts)
  S=/work/crucible_data/pm_stsfile
  echo "== $S: layout (2 levels, file counts)"
  find $S -maxdepth 2 -type d 2>/dev/null | sort | while read -r d; do echo "   $d  ($(find "$d" -maxdepth 1 -type f | wc -l) files)"; done | head -60
  echo "== newest 25 files"
  find $S -type f -printf '%TY-%Tm-%Td %TH:%TM  %s  %p
' 2>/dev/null | sort -r | head -25 | cut -c1-200
  echo "== small text files naming a model (first 3 matching lines each, 40 files at most)"
  find $S -type f -size -2000k \( -name '*.json' -o -name '*.json5' -o -name '*.cfg' -o -name '*.conf' -o -name '*.ini' -o -name '*.txt' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' \) 2>/dev/null     | xargs -r grep -lE "model_s[0-9]|tk5m|livev1|/lgb/|models/" 2>/dev/null | sort | head -40 | while read -r f; do
      echo "-- $f"; grep -nE "model_s[0-9]|tk5m|livev1|/lgb/|models/" "$f" | head -3 | cut -c1-220 | sed 's/^/     /'; done
  echo "== file names with a date of November (last 20)"
  find $S -type f -name '*202611*' 2>/dev/null | sort | tail -20 | cut -c1-200 ;;
esac
echo "[$(date +%T)] r150 $1 done"
