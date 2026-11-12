#!/usr/bin/env bash
# Bring one new 224 day (DataYes) into the scoring set and test R17 / R24 / R24 gated on it, on 97, foreground.
# Reads 224 and 2.23 only (rsync / ssh pulls); every write is under /work.
#   0 daily_pm.sh D: 224 logs -> pm_signals, DataYesSharesL2 -> dysl2_raw + catalog_dysl2, decisions / orders
#   1 2.23: hstats of D into the view, stsfile (/data/share/s) into pm_stsfile, PM's .jz of D-1 and D into seed_probe/HS300
#   2 target link pm_target_r22/date=D -> pm_signals
#   3 whole-book factors (full_book, dysl2) and engine state at the marks, seeded with PM's D-1 .jz
#   4 R17 / R27 both (the baseline) / R24 / R24 gated scored on D
#   5 our engine's day-end state against PM's D.jz (dcp_build), per exchange
#   bash research/add_day.sh 20261111 2>&1 | tee data/reports/add_20261111.txt
set -u
D=${1:?trading day YYYYMMDD}
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBD=$DATA/store/pm_factors/full_book_dysl2
SEED=$DATA/seed_probe/HS300
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH="ssh -p $P23 -o BatchMode=yes"
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"

say "0 daily_pm $D"
bash research/daily_pm.sh $D

say "1 2.23: hstats $D, stsfile, PM state files"
mkdir -p $DATA/hstats_extra/HS300 $DATA/pm_stsfile $SEED
rsync -aLm -e "$SSH" --include="*/" --include="$D/***" --exclude="*" $H23:/data/share/data/prod/hstats/HS300/ $DATA/hstats_extra/HS300/ \
  > $R/add_${D}_rsync.txt 2>&1 || say "  hstats rsync failed: $(tail -1 $R/add_${D}_rsync.txt)"
for fam in $(ls $DATA/hstats_extra/HS300); do mkdir -p $HV/$fam; for x in $DATA/hstats_extra/HS300/$fam/*; do [ -e $HV/$fam/$(basename $x) ] || ln -sfn $x $HV/$fam/; done; done
echo "  $D: $(ls -d $HV/*/$D/samplerS.nc 2>/dev/null | wc -l) families with samplerS.nc"
rsync -aL -e "$SSH" $H23:/data/share/s/ $DATA/pm_stsfile/ >> $R/add_${D}_rsync.txt 2>&1 || say "  stsfile rsync failed: $(tail -1 $R/add_${D}_rsync.txt)"
ls $DATA/pm_stsfile/stats/sod $DATA/pm_stsfile/index 2>/dev/null | tail -3 | sed 's/^/  stsfile /'
rsync -aL -e "$SSH" --include="*.jz" --exclude="*" $H23:/data/share/data/prod/h/HS300/p1_intrading_dcp/ $SEED/ >> $R/add_${D}_rsync.txt 2>&1 \
  || say "  .jz rsync failed: $(tail -1 $R/add_${D}_rsync.txt)"
P=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
echo "  PM state: D-1 = ${P:-none} $([ -e $SEED/$D.jz ] && echo ", $D.jz present" || echo ", no $D.jz yet")"
[ -n "$P" ] || { say "no PM state before $D: stop"; exit 1; }

say "2 target"
mkdir -p $TGT
[ -e $DATA/pm_signals/date=$D.parquet ] && ln -sf $DATA/pm_signals/date=$D.parquet $TGT/ || { say "no pm_signals for $D: stop"; exit 1; }
python -c "import polars as pl; t=pl.read_parquet('$TGT/date=$D.parquet'); print('  target rows', t.height, 'stocks', t['sid'].n_unique() if 'sid' in t.columns else '?')"

say "3 full_book + engine state for $D"
Q=$(CRUCIBLE_SOURCE=dysl2 python -c "import sys; sys.path.insert(0,'research/pm_factors'); import duckdb, dcp_build as B; c=duckdb.connect(str(B.CATALOG), read_only=True); print(c.sql('SELECT count(*) FROM raw_quotation WHERE date = $D').fetchone()[0])" 2>/dev/null)
echo "  catalog_dysl2 quotes on $D: ${Q:-?}"
[ "${Q:-0}" -gt 0 ] || { say "no DataYes ticks for $D in catalog_dysl2 (package not on 97 yet?): rerun later"; exit 1; }
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/full_book.py --dates $D --marks rand:240 --workers 1 > $R/add_${D}_fullbook.txt 2>&1 \
  || say "  full_book FAILED: $(tail -3 $R/add_${D}_fullbook.txt)"
ls -la $FBD/date=$D.parquet 2>&1 | sed 's/^/  /'
python -u research/engine/run_cpp.py --date $D --universe pm:$SEED/$P.jz --prev $SEED/$P.jz --catalog $DATA/catalog_dysl2.duckdb \
  --marks rand:240 --threads 32 > $R/add_${D}_engine.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/add_${D}_engine.txt)"
ls -la data/engine/marks_$D.parquet 2>&1 | sed 's/^/  /'

say "4 R17 / R24 / R24 gated on $D"
for tag in r17 r27both r24 r24g; do
  case $tag in r17) A=$R/stage6_r17/ic_assign.json ;; r27both) A=$R/r27_both.json ;; r24) A=$R/stage6_r24/ic_assign.json ;; r24g) A=$R/stage6_r24_gate/ic_assign_gated.json ;; esac
  [ -e $A ] || { echo "  $tag: no $A"; continue; }
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $D --assign $A $SCORE --full-book $FBD $M \
    --out $R/stage6_add_${D}_$tag > $R/add_${D}_$tag.txt 2>&1
  say "  $tag ($(python -c "import json; print(len(json.load(open('$A'))))") columns): $D $(ic $R/add_${D}_$tag.txt)"
done
grep -hE "Error|Traceback|model generation" $R/add_${D}_r*.txt | sort -u | head

say "5 day-end state against PM's $D.jz"
if [ -e $SEED/$D.jz ]; then
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/dcp_build.py --dates $D --dcp-dir $SEED > $R/add_${D}_dcp.txt 2>&1 \
    || say "  dcp_build FAILED: $(tail -3 $R/add_${D}_dcp.txt)"
  tail -40 $R/add_${D}_dcp.txt | cut -c1-200
else
  say "  no PM $D.jz on 2.23 yet (it lands ~19:30-21:00): rerun step 5 later"
fi
say "done $D"
