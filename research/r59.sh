#!/usr/bin/env bash
# Round 59 on 97, foreground. 2.23 is only read; every write is under /work.
# The user: the mp files of 20261111 / 20261112 on 2.23 are DataYes ticks produced for PM, so they are most likely
# what PM's process computed its factors from. Pull them and compare tick by tick with the DataYes ticks on 97.
#   0 pull prod/mp/<D> (HS300 files, ~0.9 GB a day) from 2.23 into /work/crucible_data/pm_mp
#   1 decode (decode_pm_mp.py --check) into pmmp_raw, catalog_pmmp.duckdb          (CRUCIBLE_SOURCE=pmmp)
#   2 tick comparison mp against DataYes, restricted to the stocks of the mp files (r59_cmp.py): rows, stocks, keys only on
#     one side, equal price / volume / exchange time on matched keys, local arrival time difference
#   3 day-end state against PM's <D>.jz on the mp feed and on the DataYes feed (dcp_build)
#   bash research/r59.sh 2>&1 | tee data/reports/r59.txt        (days: DAYS="20261111 20261112")
set -u
DAYS=${DAYS:-"20261111 20261112"}
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH="ssh -p $P23 -o BatchMode=yes"
say() { echo "[$(date +%T)] $*"; }

for D in $DAYS; do
  say "0 pull mp/$D from 2.23"
  mkdir -p $DATA/pm_mp/$D
  $SSH $H23 "ls -l /data/share/data/prod/mp/$D/ | awk 'NR>1 {print \"   2.23:\", \$5, \$6, \$7, \$8, \$9}'" 2>&1 | head -30
  rsync -aL -e "$SSH" --include="*HS300*.mp" --exclude="*" $H23:/data/share/data/prod/mp/$D/ $DATA/pm_mp/$D/ > $R/r59_rsync_$D.txt 2>&1 || say "  rsync failed: $(tail -1 $R/r59_rsync_$D.txt)"
  ls -l $DATA/pm_mp/$D | awk 'NR>1 {print "   97:", $5, $9}'
  [ "$(ls $DATA/pm_mp/$D/*.mp 2>/dev/null | wc -l)" -ge 6 ] || { say "mp files of $D missing: skip the day"; continue; }
  say "1 decode $D (source pmmp)"
  python -u research/decode_pm_mp.py --root $DATA/pm_mp --date $D --out $DATA/pmmp_raw --workers 8 --check > $R/r59_decode_$D.txt 2>&1 || say "  decode FAILED"
  tail -25 $R/r59_decode_$D.txt | cut -c1-200
done
CRUCIBLE_SOURCE=pmmp python -u research/catalog.py > $R/r59_catalog.txt 2>&1 || say "  catalog FAILED: $(tail -2 $R/r59_catalog.txt)"

for D in $DAYS; do
  say "2 ticks of $D: mp against DataYes"
  python -u research/r59_cmp.py $D $DATA/catalog_pmmp.duckdb $DATA/catalog_dysl2.duckdb 2>&1 | cut -c1-230
done

for D in $DAYS; do
  [ -e $SEED/$D.jz ] || { say "3 no PM $D.jz in $SEED: skip"; continue; }
  say "3 day-end state against PM's $D.jz, mp feed"
  CRUCIBLE_SOURCE=pmmp python -u research/pm_factors/dcp_build.py --dates $D --dcp-dir $SEED --out $R/stage4_r59_pmmp > $R/r59_dcp_pmmp_$D.txt 2>&1 || say "  dcp_build FAILED: $(tail -3 $R/r59_dcp_pmmp_$D.txt)"
  tail -40 $R/r59_dcp_pmmp_$D.txt | cut -c1-200
  say "  the same on DataYes"
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/dcp_build.py --dates $D --dcp-dir $SEED --out $R/stage4_r59_dysl2 > $R/r59_dcp_dysl2_$D.txt 2>&1 || say "  dcp_build (dysl2) FAILED: $(tail -3 $R/r59_dcp_dysl2_$D.txt)"
  tail -40 $R/r59_dcp_dysl2_$D.txt | cut -c1-200
done
grep -lE "Error|Traceback" $R/r59*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
