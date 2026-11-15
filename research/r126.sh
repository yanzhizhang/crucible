#!/usr/bin/env bash
# Round 126 on 97, foreground. Read-only on PM data; 2.23 is only READ (rsync of 4 files, user-approved 2026-11-15 in the main session
# ). Writes /work/crucible_data/feitu_zst, /work/crucible_data/feitu_index_raw, data/day_r126, data/full_0430,
# data/reports. ~20-40 min.
# A: 0429 / 0430 NonConvex (feitu) index stream (v3_index_*.zst, ~78 MB a day) from 2.23 -> 97, decoded into its own directory.
# B: 0430 at the full 3 s grid (day.py, feitu = what PM trained on), PM's alpha at every mark (~20x the rows of the 240
#    random marks every scan used so far) -> data/full_0430 (an implied_scan / greedy_fill day dir), and residual studies:
#    by time of day, by drawdown from the day's high (zzmh), and by time scale.
#   bash research/r126.sh 2>&1 | tee data/reports/r126.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
H23=${H23:-root@172.21.2.23}
SSH="ssh -p ${P23:-60777} -o BatchMode=yes"
say() { echo "[$(date +%T)] $*"; }
say "A 0429 / 0430 index stream from 2.23 (read-only there)"
for D in 20260429 20260430; do
  mkdir -p $DATA/feitu_zst/$D
  rsync -a -e "$SSH" --include="v3_index_*.zst" --exclude="*" $H23:/mnt/nas/pica_data_hdd_1/TickRepository/hdfs_data/2026/$D/ $DATA/feitu_zst/$D/ 2>&1 | tail -2
  ls -l $DATA/feitu_zst/$D | sed 's/^/  /'
  python -u research/decode_feitu_day.py --root $DATA/feitu_zst/$D --date $D --out $DATA/feitu_index_raw --kinds index 2>&1 | tail -3 | cut -c1-200
done
say "B 0430 at 3 s"
ST=$P1/20260430.jz; [ -e $ST ] || ST=$P1/20260429.jz
echo "  state $ST"
CRUCIBLE_SOURCE=feitu python -u research/featgen/day.py --date 20260430 --source feitu --state $ST --seed-dir $P1 --catalog $DATA/catalog.duckdb --grid 3 \
  --side data/reports/stage6_r70_0430/side_20260430.npz --cal data/reports/stage6_r70_0430/cal_20260430.json --models $DATA/models_gen/B --out data/day_r126 2>&1 | tail -12 | cut -c1-200
python -u research/featgen/full_day.py --date 20260430 --day data/day_r126 --target $DATA/pm_alpha/date=20260430.parquet \
  --cal data/reports/stage6_r70_0430/cal_20260430.json --out data/full_0430 --study 2>&1 | cut -c1-200
say "done"
echo "===== summary"; grep -E "^\[|index|rows|corr|var|min |\[|<15|15-60|1-5|5-30|>30" data/reports/r126.txt 2>/dev/null | tail -60
