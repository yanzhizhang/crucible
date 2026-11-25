#!/usr/bin/env bash
# Round 207 on 97, foreground, read-only on PM data; writes data/engine_r207, data/side_r207, data/single_fill_r207,
# data/reports/r207. ~10 min per day, moderate (32 threads). ONE ENGINE RUN AT A TIME.
# zzvm.c.re (R204) and zzmh.hm (R193) are exact in the engine now; do they give PM's model columns? Per day 1118 / 19 / 20:
#   1 engine on dysl2 with the scoring marks (self-test first) -> marks_<D>.parquet
#   2 state_formulas: key / px, (px - key) / key, (px - key) / px and the in-family pairs
#   3 target-free check: each zzvm / zzmh candidate's quantiles next to the column's training range and thresholds
#   4 single_fill: each candidate alone in the column, change in agreement with PM's logged prediction
#   bash research/r207.sh 2>&1 | tee data/reports/r207.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
G=$DATA/models_gen/B
O=data/reports/r207; mkdir -p $O
# the model's column list: /data/prod/models on the old 97; after the rebuild it may sit elsewhere (FCSV=... overrides)
FCSV=${FCSV:-}
for c in $DATA/prod_models/lgb_fn_v1_209_cte.csv /data/prod/models/lgb_fn_v1_209_cte.csv /data/prod/models/tk5m/lgb_fn_v1_209_cte.csv $DATA/models_gen/B/lgb_fn_v1_209_cte.csv; do
  [ -z "$FCSV" ] && [ -e "$c" ] && FCSV=$c
done
[ -n "$FCSV" ] || FCSV=$(find /data/prod $DATA -maxdepth 5 -name lgb_fn_v1_209_cte.csv 2>/dev/null | head -1)
[ -n "$FCSV" ] && [ -e "$FCSV" ] || { echo "no lgb_fn_v1_209_cte.csv under /data/prod or $DATA: set FCSV=<path> and rerun"; exit 1; }
echo "feature csv: $FCSV ($(wc -l < $FCSV) lines)"
COLS=$(FCSV=$FCSV python - <<'PY'
import csv, os
rows = [r for r in csv.reader(open(os.environ["FCSV"]))]
names = [r[0].strip() for r in rows if r and r[0].strip().startswith(("zzvm", "zzmh"))]
print(",".join(dict.fromkeys(names)))
PY
)
echo "model columns of the two families: $COLS"
[ -n "$COLS" ] || { echo "no zzvm / zzmh column in $FCSV: stopping"; exit 1; }
python -u research/engine/selftest.py --stocks 20 --events 4000 > $O/selftest.txt 2>&1; RC=$?
tail -3 $O/selftest.txt | cut -c1-220
[ $RC -eq 0 ] || { echo "SELFTEST FAILED (rc $RC): not running the engine"; exit 1; }
for D in ${DAYS:-20261118 20261119 20261120}; do
  echo "[$(date +%T)] == $D"
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  if [ ! -e data/engine_r207/dysl2/marks_$D.parquet ]; then
    python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --chan-hist "dysl2_raw,feitu_raw" --marks rand:240 --threads 32 \
      --out data/engine_r207/dysl2 --work data/engine_r207/dysl2/work > $O/engine_$D.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $O/engine_$D.txt)"; continue; }
  fi
  python -u research/featgen/state_formulas.py --date $D --engine data/engine_r207/dysl2/marks_$D.parquet --raw $DATA/dysl2_raw --out data/side_r207 2>&1 | grep -vE "Warning|return grid" | tail -1 | cut -c1-240
  for C in $(echo $COLS | tr ',' ' '); do
    python -u research/featgen/col_probe.py --col $C --date $D --models $G --feature-csv $FCSV --subs s1,s2,s3,s4 2>&1 | grep -vE "Warning" | head -14 | cut -c1-240
  done
  python - <<PY 2>&1 | grep -v Warning | cut -c1-240
import numpy as np, polars as pl
s = pl.read_parquet("data/side_r207/state_$D.parquet")
cs = [c for c in s.columns if c.startswith(("zzvm", "zzmh", "(px-zzvm", "(px-zzmh")) or "/zzvm" in c or "/zzmh" in c]
print(f"   candidates of the two families ({len(cs)}): finite share | q0.1 q1 q50 q99 q99.9")
for c in cs:
    v = s[c].drop_nulls().to_numpy(); v = v[np.isfinite(v)]
    if v.size:
        print(f"   {c:<44} {v.size / s.height:.3f} | " + " ".join(f"{x:.6g}" for x in np.quantile(v, [.001, .01, .5, .99, .999])))
with open("$O/fills_$D.txt", "w") as f:
    f.write(",".join(f"{col}:{t}:x.{c}" for col in "$COLS".split(",") for c in cs for t in ("id", "-id")))
PY
  X=data/reports/stage6_r99_${D:4}; B=data/featgen_r99/base_$D.parquet
  [ -e $X/x.parquet ] || { X=data/reports/stage6_s224_${D:4}; B=data/featgen_s224/base_$D.parquet; }
  echo "   scoring matrix: $X"
  python -u research/featgen/single_fill.py --day $D --x-dir $X --base $B --extra data/side_r207/state_$D.parquet --fills "$(cat $O/fills_$D.txt)" \
    --models $G --feature-csv $FCSV --subs s1,s2,s3,s4 --work data/single_fill_r207 2>&1 | grep -v Warning | grep -v "^  skip" | head -40 | cut -c1-200
done
echo "[$(date +%T)] r207 done"
