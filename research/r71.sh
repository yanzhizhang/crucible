#!/usr/bin/env bash
# Round 71 on 97, foreground. Local only (writes under /work; nothing the baseline reads is overwritten; nothing installed).
# The conda env's Arrow / Parquet C++ libraries (already on 97) and the one-command day:
#   0 CMake build of research/featgen/cpp inside the env -> lgbpred_pq (reads the feature parquet, writes the score parquet)
#   1 its self-test: the parquet way against the binary way and against ic_score.TreeModel on made-up trees
#   2 20261112 every 3 s through research/featgen/day.py, tick readers one after another, model step the old way
#   3 the same day, tick readers side by side, model step through lgbpred_pq
#   4 the two runs' vectors and scores compared with each other and with R69's vectors
#   bash research/r71.sh 2>&1 | tee data/reports/r71.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
B=/work/crucible/research/featgen/cpp/build/cmake
say() { echo "[$(date +%T)] $*"; }
say "0 cmake build"
cmake -S research/featgen/cpp -B $B -G Ninja -DCMAKE_PREFIX_PATH=$CONDA_PREFIX > $R/r71_cmake.txt 2>&1; grep -E "featgen:" $R/r71_cmake.txt | cut -c1-200
cmake --build $B >> $R/r71_cmake.txt 2>&1; echo "  build exit $?; $(grep -cE 'warning' $R/r71_cmake.txt) warnings, $(grep -cE 'error' $R/r71_cmake.txt) errors"; grep -E "error" $R/r71_cmake.txt | head -5 | cut -c1-240
ls $B | grep -E "^(lgbpred|lgbpred_pq|libcheck|featgen|featvec|bookgen)$" | tr '\n' ' '; echo
say "1 self-test"
python -u research/featgen/pred_day.py --selftest --work data/lgbpred_work --pq $B/lgbpred_pq 2>&1 | tail -2 | cut -c1-200
O=$R/stage6_r67_1112_a
A="--date 20261112 --source dysl2 --state $SEED/20261112.jz --seed-dir $SEED --catalog $DATA/catalog_dysl2.duckdb --grid 3 --side $O/side_20261112.npz --cal $O/cal_20261112.json --models $DATA/models_gen/B"
say "2 one after another, model the old way"
python -u research/featgen/day.py $A --serial --out data/day_r71_serial 2>&1 | cut -c1-260
say "3 side by side, model through lgbpred_pq"
python -u research/featgen/day.py $A --pq $B/lgbpred_pq --out data/day_r71 2>&1 | cut -c1-260
say "4 compare"
python - <<'PY' 2>&1 | cut -c1-240
import polars as pl
for what, a, b in (("vectors serial vs side by side", "data/day_r71_serial/vec/vec_20261112.parquet", "data/day_r71/vec/vec_20261112.parquet"),
                   ("vectors vs R69", "data/featvec_3s/vec_20261112.parquet", "data/day_r71/vec/vec_20261112.parquet"),
                   ("scores old way vs lgbpred_pq", "data/day_r71_serial/pred/pred_20261112.parquet", "data/day_r71/pred/pred_20261112.parquet")):
    try:
        x, y = pl.read_parquet(a).sort("sid", "mark"), pl.read_parquet(b).sort("sid", "mark")
        x, y = x.with_columns(pl.col("sid").cast(pl.Int64)), y.with_columns(pl.col("sid").cast(pl.Int64))
        bad = [c for c in x.columns if c not in y.columns or not x[c].equals(y[c], null_equal=True)] if x.height == y.height else ["row counts"]
        print(f"  {what}: {x.height:,} / {y.height:,} rows, {len(x.columns)} columns, columns not identical: {len(bad)} {bad[:6]}")
    except Exception as e:
        print(f"  {what}: {type(e).__name__}: {e}")
PY
ls -la data/day_r71/*/*.parquet 2>&1 | sed 's/^/  /'
say "done"
