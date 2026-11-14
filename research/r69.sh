#!/usr/bin/env bash
# Round 69 on 97, foreground. Local only (writes under /work; nothing the baseline reads is overwritten).
# The whole-book replay in C++ (research/featgen/cpp/bookgen.cpp), the last piece of the 144 columns:
#   0 its self-test against full_book.book_at (numba) on synthetic events
#   1 0430 and 1112 at the scoring marks, against full_book.py's parquet (the Python replay the baseline reads)
#   2 the 144 columns with the C++ book column, against ic_score's matrix (R67: 141 of 144 on both days)
#   3 20261112 on a 3-second grid with every piece in C++: book, state engine, base, formula layer -> all 144 columns
#   bash research/r69.sh 2>&1 | tee data/reports/r69.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
say() { echo "[$(date +%T)] $*"; }
args_of() { if [ $1 = 20260430 ]; then echo "feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz"; else echo "dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/$1.jz"; fi; }
say "0 self-test"
python -u research/featgen/book_day.py --selftest --work data/bookgen_work > $R/r69_self.txt 2>&1; echo "  exit $?, $(tail -1 $R/r69_self.txt)"; grep -E "OFF|Error" $R/r69_self.txt | head -5 | cut -c1-200
for D in 20260430 20261112; do
  set -- $(args_of $D)
  say "1 $D book at the scoring marks"
  CRUCIBLE_SOURCE=$1 python -u research/featgen/book_day.py --date $D --state $6 --marks $2 --out data/featgen_book --check $4/date=$D.parquet 2>&1 | tail -3 | cut -c1-300
  say "2 $D the 144 columns with the C++ book"
  O=$R/stage6_r67_${D:4}_a
  python -u research/featgen/vec_day.py --date $D --base data/featgen_r67/base_$D.parquet --full-book data/featgen_book/date=$D.parquet --engine $3/marks_$D.parquet \
    --side $O/side_$D.npz --cal $O/cal_$D.json --x $O/x.parquet --float32 --out data/featvec_r69 2>&1 | grep -vE "assemble|names nobody" | cut -c1-300
done
say "3 20261112 every 3 s, all in C++"
G=$(python -c "import numpy as np; print(','.join(repr(float(x)) for x in np.arange(3.0, 14220.0 + 1e-9, 3.0)))")
t0=$(date +%s)
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/book_day.py --date 20261112 --state $SEED/20261112.jz --grid 3 --out data/featgen_book_3s 2>&1 | tail -1 | cut -c1-300; t1=$(date +%s)
python -u research/engine/run_cpp.py --date 20261112 --universe pm:$SEED/20261112.jz --prev $SEED/20261111.jz --catalog $DATA/catalog_dysl2.duckdb \
  --marks "$G" --threads 32 --out data/engine_3s --work data/engine_3s/work > $R/r69_engine3s.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/r69_engine3s.txt | cut -c1-200)"
tail -1 $R/r69_engine3s.txt | cut -c1-200; t2=$(date +%s)
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED --grid 3 --out data/featgen_3s 2>&1 | tail -2 | cut -c1-220; t3=$(date +%s)
python -u research/featgen/vec_day.py --date 20261112 --base data/featgen_3s/base_20261112.parquet --full-book data/featgen_book_3s/date=20261112.parquet \
  --engine data/engine_3s/marks_20261112.parquet --side $R/stage6_r67_1112_a/side_20261112.npz --cal $R/stage6_r67_1112_a/cal_20261112.json --out data/featvec_3s 2>&1 | cut -c1-300; t4=$(date +%s)
echo "  seconds: book $((t1 - t0)), state engine $((t2 - t1)), base $((t3 - t2)), formula layer $((t4 - t3)), total $((t4 - t0))"
python - <<'PY' 2>&1 | cut -c1-220
import polars as pl
df = pl.scan_parquet("data/featvec_3s/vec_20261112.parquet")
cols = [c for c in df.collect_schema().names() if c not in ("sid", "mark")]
fin = df.select([pl.col(c).is_finite().mean().alias(c) for c in cols]).collect().row(0)
low = sorted(zip(fin, cols))[:8]
print(f"  vec_20261112.parquet: {df.select(pl.len()).collect().item():,} rows x {len(cols)} columns; columns with no finite value: {sum(v == 0 for v in fin)}; "
      f"lowest finite shares: {', '.join(f'{c} {v:.3f}' for v, c in low)}")
PY
ls -la data/featvec_3s/*.parquet data/featgen_book_3s/*.parquet 2>&1 | sed 's/^/  /'
say "done"
