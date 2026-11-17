#!/usr/bin/env bash
# Round 149 on 97, foreground: the engine restructured into one class per family (engine/families/*.py, engine/cpp/families/*.hpp)
# must give the same state as before. Read-only on inputs; writes only data/engine_fam (a new directory).
#  1 selftest: Python engine vs C++ engine on synthetic events, plus the C++ built-in rule table vs the Python families' table
#  2 the family table (status, averages, columns per family)
#  3 20261113 rerun with the new code into data/engine_fam, compared value for value with the existing data/engine output
#    (day-end state and the 240 marks) made by the old code. ~3 min.
#   bash research/r149.sh 2>&1 | tee data/reports/r149.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
D=20261113
echo "== 1 selftest"
python -u research/engine/selftest.py 2>&1 | tail -4
echo "== 2 families"
(cd research/engine && python -m families) | cut -c1-200
echo "== 3 $D before / after"
P=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
python -u research/engine/run_cpp.py --date $D --universe pm:$SEED/$P.jz --prev $SEED/$P.jz --catalog $DATA/catalog_dysl2.duckdb \
  --marks rand:240 --threads 32 --out data/engine_fam --work data/engine_fam/work 2>&1 | grep -v Warning | tail -3
python - <<PY
import gzip, json, math
import polars as pl
a = json.load(gzip.open("data/engine/engine_$D.json.gz")); b = json.load(gzip.open("data/engine_fam/engine_$D.json.gz"))
n = bad = miss = 0
ex = []
for sid, kv in a.items():
    kb = b.get(sid)
    if kb is None or set(kv) != set(kb):
        miss += 1
        continue
    for k, v in kv.items():
        n += 1
        w = kb[k]
        if not ((v is None and w is None) or v == w):
            bad += 1
            if len(ex) < 8:
                ex.append((sid, k, v, w))
print(f"day-end: {len(a)} / {len(b)} stocks, {n:,} values compared, {bad} differ, {miss} stocks with a different key set")
for e in ex:
    print("   ", e)
try:
    ma, mb = pl.read_parquet("data/engine/marks_$D.parquet"), pl.read_parquet("data/engine_fam/marks_$D.parquet")
    cols = [c for c in ma.columns if c in mb.columns]
    key = [c for c in ("sid", "mark") if c in cols]
    ma, mb = ma.sort(key).select(cols), mb.sort(key).select(cols)
    if ma.height != mb.height:
        print(f"marks: {ma.height} vs {mb.height} rows (the old file may be from another mark set): not compared")
    else:
        d = sum(int(((ma[c] != mb[c]) & ~(ma[c].is_nan() & mb[c].is_nan())).fill_null(False).sum()) if ma[c].dtype.is_float() else int((ma[c] != mb[c]).sum()) for c in cols)
        print(f"marks: {ma.height:,} rows x {len(cols)} columns ({len(set(ma.columns) ^ set(mb.columns))} columns not shared), {d} values differ")
except Exception as e:
    print("marks: not compared:", e)
PY
echo "[$(date +%T)] r149 done"
