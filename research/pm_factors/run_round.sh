#!/usr/bin/env bash
# One round on 97, on CPUS cores (taskset + nice; libraries told the same budget):
#   A. Stage 4 for real: PM's samplerS of every family on the days we hold ticks for -- export, our candidate formulas
#      (size tiers from PM's own ZZCR samplerR of the previous PM day), rank_candidates per family and day, one summary;
#   B. Stage 4f: the six open day-end state keys against fresh candidates (dcp_open.py).
# Foreground; logs under data/reports/stage4 and stage4f. DAYS = the days to run (default: the SZSE-reference days
# that PM's hstats tree also has; set DAYS=all for every overlap day).
set -u
cd /work/crucible
CPUS=${CPUS:-32}
DAYS=${DAYS:-20260923,20260924,20260929,20260930,20261009}
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
H=/data/prod/hstats/HS300
R4=data/reports/stage4
RF=data/reports/stage4f
X=data/pm_export
mkdir -p $R4 $RF $X
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
echo "cpus $CPUS of $(nproc), $(free -g | awk '/Mem:/ {print $7}') GB free"

echo "### A0 inventory: PM hstats dates per family vs our tick days"
python - "$H" "$DAYS" <<'EOF' | tee $R4/inventory.txt
import sys
from pathlib import Path
import duckdb
h, days = Path(sys.argv[1]), sys.argv[2]
ours = {str(r[0]) for r in duckdb.connect("/work/crucible_data/catalog.duckdb", read_only=True).sql("SELECT DISTINCT date FROM raw_quotation").fetchall()}
fams = sorted(p.name for p in h.iterdir() if p.is_dir())
allpm = set()
for f in fams:
    ds = sorted(p.name for p in (h / f).iterdir() if p.is_dir() and p.name.isdigit())
    files = sorted({q.name for d in ds[-1:] for q in (h / f / d).iterdir()})
    both = [d for d in ds if d in ours]
    allpm |= set(ds)
    print(f"{f}: {len(ds)} days {ds[0] if ds else '-'}..{ds[-1] if ds else '-'}, {len(both)} with our ticks; files {files}")
both = sorted(allpm & ours)
print(f"overlap days: {both}")
want = both if days == "all" else [d for d in days.split(",") if d in both]
print(f"DAYS: {' '.join(want)}")
Path("data/reports/stage4/days.txt").write_text(" ".join(want))
EOF
WANT=$(cat $R4/days.txt)
[ -n "$WANT" ] || { echo "no overlap day to run"; exit 1; }

echo "### A1 export PM samplerS, candidates with PM's ZZCR tiers of the previous PM day, rank per family"
for d in $WANT; do
  prev=$(ls $H/ZZCR | grep -E '^[0-9]{8}$' | awk -v d=$d '$1 < d' | tail -1)
  thr=$H/ZZCR/$prev/samplerR.parquet
  echo "-- $d (size tiers: $thr)"
  run research/pm_factors/candidates.py --dates $d --thr-file $thr --memory 16GB 2>&1 | tee $R4/candidates_$d.txt | grep -E "rows,|Error|error:|WARN"
  for f in $(ls $H); do
    s=$(ls $H/$f/$d/sampler*.nc 2>/dev/null | head -1)
    [ -n "$s" ] || continue
    run research/compare_pm.py --pm $s --export $X/${f}_$d.parquet > $R4/export_${f}_$d.txt 2>&1 || { echo "export $f $d failed"; continue; }
    run research/pm_factors/rank_candidates.py --family $f --pm $X/${f}_$d.parquet --out $R4/${f}_$d.parquet > $R4/rank_${f}_$d.txt 2>&1 \
      && echo "  $f: $(grep -m1 '^PM ' $R4/rank_${f}_$d.txt | cut -c1-150)" || echo "  $f: rank failed ($(tail -1 $R4/rank_${f}_$d.txt | cut -c1-120))"
  done
done
run research/pm_factors/stage4_summary.py --reports $R4 2>&1 | tee $R4/summary.txt | tail -3

echo "### B open state keys"
run research/pm_factors/dcp_open.py --dates $(echo $WANT | tr ' ' ',') --out $RF 2>&1 | tee $RF/dcp_open.txt | grep -E "stocks scored|Error|error:|^tables"

echo
echo "########## SUMMARY (photograph from here) ##########"
cat $R4/inventory.txt
sed -n '/^== families/,$p' $R4/summary.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
sed -n '/^== zzcb.r/,$p' $RF/dcp_open.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
