#!/usr/bin/env bash
# One round on 97, on CPUS cores (taskset + nice; libraries told the same budget):
#   A. Stage 4 for real: PM's sampler files of every family on the days we hold ticks for -- export (every sampler*.nc of
#      the folder), our candidate formulas (size tiers = PM's own ZZCR samplerR of the previous PM day; reused when
#      already built, REUSE=0 rebuilds), each PM file routed to the candidate family its COLUMNS match (stage4_route),
#      rank_candidates per file and day, a value profile of PM's columns on the first day, one summary; then the
#      basics table (basics.py: every plain tick quantity per stock-minute) and every PM column scored against every basic
#      (basic_match.py) -- the generic answer to "what is this column";
#   B. Stage 4f: the six open day-end state keys against fresh candidates (dcp_open.py), plus the raw key names of the
#      prefixes still open (zzvx / zzba / zzcb / zztx) for one stock, so a key we look for under the wrong name shows.
# Foreground; logs under data/reports/stage4 and stage4f. DAYS = the days to run (default: the SZSE-reference days
# that PM's hstats tree also has; DAYS=all for every overlap day).
set -u
cd /work/crucible
CPUS=${CPUS:-32}
ALL=$(nproc --all); [ $CPUS -gt $ALL ] && CPUS=$ALL
DAYS=${DAYS:-20260923,20260924,20260929,20260930,20261009}
REUSE=${REUSE:-1}
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
H=/data/prod/hstats/HS300
C=/work/crucible_data/store/pm_factors/candidates
R4=data/reports/stage4
RF=data/reports/stage4f
X=data/pm_export
mkdir -p $R4 $RF $X
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
echo "cpus $CPUS of $ALL on the box ($(nproc) in this shell's mask), $(free -g | awk '/Mem:/ {print $7}') GB free"

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

echo "### A1 export PM sampler files, candidates with PM's ZZCR tiers of the previous PM day, rank per file (routed by columns)"
first=
for d in $WANT; do
  prev=$(ls $H/ZZCR | grep -E '^[0-9]{8}$' | awk -v d=$d '$1 < d' | tail -1)
  [ -n "$prev" ] || { echo "-- $d: no earlier PM day for the size tiers, skipped"; continue; }
  thr=$H/ZZCR/$prev/samplerR.parquet
  echo "-- $d (size tiers: $thr)"
  if [ "$REUSE" = 1 ] && [ -f $C/family=ZZXC/date=$d/part.parquet ] && [ -f $C/family=ZZCR/date=$d/part.parquet ]; then
    echo "  candidates already built, reused"
  else
    run research/pm_factors/candidates.py --dates $d --thr-file $thr --memory 16GB > $R4/candidates_$d.txt 2>&1
    grep -E "rows,|universe|WARN" $R4/candidates_$d.txt | head -20
    grep -q "rows," $R4/candidates_$d.txt || { echo "  candidates failed:"; tail -3 $R4/candidates_$d.txt; continue; }
  fi
  [ -n "$first" ] || first=$d
  if [ "$REUSE" = 1 ] && [ -f /work/crucible_data/store/pm_factors/basics/date=$d/m1.parquet ]; then
    echo "  basics already built, reused"
  else
    run research/pm_factors/basics.py --dates $d --thr-file $thr --memory 16GB 2>&1 | grep -E "stock-minutes|Error|error:" | tail -2
  fi
  for f in $(ls $H); do
    for s in $(ls $H/$f/$d/sampler*.nc 2>/dev/null); do
      tag=$f; [ "$(basename $s)" = samplerS.nc ] || tag=${f}$(basename $s .nc | sed 's/^sampler//')  # ZZDS has sampler.nc and samplerS.nc
      run research/compare_pm.py --pm $s --export $X/${tag}_$d.parquet > $R4/export_${tag}_$d.txt 2>&1 || { echo "  export $tag $d failed"; continue; }
      route=$(run research/pm_factors/stage4_route.py --pm $X/${tag}_$d.parquet --date $d)
      [ -n "$route" ] || { echo "  $tag: no candidate family covers its columns"; continue; }
      run research/pm_factors/rank_candidates.py --family $f --pm $X/${tag}_$d.parquet $route --dates $d --out $R4/${tag}_$d.parquet > $R4/rank_${tag}_$d.txt 2>&1 \
        && echo "  $tag -> ${route#--ours */family=}: $(grep -m1 '^PM ' $R4/rank_${tag}_$d.txt | cut -c1-120)" \
        || { echo "  $tag -> ${route#--ours */family=}: rank failed ($(tail -1 $R4/rank_${tag}_$d.txt | cut -c1-100))"; grep -E "^PM |^ours |^PM columns" $R4/rank_${tag}_$d.txt | cut -c1-200; }
    done
  done
done
[ -n "$first" ] && run research/pm_factors/stage4_profile.py --exports $X --date $first > $R4/profile.txt 2>&1
run research/pm_factors/stage4_summary.py --reports $R4 2>&1 | tee $R4/summary.txt | tail -3
echo "### A2 every PM column against every basic quantity (basic_match)"
run research/pm_factors/basic_match.py --dates $(echo $WANT | tr ' ' ',') --exports $X --out $R4/basic_match.csv > $R4/basic_match.txt 2>&1
grep -E "^[0-9]{8} |^table|Error|error:" $R4/basic_match.txt | tail -5

echo "### B open state keys"
run research/pm_factors/dcp_open.py --dates $(echo $WANT | tr ' ' ',') --out $RF 2>&1 | tee $RF/dcp_open.txt | grep -E "stocks scored|Error|error:|^tables"
python - <<'EOF' | tee $RF/keys.txt
import gzip, json, re
from pathlib import Path
f = sorted(Path("/data/prod/h/HS300/p1_intrading_dcp").glob("202610*.jz"))[-1]
d = json.loads(gzip.open(f).read())
sid, kv = next(iter(d.items()))
print(f"== {f.name} stock {sid}: keys of the open prefixes (histogram bins collapsed)")
for p in ("zzvx", "zzba", "zzcb", "zztx", "zzgh", "zzhj"):
    ks = sorted({re.sub(r"\.c\.\d+$", ".c.<n>", k) for k in kv if k.startswith(p + ".")})
    print(f"{p}: {ks}")
EOF

echo
echo "########## SUMMARY (photograph from here) ##########"
cat $R4/inventory.txt | tail -4
grep -vE "^│ (---|str|f64|u32|i64)|^shape" $R4/profile.txt
sed -n '/^== PM column -> basic/,$p' $R4/basic_match.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
sed -n '/^== families/,/^== per PM column/p' $R4/summary.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
cat $RF/keys.txt
sed -n '/^== zzba/,$p' $RF/dcp_open.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
