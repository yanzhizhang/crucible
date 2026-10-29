#!/usr/bin/env bash
# One round on 97, on CPUS cores (taskset + nice; libraries told the same budget):
#   A. Stage 4: PM's sampler files of every family on the days we hold ticks for -- export, candidate formulas (size tiers =
#      PM's own ZZCR samplerR of the previous PM day; reused once built, REUSE=0 rebuilds), each PM file ranked against the
#      candidate family its COLUMNS match (stage4_route), one summary; the basics table for EVERY overlap day (basics.py) and
#      every PM column scored against every basic at the same day / the day before / multi-day blends (basic_match.py);
#      the size-like columns against PM's own samplerR quantiles (sampler_pm.py);
#   B. (DCP=1 only) Stage 4f: the six open day-end state keys (dcp_open.py);
#   C. the engine (research/engine): run + verify against PM's .jz and the vectorized rebuild, the keys not at 1.0 listed.
# Foreground; logs under data/reports/stage4, stage4f, engine. DAYS = the days to run (default: the SZSE-reference days
# that PM's hstats tree also has; DAYS=all for every overlap day).
set -u
cd /work/crucible
CPUS=${CPUS:-64}
ALL=$(nproc --all); [ $CPUS -gt $ALL ] && CPUS=$ALL
DAYS=${DAYS:-20260923,20260924,20260929,20260930,20261009}
REUSE=${REUSE:-1}
export POLARS_MAX_THREADS=$CPUS OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
H=/data/prod/hstats/HS300
C=/work/crucible_data/store/pm_factors/candidates
BS=/work/crucible_data/store/pm_factors/basics
R4=data/reports/stage4
RF=data/reports/stage4f
RE=data/reports/engine
X=data/pm_export
mkdir -p $R4 $RF $RE $X data/engine
run() { taskset -c 0-$((CPUS - 1)) nice -n 5 python -u "$@"; }
echo "cpus $CPUS of $ALL on the box ($(nproc) in this shell's mask), $(free -g | awk '/Mem:/ {print $7}') GB free"

echo "### A0 inventory: PM hstats dates per family vs our tick days"
python - "$H" "$DAYS" <<'PYEOF' | tee $R4/inventory.txt
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
print(f"overlap days: {' '.join(both)}")
want = both if days == "all" else [d for d in days.split(",") if d in both]
print(f"DAYS: {' '.join(want)}")
Path("data/reports/stage4/days.txt").write_text(" ".join(want))
Path("data/reports/stage4/overlap.txt").write_text(" ".join(both))
PYEOF
WANT=$(cat $R4/days.txt)
[ -n "$WANT" ] || { echo "no overlap day to run"; exit 1; }
tiers() { local prev; prev=$(ls $H/ZZCR | grep -E '^[0-9]{8}$' | awk -v d=$1 '$1 < d' | tail -1); [ -n "$prev" ] && echo $H/ZZCR/$prev/samplerR.parquet; }

echo "### A1a basics for every overlap day (basic_match's blends need the days before each PM day)"
todo=
for d in $(cat $R4/overlap.txt); do
  [ -n "$(tiers $d)" ] || continue
  [ "$REUSE" = 1 ] && [ -f $BS/date=$d/m1.parquet ] && continue
  todo="$todo,$d"
done
[ -n "$todo" ] && run research/pm_factors/basics.py --dates ${todo#,} --thr-dir $H/ZZCR --workers $((CPUS / 4)) --threads 4 --memory 8GB 2>&1 | grep -E "stock-minutes|Error|error:|Traceback"

echo "### A1 export PM sampler files, candidates with PM's ZZCR tiers of the previous PM day, rank per file (routed by columns)"
first=
for d in $WANT; do
  thr=$(tiers $d); [ -n "$thr" ] || { echo "-- $d: no earlier PM day for the size tiers, skipped"; continue; }
  echo "-- $d (size tiers: $thr)"
  if [ "$REUSE" = 1 ] && [ -f $C/family=ZZXC/date=$d/part.parquet ] && [ -f $C/family=ZZCR/date=$d/part.parquet ]; then
    echo "  candidates already built, reused"
  else
    run research/pm_factors/candidates.py --dates $d --thr-file $thr --memory 16GB > $R4/candidates_$d.txt 2>&1
    grep -E "rows,|universe|WARN" $R4/candidates_$d.txt | head -20
    grep -q "rows," $R4/candidates_$d.txt || { echo "  candidates failed:"; tail -3 $R4/candidates_$d.txt; continue; }
  fi
  [ -n "$first" ] || first=$d
  for f in $(ls $H); do
    for s in $(ls $H/$f/$d/sampler*.nc 2>/dev/null); do
      tag=$f; [ "$(basename $s)" = samplerS.nc ] || tag=${f}$(basename $s .nc | sed 's/^sampler//')  # ZZDS has sampler.nc and samplerS.nc
      if [ "$REUSE" = 1 ] && [ -f $X/${tag}_$d.parquet ]; then :; else
        run research/compare_pm.py --pm $s --export $X/${tag}_$d.parquet > $R4/export_${tag}_$d.txt 2>&1 || { echo "  export $tag $d failed"; continue; }
      fi
      [ "$REUSE" = 1 ] && [ -f $R4/${tag}_$d.parquet ] && continue
      route=$(run research/pm_factors/stage4_route.py --pm $X/${tag}_$d.parquet --date $d)
      [ -n "$route" ] || { echo "  $tag: no candidate family covers its columns"; continue; }
      run research/pm_factors/rank_candidates.py --family $f --pm $X/${tag}_$d.parquet $route --dates $d --out $R4/${tag}_$d.parquet > $R4/rank_${tag}_$d.txt 2>&1 \
        && echo "  $tag -> ${route#--ours */family=}: $(grep -m1 '^PM ' $R4/rank_${tag}_$d.txt | cut -c1-120)" \
        || { echo "  $tag -> ${route#--ours */family=}: rank failed ($(tail -1 $R4/rank_${tag}_$d.txt | cut -c1-100))"; grep -E "^PM |^ours |^PM columns" $R4/rank_${tag}_$d.txt | cut -c1-200; }
    done
  done
done
run research/pm_factors/stage4_summary.py --reports $R4 2>&1 | tee $R4/summary.txt | tail -3

echo "### A2 every PM column against every basic quantity, same day / day before / multi-day blends (basic_match)"
POLARS_MAX_THREADS=4 run research/pm_factors/basic_match.py --dates $(echo $WANT | tr ' ' ',') --exports $X --workers $((CPUS / 4)) --out $R4/basic_match.csv > $R4/basic_match.txt 2>&1
grep -E "^[0-9]{8} |^table|Error|error:|nothing joined" $R4/basic_match.txt | tail -8
echo "### A3 size-like samplerS columns against PM's own samplerR quantiles (sampler_pm)"
for d in $WANT; do
  run research/pm_factors/sampler_pm.py --date $d --hstats $H --exports $X --out $R4 > $R4/sampler_pm_$d.txt 2>&1 || tail -2 $R4/sampler_pm_$d.txt
done

if [ "${DCP:-0}" = 1 ]; then
echo "### B open state keys"
run research/pm_factors/dcp_open.py --dates $(echo $WANT | tr ' ' ',') --out $RF 2>&1 | tee $RF/dcp_open.txt | grep -E "stocks scored|Error|error:|^tables"
fi

echo "### C the engine: every solved rule as one event loop, day-end state against PM's .jz and the vectorized rebuild"
for d in $WANT; do
  jz=/data/prod/h/HS300/p1_intrading_dcp/$d.jz
  [ -f $jz ] || { echo "-- $d: no state file"; continue; }
  run research/engine/run.py --date $d --universe pm:$jz --workers $((CPUS / 2)) --out data/engine > $RE/run_$d.txt 2>&1; tail -1 $RE/run_$d.txt
  run research/engine/verify.py --date $d --engine data/engine/engine_$d.json.gz --pm $jz --reference > $RE/verify_$d.txt 2>&1 \
    && echo "  verified -> $RE/verify_$d.txt" || { echo "  verify failed:"; tail -3 $RE/verify_$d.txt; }
done

echo
echo "########## SUMMARY (photograph from here) ##########"
tail -2 $R4/inventory.txt
sed -n '/^== PM column -> basic/,$p' $R4/basic_match.txt | grep -vE "^│ (---|str|f64|u32|i64)|^shape"
[ -f $R4/sampler_pm_$first.txt ] && grep -vE "^│ (---|str|f64|u32|i64)|^shape" $R4/sampler_pm_$first.txt
for d in $WANT; do
  echo "== engine $d: $(tail -1 $RE/run_$d.txt 2>/dev/null | cut -c1-120)"
  [ -f $RE/verify_$d.txt ] && grep -E "^PM |^== engine|^│ (zz|cdc)|Error|Traceback" $RE/verify_$d.txt | grep -vE "┆ 1.0000 ┆ 1.0000" | head -30
done
