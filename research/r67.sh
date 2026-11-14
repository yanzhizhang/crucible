#!/usr/bin/env bash
# Round 67 on 97, foreground. Local only (writes under /work).
# R66: gen.stoch on 0430 differed because the trade loader sorted by arrival stamp only; trades of one arrival batch share
# that stamp (feitu, 0430), so "the last trade at a mark" changed from run to run -- four cached builds of the same day
# disagreed with each other. The loaders now break ties by the channel sequence number (dcp_ema.load, tick_flow.load_orders).
# dcp_ema / tick_flow changed, so every day's cached build is rebuilt once more.
#   1 0430 built twice from scratch (--rebuild): are the two matrices identical now?
#   2 C++ columns against ic_score's matrix on 0430 and 1112 (R64: 142 / 143 of 144), and the similarity
#   bash research/r67.sh 2>&1 | tee data/reports/r67.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
C1=$R/stage6_r41c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99 --models $DATA/models_gen/B --dump-x"
args_of() { if [ $1 = 20260430 ]; then echo "feitu rand:240:7 data/engine_m7 $DATA/store/pm_factors/full_book_m7 $P1 $P1/20260430.jz"; else echo "dysl2 rand:240 data/engine $DATA/store/pm_factors/full_book_dysl2 $SEED $SEED/$1.jz"; fi; }
score() {  # score <day> <tag> [--rebuild]
  local D=$1 tag=$2; shift 2
  set -- $(args_of $D) "$@"
  local src=$1 marks=$2 eng=$3 fb=$4 seed=$5; shift 6
  CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks $marks --engine-dir $eng --full-book $fb --seed-dir $seed "$@" \
    --out $R/stage6_r67_${D:4}_$tag > $R/r67_${D}_$tag.txt 2>&1
  echo "  $D $tag: similarity $(ic $R/r67_${D}_$tag.txt)  $(grep -E 'Error|Traceback' $R/r67_${D}_$tag.txt | head -1 | cut -c1-120)"
}
say "1 0430 built twice from scratch (r41c before: .6282)"
score 20260430 a --rebuild
score 20260430 b --rebuild
python - <<'PY' 2>&1 | cut -c1-200
import numpy as np, polars as pl
a = pl.read_parquet("data/reports/stage6_r67_0430_a/x.parquet").sort("sid", "mark")
b = pl.read_parquet("data/reports/stage6_r67_0430_b/x.parquet").sort("sid", "mark")
off = []
for c in a.columns[3:]:
    u, v = a[c].to_numpy(), b[c].to_numpy()
    n = int((~((u == v) | (np.isnan(u) & np.isnan(v)))).sum())
    if n:
        off.append(f"{c}:{n}")
print(f"  two builds of 0430: {len(a.columns) - 3} columns, {a.height:,} rows; columns with any differing cell: {off or 'none'}")
PY
say "2 1112 (r41c before: .6558), then the C++ columns against both days' matrices"
score 20261112 a
for D in 20260430 20261112; do
  set -- $(args_of $D)
  O=$R/stage6_r67_${D:4}_a
  CRUCIBLE_SOURCE=$1 python -u research/featgen/run_day.py --date $D --state $6 --seed-dir $5 --marks $2 --out data/featgen_r67 2>&1 | tail -1 | cut -c1-200
  python -u research/featgen/vec_day.py --date $D --base data/featgen_r67/base_$D.parquet --full-book $4/date=$D.parquet --engine $3/marks_$D.parquet \
    --side $O/side_$D.npz --cal $O/cal_$D.json --x $O/x.parquet --float32 --out data/featvec_r67 2>&1 | grep -vE "assemble|names nobody" | cut -c1-300
done
say "done"
