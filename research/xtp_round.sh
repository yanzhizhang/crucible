#!/usr/bin/env bash
# One XTP round on 97: probe the TDF layout, convert every day of the dump, compare the overlap day with feitu,
# refresh the catalog, then one coverage line per (day, exchange). Foreground; every step tees its log to data/reports/stage8.
# Days that feitu_raw already holds go to xtp_raw (two sources in one partition would double-count); the rest to feitu_raw.
set -u
cd /work/crucible
R=data/reports/stage8
mkdir -p $R
ROOT=${ROOT:-/data/xtp_tdf}
OVERLAP=${OVERLAP:-20261009}
DAYS=${DAYS:-all}
T=${T:-96}  # worker processes, polars single-threaded in each; lower it if memory runs short
FEITU=/work/crucible_data/feitu_raw
XTP=/work/crucible_data/xtp_raw

if [ "$DAYS" = all ]; then
  DAYS=$(ls $ROOT/600000.SH/3 $ROOT/000001.SZ/3 2>/dev/null | grep -o '^20[0-9]\{6\}' | sort -u | tr '\n' ' ')
fi
echo "days: $DAYS"

echo "### 0/4 probe the layout ($OVERLAP)"
python -u research/decode_tdf_day.py --root $ROOT --date $OVERLAP --probe 2>&1 | tee $R/probe.txt | grep -E "^==|UNMAPPED|left at 0|symbols under|rows [0-9]"

echo "### 1/4 convert"
NEW=; OLD=
for d in ${DAYS//,/ }; do
  if ls $FEITU/kind=transaction/date=$d/part-*.parquet 2>/dev/null | grep -vqE 'part-(SH|SZ)-[0-9]{3}\.parquet$'; then
    OLD="$OLD,$d"  # feitu already has this day
  else
    NEW="$NEW,$d"
  fi
done
for pair in "$XTP:${OLD#,}" "$FEITU:${NEW#,}"; do
  out=${pair%%:*}; ds=${pair#*:}
  [ -n "$ds" ] || continue
  echo "-- $ds -> $out"
  python -u research/decode_tdf_day.py --root $ROOT --date $ds --out $out --workers $T 2>&1 | tee $R/convert_$(basename $out).txt \
    | grep -E "parts to write|rows in|Error|error:|\((25|50|100|200|400|800)/"
done

echo "### 2/4 compare $OVERLAP with feitu"
python -u research/compare_xtp_feitu.py --date $OVERLAP --xtp $XTP 2>&1 | tee $R/compare_$OVERLAP.txt | tail -5

echo "### 3/4 catalog"
python -u research/catalog.py 2>&1 | tail -3

echo "### 4/4 coverage per day"
python - $FEITU $XTP <<'EOF' 2>&1 | tee $R/coverage.txt
import sys
from pathlib import Path
import polars as pl
rows = []
for root in map(Path, sys.argv[1:]):
    for q in sorted(root.glob("kind=order/date=*/_quality.parquet")):
        d = q.parent.name.removeprefix("date=")
        a = (pl.read_parquet(q).group_by("market_id").agg(pl.col("records").sum(), pl.col("seq_span").sum(), pl.col("seq_distinct").sum())
             .with_columns((1 - pl.col("seq_distinct") / pl.col("seq_span")).alias("missing_frac")))
        qd = root / "kind=quotation" / f"date={d}"
        s = (pl.scan_parquet(str(qd / "part-*.parquet")).group_by("market_id", "symbol_id").agg(pl.len().alias("n"))
             .group_by("market_id").agg(pl.col("n").median().alias("snap_med"), pl.len().alias("symbols")).collect()) if list(qd.glob("part-*.parquet")) else None
        if s is not None:
            a = a.join(s, on="market_id", how="left")
        rows.append(a.with_columns(pl.lit(d).alias("date"), pl.lit(root.name).alias("tree")))
if not rows:
    sys.exit("no kind=order/date=*/_quality.parquet yet: the order/transaction conversion did not finish")
df = pl.concat(rows, how="diagonal_relaxed").select("date", "tree", "market_id", "records", "missing_frac", "snap_med", "symbols").sort("date", "market_id")
with pl.Config(tbl_rows=120, tbl_width_chars=150, float_precision=5):
    print(df)
EOF

echo
echo "########## SUMMARY (photograph from here) ##########"
grep -E "UNMAPPED|left at 0" $R/probe.txt | sort -u
sed -n '/totals per exchange/,$p' $R/compare_$OVERLAP.txt | grep -vE "^shape|^│ ---|^│ i16|^│ str|^│ f64|^│ u32|^│ i64|^│ decimal" | head -60
grep -E "^│ 20|^│ date" $R/coverage.txt
