#!/usr/bin/env bash
# Round 175 on 97, foreground, read-only, light (~1 min): first look at what the 2026-11-19 pulls brought.
#  1 tdf_224 (224's own DataYes feed as TDF, from 2.23): per exchange which streams exist and how many symbols; for four
#    symbols the zip listing of each stream, and the head of the first inner file (text lines, or a hex dump if binary)
#  2 alpha.parquet of /tmp/o/sig for 1117 / 1118 / 1119: schema, rows, a few rows; for 1118 against pm_signals (the signal
#    lines parsed from the chrono log): same stocks, same times, same values?
#  3 PM's config2_HS300.json and sc.p3.json: 1118's copy against 1119's (they changed after the close on 11-18)
# No *.json5 is opened; lines containing passw / secret / token are dropped.
#   bash research/r175.sh 2>&1 | tee data/reports/r175.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
T=$DATA/tdf_224
safe() { grep -a -v -i -E "passw|secret|token" | cut -c1-230; }
echo "== 1 tdf_224: $(ls $T 2>/dev/null | wc -l) symbol dirs, $(du -sh $T 2>/dev/null | cut -f1)"
for ex in SH SZ; do
  echo "   $ex: streams -> symbol count: $(ls -d $T/*.$ex/* 2>/dev/null | sed 's#.*/##' | sort | uniq -c | tr '\n' ' ')"
done
for s in 600000.SH 600519.SH 000001.SZ 300750.SZ; do
  for st in $(ls $T/$s 2>/dev/null); do
    z=$T/$s/$st/20261118.zip
    [ -e $z ] || { echo "-- $s stream $st: no 20261118.zip"; continue; }
    echo "-- $s stream $st: $(stat -c %s $z) bytes"
    unzip -l $z 2>&1 | tail -n +4 | head -6 | safe
    f=$(unzip -Z1 $z 2>/dev/null | head -1)
    if unzip -p $z "$f" 2>/dev/null | head -c 4096 | grep -qaP '[^\x09\x0a\x0d\x20-\x7e]'; then
      echo "   binary; first 256 bytes of $f:"; unzip -p $z "$f" | head -c 256 | xxd | head -16
      echo "   uncompressed size $(unzip -p $z "$f" | wc -c)"
    else
      echo "   text; $(unzip -p $z "$f" | wc -l) lines; first 4 and last 2 of $f:"
      unzip -p $z "$f" | head -4 | safe; unzip -p $z "$f" | tail -2 | safe
    fi
  done
done
echo "== 2 alpha.parquet"
python - <<'PY' 2>&1 | grep -v Warning | cut -c1-230
import polars as pl
from pathlib import Path
D = Path("/work/crucible_data")
for d in ("20261117", "20261118", "20261119"):
    f = D / "pm_224" / d / "sig" / "alpha.parquet"
    if not f.exists():
        print(f"-- {d}: no {f}")
        continue
    t = pl.read_parquet(f)
    print(f"-- {d}: {t.height:,} rows x {t.width} columns, {f.stat().st_size:,} bytes")
    print("   schema: " + ", ".join(f"{k}:{v}" for k, v in t.schema.items()))
    with pl.Config(tbl_cols=30, tbl_width_chars=220, fmt_float="full", tbl_rows=6):
        print(t.head(3)); print(t.tail(2))
    for c in t.columns:
        if t[c].dtype.is_numeric():
            s = t[c]
            print(f"   {c}: nulls {s.null_count()}, min {s.min()}, max {s.max()}, distinct {s.n_unique()}")
        else:
            print(f"   {c}: distinct {t[c].n_unique()}, e.g. {t[c].unique().head(5).to_list()}")
s = D / "pm_signals" / "date=20261118.parquet"
if s.exists():
    p = pl.read_parquet(s)
    print(f"-- pm_signals 20261118 (parsed from the chrono log): {p.height:,} rows; schema: " + ", ".join(f"{k}:{v}" for k, v in p.schema.items()))
    with pl.Config(tbl_cols=30, tbl_width_chars=220, fmt_float="full", tbl_rows=4):
        print(p.head(3))
PY
echo "== 3 PM configs, 1118 vs 1119"
for f in config2_HS300.json sc.p3.json; do
  A=$DATA/pm_224/20261118/$f; B=$DATA/pm_224/20261119/$f
  echo "-- $f: 1118 $([ -e $A ] && stat -c '%s bytes %y' $A | cut -c1-36 || echo missing) | 1119 $([ -e $B ] && stat -c '%s bytes %y' $B | cut -c1-36 || echo missing)"
  [ -e $B ] && { echo "   1119 content:"; safe < $B | sed -E 's#/home/[^/"]+#/home/<user>#g'; }
  [ -e $A ] && [ -e $B ] && { echo "   diff (1118 -> 1119):"; diff <(python -m json.tool $A 2>/dev/null || cat $A) <(python -m json.tool $B 2>/dev/null || cat $B) | sed -E 's#/home/[^/"]+#/home/<user>#g' | safe | head -40; }
done
ls -la $DATA/pm_224/20261119/ 2>/dev/null | awk '{print $5, $6, $7, $8, $9}' | grep -v json5 | head -30
echo "[$(date +%T)] r175 done"
