#!/usr/bin/env bash
# Is part of PM's input static (industry / financials / neutralisation) and outside hstats? Read-only.
#   1 model columns that look like codes or counts: categorical, or integer thresholds over a small range
#     (from stage5/fingerprint.parquet), with R17's assignment next to each
#   2 what else ships next to the LightGBM files in /data/prod/models/tk5m (coefficient tables, maps)
#   3 a sample of 2.23:/data/share/s (224's stsfile): type, size, first lines of the newest files
#   4 the columns still open, grouped by where their family's inputs live: state (.jz keys) / hstats / neither;
#     at R17 and, when it has run, at R24
#   bash research/static_probe.sh 2>&1 | tee data/reports/static_probe.txt
set -u
cd /work/crucible
R=data/reports
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
S="ssh -p $P23 -o BatchMode=yes $H23"
say() { echo "[$(date +%T)] $*"; }

say "1 code / count / categorical columns in the tk5m models"
python - <<'EOF'
import json
import polars as pl
f = pl.read_parquet("data/reports/stage5/fingerprint.parquet")
a = json.load(open("data/reports/stage6_r17/ic_assign.json"))
print("fingerprint columns:", f.columns)
span = pl.col("hi") - pl.col("lo")
sel = f.filter(pl.col("lo").is_null() | (pl.col("half_int") > 0.9) | ((span <= 64) & (pl.col("half_int") > 0.5))
               | pl.col("kind").is_in(["flag", "count", "const"]))
sel = sel.with_columns(pl.col("feature").map_elements(lambda x: str(a.get(x, "-"))[:70], return_dtype=pl.String).alias("r17"))
cols = [c for c in ("feature", "kind", "lo", "hi", "distinct", "half_int", "at_zero", "gain_share", "r17") if c in sel.columns]
with pl.Config(tbl_rows=-1, tbl_width_chars=220, fmt_str_lengths=70, float_precision=4):
    print(sel.sort("gain_share", descending=True).select(cols))
EOF

say "2 non-model files under /data/prod/models/tk5m"
find /data/prod/models/tk5m -type f ! -name 'model_s*.txt' -printf '%TY-%Tm-%Td %s %p\n' 2>/dev/null | sort -k3 | head -60
find /data/prod/models/tk5m -maxdepth 3 -type d | head -40

say "3 sample of 2.23:/data/share/s"
$S 'cd /data/share/s && ls -la --time-style=full-iso | tail -15; echo; for f in $(ls -t | head -3); do echo "== $f"; if [ -d "$f" ]; then ls -la "$f" | head -15; f2=$(ls -t "$f" | head -1); [ -n "$f2" ] && f="$f/$f2" && echo "== $f"; fi; [ -f "$f" ] || continue; file "$f"; case "$(file -b "$f")" in *gzip*) zcat "$f" | head -c 1500 ;; *text*|*CSV*|*JSON*) head -c 1500 "$f" ;; *) od -A d -c "$f" | head -8 ;; esac; echo; done' \
  > $R/static_probe_s.txt 2>&1 || say "  2.23 failed: $(tail -1 $R/static_probe_s.txt)"
cat $R/static_probe_s.txt | cut -c1-240
say "4 open columns by where their inputs live (state / hstats / neither)"
HV=/work/crucible_data/hstats_view/HS300; [ -d $HV ] || HV=/data/prod/hstats/HS300
for tag in r17 r24; do
  A=$R/stage6_$tag/ic_assign.json; [ -e $A ] || { echo "  $tag: no $A"; continue; }
  python -u research/pm_factors/state_keys.py --state /data/prod/h/HS300/p1_intrading_dcp/20261106.jz --assign $A --hstats $HV     --out $R/state_keys_$tag.csv > $R/state_keys_$tag.txt 2>&1 || say "  $tag FAILED: $(tail -3 $R/state_keys_$tag.txt)"
  echo "  $tag:"; head -1 $R/state_keys_$tag.txt; sed -n '/^open columns by where/,$p' $R/state_keys_$tag.txt | cut -c1-240
done
say "done"
