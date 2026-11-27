#!/usr/bin/env bash
# r230: score the decoded candidates through PM's models: r27both alone, then with each candidate column on 1119 / 1120.
# Read-only on the data; writes only under data/reports/r230. One process at a time.
set -u
cd /work/crucible
R=data/reports; DATA=/work/crucible_data; O=$R/r230; mkdir -p $O
SEED=$DATA/seed_probe/HS300; HV=$DATA/hstats_view/HS300; FBD=$DATA/store/pm_factors/full_book_dysl2
M="--models $DATA/prod_models/tk5m --marks rand:240 --target $DATA/pm_target_r22 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
python - <<'PY'
import json, pathlib
o = pathlib.Path("data/reports/r230")
base = json.load(open("data/reports/r27_both.json"))
v = {"base": {}, "state3": {"zzeg": "cal:gen.dz_eg", "zzba": "cal:gen.dz_ba", "zzmh": "cal:gen.dz_mh"}}
for x in ("ab", "a", "b"):
    v[f"op_i_{x}"] = {"zzop.i": f"cal:ced.dz.ZZSQ.{x}"}
for x in ("ms", "msr", "mb", "vr", "o", "h", "l", "c"):
    v[f"el_b_{x}"] = {"zzel.b": f"cal:ced.dz.ZZDS.{x}"}
for x in ("SRet", "Yield", "TRisk", "SRisk", "HBeta", "PBeta", "Close", "Capt", "Ret", "EstU"):
    v[f"el_a_{x}"] = {"zzel.a": f"cal:ced.dz.barra.{x}"}
for k, add in v.items():
    a = dict(base)
    a.update(add)
    (o / f"{k}.json").write_text(json.dumps(a, indent=1))
print(len(v), "assignments; base has", len(base), "columns; already in base:", [c for c in ("zzeg", "zzba", "zzmh", "zzop.i", "zzel.a", "zzel.b") if c in base])
PY
for A in $O/*.json; do
  n=$(basename $A .json)
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261119,20261120 --dcp-dir $SEED --assign $A $SCORE --full-book $FBD $M \
    --out $O/out_$n > $O/$n.txt 2>&1
  echo "$n $(grep -E 'IC: start' $O/$n.txt | tail -1 | sed -E 's/.*IC: start ([0-9.]+).*/\1/') err=$(grep -cE 'Traceback|Error' $O/$n.txt)"
done | sort -k2 -r | tee $O/summary.txt
grep -h "decoded start-up constants" $O/base.txt | head -2
