#!/usr/bin/env bash
# Round 22b on 97, foreground: R22 again after the 0430 fix. Feitu's 20260430 snapshots carry no price limits for 189 of
# the 299 stocks, so the whole-book replay skipped them; bt.events now takes pre_close +-30 % there (EVENTS_VERSION 3).
# Rebuilds the whole-book factors, rescores 0430 with R17 / R20 / R21, reruns the scan on 0430 + 1104-1106 from R21 and
# scores it on 1109 / 1110. Outputs *_r22b; R22's stay. Writes under /work only.
#   bash research/r22b_alpha.sh 2>&1 | tee data/reports/r22b.txt
set -u
cd /work/crucible
R=data/reports
D=/work/crucible_data
ALPHA=$D/pm_livev1/rec10_p/20260430/alpha.parquet
TGT=$D/pm_target_r22  # symlinks: PM's alpha for 0430, the 224 signals for the November days
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --implied 0 --state-fallback"

say "0 inputs"
ls -la /data/prod/h/HS300/p1_intrading_dcp/20260429.jz /data/prod/h/HS300/p1_intrading_dcp/20261106.jz
for k in order transaction quotation index; do echo "  feitu_raw $k 20260430: $(ls $D/feitu_raw/kind=$k/date=20260430 2>/dev/null | wc -l) files"; done
ls /data/prod/models/tk5m/202604/lgb 2>&1 | head -3
python -u research/pm_factors/alpha_target.py --alpha $ALPHA --out $D/pm_alpha
mkdir -p $TGT
ln -sf $D/pm_alpha/date=20260430.parquet $TGT/
for d in 20261104 20261105 20261106 20261109 20261110; do ln -sf $D/pm_signals/date=$d.parquet $TGT/; done
ls -la $TGT | tail -6
python -u research/catalog.py > $R/r22b_catalog.txt 2>&1 || say "  catalog FAILED: $(tail -2 $R/r22b_catalog.txt)"
python -c "import duckdb; c=duckdb.connect('$D/catalog.duckdb', read_only=True); print('  feitu catalog 20260430 quotes', c.sql('SELECT count(*), count(DISTINCT symbol_id) FROM raw_quotation WHERE date = 20260430').fetchall())"

say "1 engine state at the marks for 0430, if the November days had one"
if ls data/engine/marks_2026110[456].parquet >/dev/null 2>&1; then
  python -u research/engine/run_cpp.py --date 20260430 --universe pm:/data/prod/h/HS300/p1_intrading_dcp/20260429.jz \
    --prev /data/prod/h/HS300/p1_intrading_dcp/20260429.jz --marks rand:240 --threads 32 > $R/r22b_engine.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/r22b_engine.txt)"
  tail -3 $R/r22b_engine.txt
else
  echo "  no data/engine/marks_202611*.parquet: rounds so far ran on the Python rebuild alone, so 0430 does too"
fi

say "2 whole-book factors for 0430 again (1104-1106 had limits in every snapshot)"
python -u research/pm_factors/full_book.py --dates 20260430 --signals $D/pm_alpha --marks rand:240 --workers 1 2>&1 | grep -v "^1 days"
FBF=$D/store/pm_factors/full_book
FBD=$D/store/pm_factors/full_book_dysl2

say "3 0430 held out again: R17 / R20 / R21 / R22 (R22 was picked partly on 0430: in-sample)"
for AS in r17 r20 r21 r22; do  # 1109 / 1110 are unchanged by the fix: R22's numbers stand
  A=$R/stage6_$AS/ic_assign.json
  python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A --no-drop-harmful --max-features 0 --target $TGT --full-book $FBF $M --out $R/stage6_r22b_0430_$AS > $R/r22b_0430_$AS.txt 2>&1
  say "  $AS: 0430 (PM alpha) $(ic $R/r22b_0430_$AS.txt)"
done
grep -hE "Error|Traceback|no state file|PM daily inputs|stocks x .* marks|build read from cache" $R/r22b_0430_r17.txt | cut -c1-240

say "4 scan on 0430 + 1104-1106 (feitu) from R21, all columns still unmapped"
U=$(python -c "import json, polars as pl; a=json.load(open('$R/stage6_r21/ic_assign.json')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  unmapped at R21: $(echo $U | tr , '\n' | wc -l)"
python -u research/pm_factors/ic_score.py --dates 20260430,20261104,20261105,20261106 --assign $R/stage6_r21/ic_assign.json --calibrate --scan-feats "$U" \
  --target $TGT --full-book $FBF $M --out $R/stage6_r22b > $R/r22b_scan.txt 2>&1
grep -E "Error|Traceback|^dropped|IC: start|stocks x .* marks" $R/r22b_scan.txt | cut -c1-240
python -c "import json; a=json.load(open('$R/stage6_r22b/ic_assign.json')); b=json.load(open('$R/stage6_r21/ic_assign.json')); print('  ', len(b), '->', len(a)); [print('  ', k, ':', b.get(k), '=>', v) for k, v in a.items() if b.get(k) != v]; [print('  ', k, ':', v, '=> dropped') for k, v in b.items() if k not in a]"
say "5 the R22 assignment on 1109 / 1110 (held out)"
for d in 20261109 20261110; do
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $d --assign $R/stage6_r22b/ic_assign.json --no-drop-harmful --max-features 0 --target $TGT --full-book $FBD $M --out $R/stage6_r22b_${d}_r22 > $R/r22b_${d}_r22.txt 2>&1
done
say "  r22b: 1109 $(ic $R/r22b_20261109_r22.txt), 1110 $(ic $R/r22b_20261110_r22.txt)"
say "done"
