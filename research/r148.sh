#!/usr/bin/env bash
# Round 148 on 97, foreground, read-only on inputs. Three independent parts, one tab each (follow-ups of R147):
#  units  unit_check: every settled stand-in's own range / scale class against the model's training range and thresholds
#         (generation C and B). Flags: class differs, >1% outside, covers <20% of the range, <50% of thresholds live. ~2 min.
#  jz     jz_fit: the families famlinks ties to engine states (zzeu zzef zzel -> zzcb; zzmo zzyg -> zzba) fingerprinted
#         against the state-formula grammar over PM's own day-end .jz keys of that family. A lead on shape, not a pin. ~5 min.
#  prof   seg_fit: the families famlinks ties to the zzxc / zzts profiles (zzpz zzoe zzrb zzso zzwf zzfe), candidates =
#         live quantity / profile cell of ZZXC and ZZTS only. ~10-40 min.
#   bash research/r148.sh units 2>&1 | tee data/reports/r148_units.txt     (same for jz, prof)
set -u
cd /work/crucible
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
case "${1:?units|jz|prof}" in
units)
  for G in C B; do
    echo "######## generation $G"
    python -u research/featgen/unit_check.py --days 20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" --models $DATA/models_gen/$G \
      --out data/reports/unit_check_$G 2>&1 | grep -v Warning | cut -c1-250
  done ;;
jz)
  DAYS=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | tail -8 | paste -sd, -)
  echo "== .jz days: $DAYS"
  python -u research/featgen/jz_fit.py --jz-dir $SEED --days $DAYS --target fam:zzeu,fam:zzef,fam:zzel --key-re '^zzcb' --models $DATA/models_gen/C 2>&1 | grep -v Warning | cut -c1-250
  python -u research/featgen/jz_fit.py --jz-dir $SEED --days $DAYS --target fam:zzmo,fam:zzyg,fam:zzba --key-re '^zzba' --models $DATA/models_gen/C 2>&1 | grep -v Warning | cut -c1-250
  echo "== control: the 4 exact columns from their own keys (zzup / zzvx.a / zzxu.a / zzxu.b should come out on top with their real formula)"
  python -u research/featgen/jz_fit.py --jz-dir $SEED --days $DAYS --target zzup,zzvx.a,zzxu.a,zzxu.b --key-re '^(zzup|zzvx|zzxu)\.' --models $DATA/models_gen/C --top 3 2>&1 | grep -v Warning | cut -c1-250 ;;
prof)
  SF="--x-dir data/reports/stage6_r99_{mmdd} --base data/featgen_r99/base_{date}.parquet --models $DATA/models_gen/B --top 4"
  for S in zzpz zzoe zzrb zzso zzwf zzfe; do
    echo "== seg_fit $S: live quantity / ZZXC, ZZTS profile cell"
    python -u research/featgen/seg_fit.py --days 20260430,20261112 --target fam:$S $SF --extra "data/side_r95/hstats_prev_{date}.parquet" --cand-re "^x\.ZZ(XC|TS)\." \
      --ratio "^x\.ZZ(XC|TS)\..*@0\$=amount|vol|turn|trade|ord|cnt" --max-ratios 4000 --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,log --scales 1,1e4,100,0.01 2>&1 | grep -v Warning | cut -c1-250
  done ;;
esac
echo "[$(date +%T)] r148 $1 done"
