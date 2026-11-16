#!/usr/bin/env bash
# Round 141 on 97: one family dossier per trunk, one tab per trunk (independent), read-only on inputs, outputs under data/.
# Same as R139 but with the peer-relative wrappers R139 lacked (dmi / neu from the day's CED, -dm, 0.01 in every pass);
# no dossier. Per trunk: the dossier (.jz keys + implied half-lives, hstats profile, columns with per-segment ranges / thresholds /
# roots / gain / neighbours, earlier report lines), then seg_fit on that trunk's columns with candidates from the trunk's
# own source: engine trunks -> state formulas of bit-exact keys of the stem; profile trunks -> live quantity / the stem's
# profile cell; sourceless trunks -> all bases. ~5-30 min per trunk.
#   bash research/r141.sh <stem> 2>&1 | tee data/reports/r141_<stem>.txt
#   engine:  zzeg zzvm zzmh zzcb zzba zzhj zzxo zzab zzpk zzfb zzgh zzjb     profile: zzwa zzqo zzal zzxc zzts     control: zzro zzqf
set -u
cd /work/crucible
DATA=/work/crucible_data
S=${1:?stem}
DAYS=20260430,20261112
SF="--x-dir data/reports/stage6_r99_{mmdd} --base data/featgen_r99/base_{date}.parquet --models $DATA/models_gen/B --top 5"
P=$(echo $S | tr a-z A-Z)
case $S in
zzwa|zzqo|zzal|zzxc|zzts)
  case $S in zzwa) NUM='spread|lpe|lee|rpe|half|ask|bid';; zzqo) NUM='ord|flow|refill|vol|cnt';; zzal) NUM='amount|depth|imbv|amihud|buy|sell';; *) NUM='amount|vol|turn|trade';; esac
  echo "== seg_fit $S: live quantity / $P profile cell (@0 and @-1)"
  python -u research/featgen/seg_fit.py --days $DAYS --target fam:$S $SF --extra "data/side_r95/hstats_prev_{date}.parquet" --cand-re "^x\.$P\." \
    --ratio "^x\.$P\..*@0\$=$NUM" --ratio "^x\.$P\..*@-1\$=$NUM" --max-ratios 4000 --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,log --scales 1,1e4,100,0.01 2>&1 | grep -v Warning ;;
zzro|zzqf)
  echo "== seg_fit $S: all bases (sourceless control)"
  python -u research/featgen/seg_fit.py --days $DAYS --target fam:$S $SF --extra "data/side_r105/ratio_{date}.parquet" \
    --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,abs4,log --scales 1,1e4,100,0.01 2>&1 | grep -v Warning ;;
*)
  echo "== seg_fit $S: state formulas of bit-exact $S keys"
  python -u research/featgen/seg_fit.py --days $DAYS --target fam:$S $SF --no-base --cand-re "$S" \
    --extra "data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" --exact-csv data/reports/r125/verify_20261113.csv \
    --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,log,abs4 --scales 1,1e4,100,0.01 2>&1 | grep -v Warning
  echo "== seg_fit $S: same, any $S key (leads, not pins)"
  python -u research/featgen/seg_fit.py --days $DAYS --target fam:$S $SF --no-base --cand-re "$S" \
    --extra "data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,log --scales 1,1e4,100,0.01 2>&1 | grep -v Warning ;;
esac
echo "[$(date +%T)] r141 $S done"
