#!/usr/bin/env bash
# Round 153 on 97, foreground, read-only on inputs. The scale check (R148) flagged stand-ins whose values are the wrong kind
# of quantity for the model's training range. For those columns: seg_fit over ALL our bases (and the ratio extras) with the
# range + threshold + root fingerprint against the production generation C -- the candidate must first land inside the
# training range, which the IC-based assignment never required. Three independent groups, one tab each, ~10-40 min each.
#  kind   the 11 columns of the wrong kind (three flags)
#  out    the 6 columns mostly outside the training range
#  scale  the 11 columns whose scale collapsed (ours covers <= 1% of the range)
#   bash research/r153.sh kind 2>&1 | tee data/reports/r153_kind.txt     (same for out, scale)
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?kind|out|scale}" in
kind)  T=zzqf7.q.c2,zzag2.c,zzwa.i,zzpe.b,zzqf5.q.c1,zzhi2.l,zzkp,zzzl2.e,zzyj3.c,zzpp.e,zzhi2.i ;;
out)   T=zzwa.g,zzro2.a,zzqf.q.c3,zzrg,zzht,zzqf2.q.c1 ;;
scale) T=zzxc.b,zzro.e,zzwa.f,zzwa.h,zzwa.k,zzqo.c,zzqo.d,zzal.h,zzaa.a,zzgh2.b,zzop.j ;;
esac
python -u research/featgen/seg_fit.py --days 20260430,20261112 --target $T --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r105/engine_{date}.parquet" --models $DATA/models_gen/C --clocks --top 5 \
  --transforms=id,-id,1/b,dm,-dm,dmi,-dmi,neu,-neu,srank,abs4,log,sqrt --scales 1,1e4,100,0.01,1e-4 2>&1 | grep -v Warning | cut -c1-240
echo "[$(date +%T)] r153 $1 done"
