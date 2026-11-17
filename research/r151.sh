#!/usr/bin/env bash
# Round 151 on 97, foreground, read-only. Two independent parts.
#  keys  R149's day-end check compared nothing (every stock had a different key set). Show the key-set difference between
#        the old data/engine/engine_20261113.json.gz and the new data/engine_fam one, then compare every SHARED key.
#  sod   runs_sod: (1) the decision log's per-stock constants vr / poscst against PM's start-of-day table (stsfile stats/sod
#        msr / mb / vr, index weights); (2) the shape of vpen between 13:00 (rvts) and 14:45. ~1 min.
#   bash research/r151.sh keys 2>&1 | tee data/reports/r151_keys.txt
#   bash research/r151.sh sod  2>&1 | tee data/reports/r151_sod.txt
set -u
cd /work/crucible
case "${1:?keys|sod}" in
keys)
  ls -la data/engine/engine_20261113.json.gz data/engine_fam/engine_20261113.json.gz 2>&1 | cut -c1-160
  python - <<'PY'
import gzip, json, re
a = json.load(gzip.open("data/engine/engine_20261113.json.gz")); b = json.load(gzip.open("data/engine_fam/engine_20261113.json.gz"))
sid = sorted(a)[0]
ka, kb = set(a[sid]), set(b[sid])
grp = lambda ks: sorted({re.sub(r"\.\d+$", ".#", k) for k in ks})
print(f"stock {sid}: old {len(ka)} keys, new {len(kb)} keys, shared {len(ka & kb)}")
print("  only in old:", grp(ka - kb)[:40])
print("  only in new:", grp(kb - ka)[:40])
n = bad = 0
badk = {}
for s, kv in a.items():
    kn = b.get(s, {})
    for k, v in kv.items():
        if k in kn:
            n += 1
            if not ((v is None and kn[k] is None) or v == kn[k]):
                bad += 1
                badk[re.sub(r"\.\d+$", ".#", k)] = badk.get(re.sub(r"\.\d+$", ".#", k), 0) + 1
print(f"shared keys: {n:,} values compared over {len(a)} stocks, {bad} differ")
for k, c in sorted(badk.items(), key=lambda kv: -kv[1])[:20]:
    print(f"   {k}: {c}")
PY
  ;;
sod)
  DAYS=$(ls -d /work/crucible_data/pm_orders/date=* 2>/dev/null | sed 's/.*date=//' | tail -2 | paste -sd, -)
  python -u research/pm_factors/runs_sod.py --orders /work/crucible_data/pm_orders --dates "$DAYS" --sts /work/crucible_data/pm_stsfile 2>&1 | grep -v -E "Warning|warn" | cut -c1-230 ;;
esac
echo "[$(date +%T)] r151 $1 done"
