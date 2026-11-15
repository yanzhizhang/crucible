#!/usr/bin/env bash
# Round 129 on 97, foreground, read-only, writes only data/reports. ~10-20 min.
# R128: the residual (PM - ours) follows short returns everywhere (+0.2..0.39) and, 5-15 min after the open, ret1800 /
# mret1800 at +0.675 -- a return reaching back over the open, i.e. against the previous close. The zzzl family is exactly
# that (settled: zzzl2.b = -lpes2400, zzzl2.d = lpes600, zzzl2.f = lpes28800, zzzl2.i = -lpes1200, zzzl.d = lpes14400) and
# three of its columns are open (zzzl2.c, zzzl2.g, zzzl.f): try every remaining lpes window, both signs, on the full grid.
#   bash research/r129.sh 2>&1 | tee data/reports/r129.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -c "import polars as pl; print('lpes / ret bases:', sorted(c for c in pl.read_parquet_schema('data/full_0430/base_20260430.parquet') if c.startswith(('lpes', 'ret', 'mret', 'overnight'))))"
python -u research/featgen/single_fill.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" --extra "data/full_0430/extra_{date}.parquet"   --fills "zzzl2.c:id:lpes300,zzzl2.c:-id:lpes300,zzzl2.c:id:lpes900,zzzl2.c:-id:lpes900,zzzl2.c:id:lpes1800,zzzl2.c:-id:lpes1800,zzzl2.c:id:lpes3600,zzzl2.c:-id:lpes3600,zzzl2.c:id:lpes7200,zzzl2.c:-id:lpes7200,zzzl2.c:id:ret1800,zzzl2.c:-id:ret1800,zzzl2.c:id:mret1800,zzzl2.c:-id:mret1800,zzzl2.c:id:overnight_gap_log,zzzl2.c:-id:overnight_gap_log,zzzl2.g:id:lpes300,zzzl2.g:-id:lpes300,zzzl2.g:id:lpes900,zzzl2.g:-id:lpes900,zzzl2.g:id:lpes1800,zzzl2.g:-id:lpes1800,zzzl2.g:id:lpes3600,zzzl2.g:-id:lpes3600,zzzl2.g:id:lpes7200,zzzl2.g:-id:lpes7200,zzzl2.g:id:ret1800,zzzl2.g:-id:ret1800,zzzl2.g:id:mret1800,zzzl2.g:-id:mret1800,zzzl2.g:id:overnight_gap_log,zzzl2.g:-id:overnight_gap_log,zzzl.f:id:lpes300,zzzl.f:-id:lpes300,zzzl.f:id:lpes900,zzzl.f:-id:lpes900,zzzl.f:id:lpes1800,zzzl.f:-id:lpes1800,zzzl.f:id:lpes3600,zzzl.f:-id:lpes3600,zzzl.f:id:lpes7200,zzzl.f:-id:lpes7200,zzzl.f:id:ret1800,zzzl.f:-id:ret1800,zzzl.f:id:mret1800,zzzl.f:-id:mret1800,zzzl.f:id:overnight_gap_log,zzzl.f:-id:overnight_gap_log" --models $DATA/models_gen/B --work data/single_fill_r129 2>&1 | grep -v RuntimeWarning | cut -c1-250
echo "[$(date +%T)] done"
