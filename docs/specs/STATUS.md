# PM alpha reproduction: status per term (as of 2026-11-28)

Ground truth: 97 /work/pmrun/20261127/out_r1/alpha.parquet. This is PM's own runtime replaying 1127 from the DataYes shm rings. It matches PM's live alpha at corr 0.9856, 98.4% of rows within 1e-4; the residual comes from the stand-in barra files (1126 used for 1127).
Model: 4 sub-models by session (s1 to 09:40, s2 to 10:00, s3 to 14:30, s4 to 15:00), each the mean of 3 fits (livev1_fit1-3, w 1/3); 209 input columns named `<cfg term>.<output>`.
Scoreboard (1119, vs QD's 224 alpha, rB_zzelop): per-session corr s1 .200 / s2 .277 / s3 .336 / s4 .394. s1 has a +0.0023 level bias, about 2/3 of it from unassigned columns (rC pred_contrib).

State check 1126 (engine 89c588d on tickaudit's lake, DataYes source, vs PM's live end-of-day state for 1126 = the 1127 loadCp; 300 stocks, share of stocks within 1e-6 / 1e-3 averaged over keys):
- matches: zzjb, zzjb2, zzfb, zzxo (1.000 / 1.000), zzhj (0.996 / 1.000).
- feed-level (all keys within 1e-3, median ratio ~1): zzab2, zzeb, zzmh, zzpk, zzup, zzvx, zzxu; zzui (0.785 / 0.979).
- real gaps: zztx (0.001 / 0.503), zzvm (0.401 / 0.729), zzbd (0.172 / 0.767), zzba (0.333 / 0.792), zzqj (0.401 / 0.989), zzgh (0.974 flat), zzgh2 (0.945 flat), zzeg (0.994 flat). SH / SZ split pending.
- not comparable: zzcb (re-reading).

Legend: **exact** = matches PM to feed precision; **spec** = decoded from the asm, spec in this folder; **partial** = state or skeleton decoded, members open; **open** = not decoded yet.

| term / type | state | output (model columns) | notes |
|---|---|---|---|
| ZZJB / zzjb, zzjb2 | exact (948 / 1422 slots) | exact: zzjb.b / zzjb2.b = price / pre_close - 1 | |
| ZZHJ / zzhj | exact (zzhj.c, zzhj.p) | partial: [+0x100] accumulates amount x log(mid / yesterday's same-slot mid), output = [+0x100] / mean(p) | multiplier / decay to confirm |
| ZZGH / zzgh, zzgh2 | ~97% / 93% slot codes (trade set) | partial: zzgh.b = 25k + 5b + code, x = imbalance - [+0xf8] | members +0x100 / 0x140 / 0xf8 / 0x180 open |
| ZZMH / zzmh | exact (hm, zzmh.r hl 10) | spec: sqrt(mean((s dd)^2) / m), m = 3, s = 2 | ZZMH.md; key binding inferred |
| ZZOP / zzop | not built | spec: 10 outputs; i / j = ZZSQ priors a / b (h unused by the model) | ZZOP.md; prior file day / I open |
| ZZEL / zzel | constants | spec: zzel.a = ZZDS msr, zzel.b = barra SRisk (T-1) | helps s2 / s3 in rB |
| ZZZL / zzzl, zzzl2 | not built | spec: 9 outputs; model uses zzzl2.c, zzzl2.g, zzzl.f | ZZZL.md complete |
| ZZEG, ZZBA, ZZEB, ZZTX, ZZUI, ZZFB, ZZHJ.p | exact or 1e-6 (feed clock) | various; dz_eg / dz_ba from slot 7 | averages at 1e-6 = PM host clock stamp |
| ZZXO, ZZVM | state in engine | open (next for the relay) | top s1 contributors |
| ZZAG (zzag2.a/c/e/f), ZZBT, ZZAL, ZZSF, ZZWA, ZZSA, ZZPP, ZZUC, ZZJX, ... | | open | ranked by s1 contribution / gain |

Shared start-up inputs (hub loader 146e10): ret_adj from CEDxr4d/daily/<D>_SOD.nc and p1..p3; PBeta from CEDxr4d/sod/barra/<D>_stats.nc; index 399300.SZ for HS300; samplerS (ZZSQ, ZZDS, ...) from hstats/sod; samplerR init_load (ZZAC os, ZZCR ts).
