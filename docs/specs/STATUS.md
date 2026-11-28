# PM alpha reproduction: status per term (as of 2026-11-28)

Ground truth: 97 /work/pmrun/20261127/out_r1/alpha.parquet. This is PM's own runtime replaying 1127 from the DataYes shm rings. It matches PM's live alpha at corr 0.9856, 98.4% of rows within 1e-4; the residual comes from the stand-in barra files (1126 used for 1127).
Model: 4 sub-models by session (s1 to 09:40, s2 to 10:00, s3 to 14:30, s4 to 15:00), each the mean of 3 fits (livev1_fit1-3, w 1/3); 209 input columns named `<cfg term>.<output>`.
Scoreboard (1119, vs QD's 224 alpha, rB_zzelop): per-session corr s1 .200 / s2 .277 / s3 .336 / s4 .394. s1 has a +0.0023 level bias, about 2/3 of it from unassigned columns (rC pred_contrib).

State check, corrected (engine 89c588d; seed = PM's end of D-1, reference = PM's live end of D; seed_probe/<D>.jz is named by END day, prodfiles sod/<D>.jz = start of D = end of D-1). 1126 on tickaudit's lake (DataYes), 1119 on PM's .mp feed recording (close to, not identical with, PM's production feed). Share of 300 stocks within 1e-6 / 1e-3:
- matches: zzxo (1.000 both days), zzhj (.998), zzjb / zzjb2 (1.000 lake; .88-.94 on the .mp cache = feed), zzfb (1.000 on PM's feed; .333 on the lake, which lacks its index input).
- feed-level (all keys within 1e-3 on the PM-feed day): zzab2, zzeb, zzpk, zzup, zzvx, zzmh, zzqj, zzba (SH .67 on the lake, 1.000 on PM's feed).
- real gaps (same pattern on both days and feeds): zztx (SH .37-.39 / SZ .74-.75), zzvm (SH .62-.72 / SZ .91-.97), zzbd (.61-.77; drSumS / drSumT not produced), zzui (~.91), zzeg (high / low / span exact, rEma wrong), zzxu (.75), zzgh / zzgh2 (.96 / .93). Trade-clock wgt PM / ours ~1.005 SH, 1.0007 SZ: PM ticks on slightly more events.
- not comparable: zzcb (not produced); ZZOP, ZZZL, ZZEL not in the engine.
- replay 2's dump.jz (/work/pmrun/20261127/out) = the LOADED state (end of 1126), not end of 1127: no PM end-of-1127 reference on 97.

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
