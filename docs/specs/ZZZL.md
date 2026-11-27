# ZZZL (cte::TermZZZL): engine spec decoded from the runtime

Sources: docs/reports_97/exec_b1_zzzl_slot5/6/7.txt, asm/ZZZL_ctor.txt, asm/HUB_loader.txt, and the 97 relay's row reads (2026-11-27).
Status: complete (2026-11-27).

## Config
| term | prevDays | blen | forH | other |
|---|---|---|---|---|
| zzzl | 3 | 20 | true | hl, lvlIdxList, name: defaults |
| zzzl2 | 3 | 30 | (default) | |

Index R = the universe's index: HS300 -> 399300.SZ (CSI 300 Index); lvlIdxList unset.

## Start-up (constructor + shared hub)
Day k = 1..3 is lag k: day 0 of the vectors = D-1, from a counter that starts at 1.
- Stock daily refs: X[k-1] = open(D-k), U[k-1] = close(D-k), from the stsfile daily dict (sdivDailyDict). Index: Y = open, V = close (sdivDailyIndexDict). NaN-initialised.
- Minute refs (1min_src-shaped nc, x[S, D, I, V], I = minute, I = 0 = 09:15): stock S_j = midPriceLastValid, index I_j = lastTradePrice, for lags j = 1..3.
- Adjustment (stock only), cumulative, applied once before each lag's fill to open, close and the minute midPriceLastValid: scale(L) = ret_adj(D) * prod_{j=1..L} ret_adj(p_j), all from the SOD files (hub +0x110 = CEDxr4d/daily/<D>_SOD.nc; +0x148 / +0x180 / +0x1b8 = daily/p1 / p2 / p3/<D>_SOD.nc; variable ret_adj, by stock name). Lag L uses p_1..p_L inclusive: reproduce that as is, even though a textbook chain would stop at p_{L-1}. The loop stops early when lag - 1 >= this+0x220 (the days available). The EOD map (+0x230) is read after the scale but does not enter it; its purpose is unread and not needed for these outputs. The in-place loop at 1393c3 is the checkpoint path (zzzl is not loadCp).
- Index refs (Y, V, I_j) are RAW: the index pass has no scale.
- beta = PBeta from CEDxr4d/sod/barra/<D>_stats.nc, by stock name (hub map +0x1f8).

## Updates
- Stock snapshot (slot 5), in the session gate and with t > 0: px = rec+0x60 if > 0, else the book mid (both sides >= 1e-10 and side1 >= side2 -> average, else the valid side; no book -> rec+0x58). If flag +0x224: sample(t, px) into the blen-block sampler. P = px.
- Index record (slot 6): only the record for 399300.SZ; if forH, the 1336f0(time) check must pass; t > 0; x = rec+0x60 > 0 -> sample into the index sampler; R = x.

## Output (slot 7), 9 values a..i
t = trading clock; t <= 0 -> all 0. k = floor(t / blen), required 0 <= k < cap. m = floor(t / 60) + 15.
f(Pr, Rr) = log(P / Pr) - beta * log(R / Rr); 0 if any of P, R, Pr, Rr is 0, or either log is NaN.

| out | reference |
|---|---|
| a | stock / index sampled value of the current block k |
| b | open D-1 |
| c, d, e | minute m of lag 1, 2, 3 (stock midPriceLastValid, index lastTradePrice) |
| f | close D-2 |
| g | open D-2 |
| h | close D-3 |
| i | open D-3 |

close(D-1) is never read. Model columns: zzzl2.c, zzzl2.g, zzzl.f.

## Open
- None for the outputs. The hub's EOD map (+0x230) is read but unused by ZZZL's outputs.
