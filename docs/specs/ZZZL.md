# ZZZL (cte::TermZZZL): engine spec decoded from the runtime

Sources: docs/reports_97/exec_b1_zzzl_slot5/6/7.txt, asm/ZZZL_ctor.txt, asm/HUB_loader.txt, and the 97 relay's row reads (2026-11-27).
Status: formula complete. One check is open: the index-side scale (see the end).

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
- Adjustment (stock), cumulative: scale(k) = ret_adj_SOD(D) * prod_{j=1..k} ret_adj_EOD(p_j). The hub map +0x110 = ret_adj from CEDxr4d/daily/<D>_SOD.nc; map +0x230[j] is read from CEDxr4d/daily/p<j>/<D>_EOD.nc, with variable ret_adj assumed (the string is reused, not verified). The scale multiplies X, U and S of that lag once (the in-place loop at 1393c3 is the checkpoint path, and zzzl is not loadCp).
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
- Index-side scale: whether the index pass resets the running scale and uses 399300.SZ's row (a proper index adjustment, normally 1) or carries the stock's scale (a quirk to reproduce).
- Check on 97 that p<j>/<D>_EOD.nc has V = ret_adj.
