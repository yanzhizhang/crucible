# ZZOP (cte::TermZZOP): engine spec decoded from the runtime

Sources: docs/reports_97/exec_b1_zzop_slot3_p1/p2.txt, exec_b1_zzop_slot7.txt, asm/ZZOP_ctor.txt (2026-11-27).
Status: formula complete. One input is open: which ZZSQ cell (day, I) the prior loader reads.

## State
Three averages of the shared rule (w = w*d + 1, s = s*d + x, d = 2^(-dt/hl)), all with hl = cfg, default 300:
C = Ema +0x108 (all trades), A = Ema +0x138 (buy), B = Ema +0x168 (sell).

## Start-up
Each average gets one update at t = 0 with its prior from ZZSQ samplerS: C <- "ab", A <- "a", B <- "b". A NaN prior is skipped, which leaves that average unseeded. h / i / j = the three priors, constant all day (zzop.i = A's prior "a").

## Update (slot 3), every trade inside the session gate
t = trading clock(rec+0x18).
- side byte rec+0x50 == 1: o = the buy order (rec+0x28; 0 -> return), looked up in the order table (not found -> return, and no update at all). x = rec+0x20 - o+0x18 (the order's age at the fill). A.update(t, x); last_A = x - mean(A).
- side == 0xff: the same with the sell order (rec+0x30) and B; last_B = x - mean(B).
- other side: x = 0.
- Then always (unless one of the returns above was taken): C.update(t, x); last_C = x - mean(C).
Means are taken after the update; mean = s / w, or x itself when w == 0.

## Output (slot 7), 10 values a..j
a = mean(A), b = mean(B), c = mean(C) (each 0 when w == 0);
d = last_A / a, e = last_B / b, f = last_C / c (0 when the denominator is 0);
g = (a == 0 && b == 0) ? 0 : (b - a) / (a + b);
h = prior ab, i = prior a, j = prior b.

## Open
- The prior loader (cd040 / 1367b0): the file day (D vs D-1) and the I index inside ZZSQ samplerS. r230's guess (I = 0 of D-1) did not match.
- The units of rec+0x20 / order+0x18 (exchange time?).
