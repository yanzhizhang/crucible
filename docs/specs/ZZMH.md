# ZZMH (cte::TermZZMH): engine spec decoded from the runtime

Sources: docs/reports_97/exec_b1_zzba_zzmh_slot7.txt (slot 7), asm/ZZMH_ctor.txt (relay's notes), cfg term[57] `zzmh hl=10 m=3 s=2`, R124b / R193 (state), r247 (hl 10).
Status: complete. The key -> member binding is inferred (see below).

## Config / constructor
- hl = 10 (cfg; ctor default 300). The state average zzmh.r uses it.
- [+0x100] = C, a cfg double, default 1152. [+0x108] = a cfg double, default 100.
- Binding, inferred: +0x100 <- "m" (= 3), +0x108 <- "s" (= 2). Evidence: the update's value was measured as 4 dd^2 = (2 dd)^2 (R124b: PM / dd^2 = 4.0000 on 294 stocks), i.e. (s * dd)^2 with s = 2. The key strings at f8d9b / f8df5 are not yet read byte-exact.
- loadCp keys: r.wgt, r.sum, r.t, hm.

## State (already exact in the engine)
- hm = highest snapshot price from 09:30 exchange time, with no end (the closing snapshot counts).
- zzmh.r = the shared average (hl 10) of (s * (hm - p) / hm)^2 on every snapshot.

## Output (slot 7), 1 value: model column "zzmh"
t = trading clock; t <= 0 -> 0.
m_r = zzmh.r.sum / zzmh.r.wgt (0 when wgt == 0); y = m_r / C.
zzmh = y == 0 ? 0 : sign(y) * sqrt(|y|). With m = 3 and s = 2: sqrt(mean((2 dd)^2) / 3).
