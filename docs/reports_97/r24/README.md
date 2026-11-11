# R24 round, 97 outputs (2026-11-11 evening)

Transcribed by hand from screen zooms of the VDI VS Code on 97 (not byte-exact; no md5 match).
Table borders dropped; empty cells written as "-". Each file is also relayed verbatim to the crucible session.

Run on 97 at 23:09 (VDI time), crucible feat/pm-repro @ 293ba97, all 9 file md5s verified on 97 first:
ic_score.py 0871dcde, seed_probe.py eae82648, state_keys.py 23ad2c0f, residual_breakdown.py 695d46bd,
gen_compare.py 6a22e85d, r24.sh 79e7a979, seed_probe.sh 76f57bac, static_probe.sh 95156d5a, r23_diag.sh 73de9e65.
Chain: seed_probe -> static_probe -> r24 -> static_probe (after r24) -> r23_diag.

| file | source on 97 | notes |
|---|---|---|
| seed_probe.txt | data/reports/seed_probe.txt (209 lines) | 2.23 table: 71 rows identical to 97's, given as one line |
| static_probe.txt | data/reports/static_probe.txt (123 lines) | |
| state_keys_r17_keyblocks.txt | data/reports/state_keys_r17.txt (253 lines) | only the 10 stems with state keys (crucible asked for these) |
| r24_progress.txt | terminal | partial r24.txt while running |

## Facts seen (raw, no analysis)
- PM .jz state chain on 97: 71 files 20230116..20261106, all 300 sids, 7498 keys (20260429: 7532). No links, no duplicate md5s.
- 2.23 has the same 71 files with the same md5s, plus 20261109 (7820c626) and 20261110 (50f7dd68).
- tk5m holds only model dirs (202509, 202601.updated, 202609; fit1-3); no other files.
- 2.23:/data/share/s = ex/ index/ (HDF5 .xr per day), sid2symbol/ (json per day), stats/.
- Open columns at R17 by input location: hstats 12 cols (gain 0.0345), neither 85 (0.3107), state 10 (0.1002).
- R24 step 2 (pooled CDF vs own-generation CDF, IC): 0430 0.3105 / 0.3107, 1109 0.5366 / 0.5357, 1110 0.7024 / 0.6958.
