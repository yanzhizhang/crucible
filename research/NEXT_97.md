# Next on 97 (handoff, 2026-11-12)

The baseline is now R27's `data/reports/r27_both.json` (103 columns). R17 = 102.

| day | target | R17 | R27 both |
|---|---|---|---|
| 0430 | production (true seed) | 0.3107 | 0.3756 |
| 1109 | 224 | 0.5357 | 0.5712 |
| 1110 | 224 | 0.6958 | 0.7173 |
| 1104 / 1105 / 1106 / 1111 | 224 (stale seed) | 0.598 / 0.702 / 0.713 / 0.459 | -0.004 / -0.001 / -0.005 / -0.005 |

The rounds below are in order. All files are on feat/pm-repro. Paste them by hand and check the LF md5. They are pure ASCII.

1. **R28**: every engine EMA is seeded from a `.jz`. The configs compared are zero / true D-1 / 224's stale seed (by the 2.23 sod link times). It scores r27_both.
   - `research/r28.sh` fe033c18eec8a7f3066ebfe924afa637 (59 lines)
   - `research/engine/export.py` 6a878b1a1d01fc36409588d79698506c (107)
   - `research/engine/run_cpp.py` a542e01ae07e502db2336dd3e11093a1 (145)
   - `bash research/r28.sh 2>&1 | tee data/reports/r28.txt`
2. **R29**: gen_flags' moved columns are dropped from r27_both. Scored on 0430 (gen A) and on 1110 (gen C). Also lists 2.23 livev1/rec10.
   - `research/r29.sh` e6c5bdeeb6d2f3531ec9f11e2c58c9b4 (44)
   - `research/pm_factors/r29_moved.py` 728bfe04bfcd63330e974fc859a3bb0d (34)
   - `bash research/r29.sh 2>&1 | tee data/reports/r29.txt`
3. `bash research/add_day.sh 20261111 2>&1 | tee data/reports/add_20261111b.txt`. 1111's hstats and `.jz` should be on 2.23 by now.
4. `bash research/add_day.sh 20261112 2>&1 | tee data/reports/add_20261112.txt`. This needs the DataYes package on 97 (1111's was there about 23:50). If it stops with "no DataYes ticks", rerun later.
   - 1112 is the first clean 224 day. Its seed was 1111.jz, the correct D-1.
   - From 1112 on, the strategy logs go fully through chrono's logger in one format.
   - Bring back pm_signals' "factor lines" / "order match check" / "factor program ... corr" lines. 1110 had 0.9887 / 0.99. 1111 was broken at 0.55 / 0.03.

Bring back:
- R28: the day x config table, the half-hour tables, and the "carry seeds exported" lines.
- R29: step 0's listing and both pick tables.
- add_day: the step 4 IC lines and the tail of step 5.
