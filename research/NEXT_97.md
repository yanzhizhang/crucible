# Next on 97 (handoff, 2026-11-13)

Done since the last handoff:
- R28: seeding the engine EMAs changes nothing.
- R29: the generation-moved columns explain little, except zzup.
- R30/R31: PM's alpha follows s1 at every time of day, so keep s1.
- add_day 1111 and 1112: day-end state matches PM 1.0000. From 1112 the logs are unified and pm_signals is fixed.

The baseline is still `data/reports/r27_both.json`.

**2026-11-13 update from the user:** EVERY 224 signal is unreliable. Only PM's production on 2.23 is correct: its `.jz` files plus `livev1/rec10_p/*/alpha.parquet`, which so far exists for 0430 only. So run this first:

- `research/prod23_probe.sh` cdb5abe4b8f4836e8071de4bc3889a21 (22 lines). It is read-only on 2.23 and prints only. It lists what production writes: the dirs under prod/h/HS300, the p1..p10 and livev1 trees, and any alpha / pred / feature files.
- Run: `bash research/prod23_probe.sh 2>&1 | tee data/reports/prod23_probe.txt`
- Bring back all of its output.

Hold R32 (it scans against 224 targets). R33 (runS rules) is still fine.

The user picked two tracks: translate runS, and rescan on cleaner days with a soft family lock. Asking PM for data is skipped. All files are on feat/pm-repro and are pure ASCII. Check the LF md5 after pasting.

0. Score r27_both on 1112. It is a single run; the builds are cached.

       CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261112 --assign data/reports/r27_both.json --no-drop-harmful --max-features 0 --implied 0 --full-book /work/crucible_data/store/pm_factors/full_book_dysl2 --models /data/prod/models/tk5m --marks rand:240 --target /work/crucible_data/pm_target_r22 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats /work/crucible_data/hstats_view/HS300 --dump-pred --out data/reports/stage6_r27both_20261112 > data/reports/r27both_20261112.txt 2>&1

   Bring back the `IC: start` line and the residual_breakdown output of its pred.parquet.

1. **R33: runS on 1112.** Quick.
   - Files:
     - `research/r33.sh` 6c7162d702af61106c79ad5731a408ed (29 lines)
     - `research/pm_factors/pm_orders.py` 9e61c5cdda0f80fc1a2d13351ff945d0 (280). Plugin lines now come through chrono's logger; the old version skipped them.
   - Run: `bash research/r33.sh 20261112 2>&1 | tee data/reports/r33.txt`
   - Bring back r33.txt in full. If it is too long, send pm_orders' census and sections 2-5 first.
2. **R32: rescan.** About 2 h.
   - Files:
     - `research/r32.sh` eb25b00af4923212d778164c0a1a66c3 (52)
     - `research/pm_factors/ic_score.py` 6e15d6a6f6d25b2a479244dc629f6ed9 (1879). Adds `--family-lock-soft`.
   - Run: `bash research/r32.sh 2>&1 | tee data/reports/r32.txt`
   - What it does: from r27_both, it scans the open columns plus the soft-lock rescans on 1110 + 1112. It then gates on 0430 and tests on 1109, 1111 and 1104-1106.
   - Bring back the family-lock lines, the assign_diff, the gate table and the held-out table.
3. For future days: `research/add_day.sh` 402770117a9a68f8405997df5f9ae6f0 (79). It also scores r27_both.

## Tonight (2026-11-13), in order, all on 0430 = the one correct target
1. R34 (running): 0430-only rescan, scan grid 0, gate grid 7.
2. Paste ic_score.py 997917b11ae71f9b30eb35d7692e87f9 (1984 lines; superset with --seed-dir and --neutral), static_resid.py fbe7a8390709e710a90258ce59e306ec (68), xs_screen.py 250c406e688d064eb2a27863f94a7747 (60).
3. R35 `research/r35.sh` 8f0110b191b8a8e725a4661cf6a04e0f (24): implied values + residual breakdown + per-stock residual vs CED daily candidates.
4. R36 `research/r36.sh` 44c894522bafedcfbf1267ae6172e67d (47): seed-state candidates (zzeg channel ...) from 0429.jz, stateful open columns.
5. R37 `research/r37.sh` 35b319bd111fc1dc2554d3dfe0263b1e (55): xs_screen + neu: (daily Barra industry + style neutralisation) scan.
runS is paused until 224 runs cleanly (user).
