r"""Which stsfile day 224's PM process loaded, per log and per trading day, from the ``/tmp/s/stsfile/...`` paths it logs.

224 does not always load the trading day's own file (on 1111 every run loaded 20261109): ``ic_score --pm-sts-map`` takes
the json written here so our PM-side inputs (stats/sod msr / mb / vr, index weights, exchange dummies) match what 224 used.
A day's entry comes from the log with the most ``trk SIG`` lines (the run that made most of the day's predictions).

    python research/pm_factors/sts_days.py --logs /work/crucible_data/chrono_logs --out data/reports/sts_map_224.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PATH = re.compile(r"/tmp/s/stsfile/(stats/sod|index|ex|sid2symbol)/(?:index-|ex-)?(\d{8})")
NAME = re.compile(r"^(\d{4})-(\d\d)-(\d\d)_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from", dest="start", default="20261101")
    a = ap.parse_args()
    best: dict[str, tuple[int, str]] = {}
    for f in sorted(a.logs.glob("20*.log")):
        m = NAME.match(f.name)
        if not m:
            continue
        day = "".join(m.groups())
        if day < a.start:
            continue
        loaded: dict[str, set[str]] = {}
        sig = 0
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "trk SIG" in line:
                    sig += 1
                if "/tmp/s/stsfile/" in line:
                    for sub, d in PATH.findall(line):
                        loaded.setdefault(sub, set()).add(d)
        days = sorted({d for v in loaded.values() for d in v})
        print(f"{f.name}: day {day}, trk SIG {sig:,}, stsfile days {days or '-'} " + " ".join(f"{k}={sorted(v)}" for k, v in sorted(loaded.items())))
        if days and sig > best.get(day, (-1, ""))[0]:
            best[day] = (sig, max(days))
    mp = {d: v for d, (_, v) in sorted(best.items())}
    stale = {d: v for d, v in mp.items() if v != d}
    print(f"\nmap ({len(mp)} days): {mp}\nnot the day's own file: {stale or 'none'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(mp, indent=1))


if __name__ == "__main__":
    main()
