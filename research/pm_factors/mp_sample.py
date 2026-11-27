r"""Raw sample records from one day of PM's .mp files, unconverted (for tickaudit's pm adapter): chosen symbols and local
times per file, plus the record count of every file. Read-only.

    python research/pm_factors/mp_sample.py /work/crucible_data/pm_mp/20261119
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_read  # noqa: E402

# file -> (sid, local hh:mm:ss windows [lo, hi), records to print per window)
WANT = {
    "SH_HS300_trade.mp": (600000, [("09:25:00", "09:25:01", 3), ("09:30:00", "09:30:30", 5)]),
    "SH_HS300_order.mp": (600000, [("09:29:30", "09:31:00", 8)]),
    "SZ_HS300_trade.mp": (1, [("09:30:00", "09:30:30", 8)]),
    "SZ_HS300_order.mp": (1, [("09:30:00", "09:30:30", 5)]),
    "SH_HS300_snap.mp": (600000, [("10:00:00", "10:00:10", 2)]),
    "SZ_HS300_snap.mp": (1, [("10:00:00", "10:00:10", 2)]),
    "SH_HS300_index.mp": (None, [("10:00:00", "10:00:10", 2)]),
    "SZ_HS300_index.mp": (None, [("10:00:00", "10:00:10", 2)]),
}


def sod(epoch: float) -> str:
    s = (epoch + 28800) % 86400
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def main() -> None:
    day = Path(sys.argv[1])
    for f in sorted(day.glob("*.mp")):
        sid, wins = WANT.get(f.name, (None, []))
        got = [0] * len(wins)
        n, sids, zero_px = 0, set(), 0
        print(f"== {f.name}")
        for r in mp_read.records(f):
            n += 1
            sids.add(r[0])
            if f.name.endswith("_trade.mp") and len(r) > 5 and r[5] == 0:
                zero_px += 1
            if not wins or (sid is not None and r[0] != sid):
                continue
            t = sod(r[1])[:8]
            for i, (lo, hi, k) in enumerate(wins):
                if lo <= t < hi and got[i] < k:
                    got[i] += 1
                    print(f"   recv {sod(r[1])} exch {sod(r[2])}  {r}")
        print(f"   records {n:,}; distinct sids {len(sids)}; sid type {type(next(iter(sids))).__name__ if sids else '-'}; e.g. {sorted(sids)[:3]}"
              + (f"; px 0 rows {zero_px:,}" if f.name.endswith("_trade.mp") else ""))
    print("[done]")


if __name__ == "__main__":
    main()
