"""CED jobs (port of shtcommon ``scripts/daily_job.py``): one dataset per call, Parquet output.

    python research/run_ced.py daily-sod                       # [T-1, T], missing days only
    python research/run_ced.py daily-sod --hist --start 20250101 --end 20260430
    python research/run_ced.py all-hist --start 20250101 --end 20260430 --overwrite
    python research/run_ced.py check-hist --date 20260429
    python research/run_ced.py calendar        # refresh the Wind calendar cache (SSE + SZSE)

Default range is [T-1, T] (T = today) and days already written are kept unless ``--overwrite``;
on a non-trading day (Wind calendar) it exits 0 unless ``--force``. ``--hist`` selects the
after-close basis for daily-sod and index-universe. ``all-hist`` runs every builder on the
authoritative basis for the range -- the research rebuild ("hist convert") -- then the checks.

Environment: ``CRUCIBLE_WIND_URL`` / ``CRUCIBLE_JY_URL`` / ``CRUCIBLE_ZY_URL`` or
``CRUCIBLE_SHTCOMMON`` (database URLs, see ``research/ced/db.py``), ``CRUCIBLE_DATA`` (default
``/work/crucible_data``), ``CRUCIBLE_BARRA_REPO`` (default ``/mnt/BarraDataRepo/BarraCNE6``).
Log: stderr and ``$CRUCIBLE_DATA/logs/ced.log``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ced import barra, check, concept, daily, index_eod, index_universe, industry, st, store, wind_div, zy_div  # noqa: E402
from ced.calendar import calendar  # noqa: E402

log = logging.getLogger("ced.run")


def _sod(s: str, e: str, *, overwrite: bool, hist: bool) -> dict:
    return (daily.convert_range_sod_hist if hist else daily.convert_range_sod)(s, e, overwrite=overwrite)


def _universe(s: str, e: str, *, overwrite: bool, hist: bool) -> dict:
    return (index_universe.convert_range_hist if hist else index_universe.convert_range)(s, e, overwrite=overwrite)


# job -> (runner(start, end, overwrite=..., [hist=...]), supports --hist)
RUNNERS = {
    "daily-sod": (_sod, True),
    "daily-eod": (daily.convert_range_eod, False),
    "index-eod": (index_eod.convert_range, False),
    "index-universe": (_universe, True),
    "st": (st.convert_range, False),
    "wind-div": (wind_div.convert_range, False),
    "zy-div": (zy_div.convert_range, False),
    "sw-industry": (industry.convert_range_sw, False),
    "wind-industry": (industry.convert_range_wind, False),
    "concept": (concept.convert_range, False),
    "barra": (barra.convert_range, False),
    "check-div": (check.check_div_range, False),
    "check-hist": (check.check_hist_range, False),
    "check-index": (check.check_index_range, False),
}
# the research rebuild: every dataset on the after-close basis, then the checks that apply
ALL_HIST = ("daily-sod", "daily-eod", "index-eod", "index-universe", "st", "wind-div", "zy-div",
            "sw-industry", "wind-industry", "concept", "barra", "check-div")


def setup_logging() -> Path:
    path = store.DATA / "logs" / "ced.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(fmt)
        root.addHandler(h)
    return path


def run_job(job: str, start: str, end: str, *, overwrite: bool, hist: bool) -> dict:
    runner, supports_hist = RUNNERS[job]
    kw = {"overwrite": overwrite}
    if supports_hist:
        kw["hist"] = hist
    elif hist and not job.startswith("check"):
        log.warning("%s has no --hist, ignored", job)
    return runner(start, end, **kw)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("job", choices=(*RUNNERS, "all-hist", "calendar"))
    ap.add_argument("--date")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--hist", action="store_true", help="after-close basis (daily-sod / index-universe)")
    ap.add_argument("--overwrite", action="store_true", help="rewrite days already written")
    ap.add_argument("--force", action="store_true", help="run on a non-trading day too")
    a = ap.parse_args(argv)
    log.info("log -> %s", setup_logging())
    if a.job == "calendar":
        from ced.calendar import CACHE, SSE, SZSE

        calendar.invalidate()
        for ex in (SSE, SZSE):
            days = calendar.trade_days("19900101", "29991231", ex)
            log.info("calendar %s: %d days %s..%s -> %s", ex, len(days), days[0], days[-1],
                     CACHE / f"exchange={ex}.parquet")
        return 0

    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d")
    if a.start and a.end:
        start, end = a.start, a.end
    elif a.date:
        start = end = a.date
    else:
        if not a.force and not calendar.is_trade_day(today):
            log.info("%s is not a trading day, nothing to do", today)
            return 0
        start, end = calendar.prev(today, 1), today

    jobs = ALL_HIST if a.job == "all-hist" else (a.job,)
    hist = a.hist or a.job == "all-hist"
    rc = 0
    for job in jobs:
        log.info("%s: %s ~ %s (hist=%s, overwrite=%s)", job, start, end, hist, a.overwrite)
        t0 = time.perf_counter()
        try:
            out = run_job(job, start, end, overwrite=a.overwrite, hist=hist)
        except Exception:
            log.exception("%s failed", job)
            rc = 2
            continue
        if not out:
            log.warning("%s: %s ~ %s produced no trading day, check the upstream source", job, start, end)
            rc = max(rc, 1)
            continue
        log.info("%s done: %d trading days in %.1fs", job, len(out), time.perf_counter() - t0)
    return rc


if __name__ == "__main__":
    sys.exit(main())
