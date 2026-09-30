"""A-share trading calendar from Wind ``dbo.ASHARECALENDAR`` (same API and semantics as shtcommon).

Source of truth is Wind (``TRADE_DAYS`` per ``S_INFO_EXCHMARKET``). Each load writes a Parquet
cache ``store/ced/calendar/exchange=<EX>.parquet``; without a database URL the cache is used,
so the calendar works offline and ``almanac`` can read the same file.

``prev(d, n)`` is the n-th trading day strictly before ``d`` and ``next(d, n)`` strictly after,
whether or not ``d`` itself trades; ``offset`` requires ``d`` to be a trading day.
"""

from __future__ import annotations

import bisect
import logging
import threading

import pandas as pd

from ced import db
from ced.store import STORE

SSE, SZSE, BSE, SZN, SHN = "SSE", "SZSE", "BSE", "SZN", "SHN"
CACHE = STORE / "calendar"
log = logging.getLogger(__name__)


class AShareCalendar:
    """Trading days per exchange, loaded once per process."""

    def __init__(self) -> None:
        self._days: dict[str, list[str]] = {}
        self._sets: dict[str, frozenset[str]] = {}
        self._lock = threading.Lock()

    def _fetch(self, exchange: str) -> list[str]:
        try:
            df = db.read_sql(
                "SELECT TRADE_DAYS AS trade_days FROM dbo.ASHARECALENDAR "
                f"WHERE S_INFO_EXCHMARKET = '{exchange}' ORDER BY TRADE_DAYS ASC",
                what="ASHARECALENDAR",
            )
        except RuntimeError:  # no URL configured: offline, use the cache
            path = CACHE / f"exchange={exchange}.parquet"
            if not path.exists():
                raise
            log.info("calendar %s: no database configured, using cache %s", exchange, path)
            return pd.read_parquet(path)["trade_day"].tolist()
        if df.empty:
            raise ValueError(f"Wind ASHARECALENDAR returned nothing for exchange={exchange!r}")
        days = sorted(df["trade_days"].astype(str).str.strip().tolist())
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"trade_day": days}).to_parquet(CACHE / f"exchange={exchange}.parquet", index=False)
        return days

    def _ensure(self, exchange: str) -> list[str]:
        if exchange not in self._days:
            with self._lock:
                if exchange not in self._days:
                    days = self._fetch(exchange)
                    self._sets[exchange] = frozenset(days)
                    self._days[exchange] = days
        return self._days[exchange]

    def is_trade_day(self, date: str, exchange: str = SSE) -> bool:
        self._ensure(exchange)
        return date in self._sets[exchange]

    def trade_days(self, start: str, end: str, exchange: str = SSE) -> list[str]:
        days = self._ensure(exchange)
        return days[bisect.bisect_left(days, start):bisect.bisect_right(days, end)]

    def prev(self, date: str, n: int = 1, exchange: str = SSE) -> str:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        days = self._ensure(exchange)
        idx = bisect.bisect_left(days, date) - n
        if idx < 0:
            raise ValueError(f"fewer than {n} {exchange} trading days before {date}")
        return days[idx]

    def next(self, date: str, n: int = 1, exchange: str = SSE) -> str:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        days = self._ensure(exchange)
        idx = bisect.bisect_right(days, date) + n - 1
        if idx >= len(days):
            raise ValueError(f"fewer than {n} {exchange} trading days after {date}")
        return days[idx]

    def offset(self, date: str, n: int, exchange: str = SSE) -> str:
        if not self.is_trade_day(date, exchange):
            raise ValueError(f"{date} is not a {exchange} trading day")
        if n == 0:
            return date
        return self.next(date, n, exchange) if n > 0 else self.prev(date, -n, exchange)

    def count(self, start: str, end: str, exchange: str = SSE) -> int:
        return len(self.trade_days(start, end, exchange))

    def nearest_prev(self, date: str, exchange: str = SSE) -> str:
        return date if self.is_trade_day(date, exchange) else self.prev(date, 1, exchange)

    def nearest_next(self, date: str, exchange: str = SSE) -> str:
        return date if self.is_trade_day(date, exchange) else self.next(date, 1, exchange)

    def set_days(self, days: list[str], exchange: str = SSE) -> None:
        """Install a calendar directly (tests)."""
        days = sorted(days)
        with self._lock:
            self._days[exchange] = days
            self._sets[exchange] = frozenset(days)

    def invalidate(self, exchange: str | None = None) -> None:
        with self._lock:
            for ex in [exchange] if exchange else list(self._days):
                self._days.pop(ex, None)
                self._sets.pop(ex, None)


calendar = AShareCalendar()
