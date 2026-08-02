"""Trading calendar and intraday slot grid.

Two concerns that are usually conflated, kept apart here:

* **Which dates trade** -- holidays, weekends. Delegated to
  ``exchange_calendars``, which tracks the CSRC holiday schedule properly.
* **What a trading day looks like intraday** -- session breaks, auction
  windows, night sessions. Held in an explicit :class:`SessionSpec`, because
  A-share equity, CFFEX index futures and SHFE commodity all differ, and
  because ``exchange_calendars`` only resolves to the minute while tick
  research needs 3-second slots.

Timestamps are tz-naive exchange local time (Asia/Shanghai) throughout. Mixing
tz-aware and tz-naive timestamps in a join is a classic silent misalignment, so
crucible picks one and never carries the other.

Slot labelling
--------------
Slots are labelled by their **closing** edge and each slot covers ``(label -
freq, label]``. This is not cosmetic. A bar labelled 09:30 that contains trades
up to 09:31 is a look-ahead bug: a feature reading it at 09:30 sees a minute of
the future. Right-labelling makes the label the moment the information is
complete, which is the only labelling that satisfies the point-in-time rule.
"""

from __future__ import annotations

import datetime as dt
import functools
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import polars as pl

__all__ = [
    "SessionSpec",
    "EQUITY",
    "CFFEX_INDEX",
    "CFFEX_BOND",
    "COMMODITY_NIGHT",
    "TradingCalendar",
    "parse_freq",
]

_FREQ_RE = re.compile(r"^(\d+)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hours?|d|days?)$")

_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "h": 3600,
    "hour": 3600,
    "hours": 3600,
    "d": 86400,
    "day": 86400,
    "days": 86400,
}


def parse_freq(freq: str) -> dt.timedelta:
    """Parse ``"3s"``, ``"1min"``, ``"5min"``, ``"1d"`` into a timedelta.

    Deliberately narrow. Pandas offset aliases allow calendar-relative units
    ("M", "Q", "BME") whose length depends on where you stand, which cannot
    tile a fixed intraday grid.
    """
    raw = freq.strip()
    # Bare uppercase "M" is pandas' *month* alias while lowercase "m" is
    # minutes. Lowercasing first would silently turn "1M" into one minute --
    # a 43,000x unit error. Reject it rather than guess which was meant.
    if raw.endswith("M") and not raw.upper().endswith("MIN"):
        raise ValueError(
            f"ambiguous freq {freq!r}: bare 'M' is month in pandas but minute elsewhere. "
            f"Write '1min' for minutes; calendar-relative units cannot tile an intraday grid."
        )
    m = _FREQ_RE.match(raw.lower())
    if m is None:
        raise ValueError(
            f"unsupported freq {freq!r}; expected forms like '3s', '30s', '1min', '5min', '1d'"
        )
    n, unit = int(m.group(1)), m.group(2)
    if n <= 0:
        raise ValueError(f"freq must be positive, got {freq!r}")
    return dt.timedelta(seconds=n * _UNIT_SECONDS[unit])


@dataclass(frozen=True)
class SessionSpec:
    """Intraday structure of one trading day for a class of instruments.

    Intervals are half-open ``[start, end)`` in exchange local time and must be
    given in chronological order.

    ``night`` intervals are the previous calendar evening's session that
    *belongs to this trading date* -- the commodity convention where Monday
    21:00 trades settle on Tuesday. Modelling them as part of the following
    trading date is what keeps a daily groupby from splitting one continuous
    session across two rows.
    """

    name: str
    intervals: tuple[tuple[dt.time, dt.time], ...]
    open_auction: tuple[dt.time, dt.time] | None = None
    close_auction: tuple[dt.time, dt.time] | None = None
    night: tuple[tuple[dt.time, dt.time], ...] = field(default=())

    def __post_init__(self) -> None:
        for lo, hi in self.intervals:
            if lo >= hi:
                raise ValueError(f"{self.name}: interval {lo}-{hi} is empty or inverted")
        starts = [lo for lo, _ in self.intervals]
        if starts != sorted(starts):
            raise ValueError(f"{self.name}: intervals must be chronological")

    @property
    def seconds_per_day(self) -> int:
        """Total continuous-trading seconds, night session included."""
        total = 0
        for lo, hi in (*self.intervals, *self.night):
            total += (
                _secs(hi) - _secs(lo) if _secs(hi) > _secs(lo) else 86400 - _secs(lo) + _secs(hi)
            )
        return total


def _secs(t: dt.time) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


#: Shanghai/Shenzhen equities. Open call auction 09:15-09:25 (match at 09:25),
#: continuous 09:30-11:30 and 13:00-14:57, close call auction 14:57-15:00.
#: The close auction is carved out of continuous trading because fills there
#: are single-price -- treating it as continuous overstates intraday capacity.
EQUITY = SessionSpec(
    name="equity",
    intervals=((dt.time(9, 30), dt.time(11, 30)), (dt.time(13, 0), dt.time(14, 57))),
    open_auction=(dt.time(9, 15), dt.time(9, 25)),
    close_auction=(dt.time(14, 57), dt.time(15, 0)),
)

#: CFFEX index futures (IF/IC/IM/IH). Aligned to the cash session since 2016.
CFFEX_INDEX = SessionSpec(
    name="cffex_index",
    intervals=((dt.time(9, 30), dt.time(11, 30)), (dt.time(13, 0), dt.time(15, 0))),
    open_auction=(dt.time(9, 25), dt.time(9, 30)),
)

#: CFFEX treasury futures (TF/T/TS). Trades 15 minutes past the cash close.
CFFEX_BOND = SessionSpec(
    name="cffex_bond",
    intervals=((dt.time(9, 30), dt.time(11, 30)), (dt.time(13, 0), dt.time(15, 15))),
    open_auction=(dt.time(9, 25), dt.time(9, 30)),
)

#: Representative SHFE/DCE metal-style day plus a 21:00-23:00 night session.
#: Night hours vary by product; construct your own SessionSpec per contract
#: rather than assuming this one fits.
COMMODITY_NIGHT = SessionSpec(
    name="commodity_night",
    intervals=(
        (dt.time(9, 0), dt.time(10, 15)),
        (dt.time(10, 30), dt.time(11, 30)),
        (dt.time(13, 30), dt.time(15, 0)),
    ),
    open_auction=(dt.time(8, 55), dt.time(9, 0)),
    night=((dt.time(21, 0), dt.time(23, 0)),),
)


class TradingCalendar:
    """Sessions and intraday slot grids for one instrument class.

    Parameters
    ----------
    exchange:
        ``exchange_calendars`` code supplying the holiday schedule. ``XSHG``
        covers both Shanghai and Shenzhen equities and the CFFEX products,
        which follow the same holiday calendar.
    session:
        Intraday structure. Defaults to :data:`EQUITY`.

    Point-in-time contract
    ----------------------
    Every method is a pure function of the calendar definition. Nothing here
    reads the current date, so a calendar built today answers historical
    questions the same way it will next year.
    """

    def __init__(self, exchange: str = "XSHG", session: SessionSpec = EQUITY) -> None:
        import exchange_calendars as xcals

        self._cal = xcals.get_calendar(exchange, side="left")
        self.exchange = exchange
        self.session = session

    # -- session dates ---------------------------------------------------

    @functools.lru_cache(maxsize=64)  # noqa: B019 -- calendar is immutable
    def _sessions_cached(self, start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
        idx = self._cal.sessions_in_range(str(start), str(end))
        return tuple(d.date() for d in idx)

    def sessions(self, start: dt.date | str, end: dt.date | str) -> list[dt.date]:
        """Trading dates in ``[start, end]`` inclusive."""
        return list(self._sessions_cached(_as_date(start), _as_date(end)))

    def is_session(self, day: dt.date | str) -> bool:
        """Whether ``day`` is a trading date."""
        return self._cal.is_session(str(_as_date(day)))

    def shift(self, day: dt.date | str, n: int) -> dt.date:
        """The trading date ``n`` sessions from ``day`` (``n`` may be negative).

        ``day`` need not itself be a session; the search starts from the
        nearest session in the direction of travel.
        """
        d = _as_date(day)
        if n == 0:
            if not self.is_session(d):
                raise ValueError(f"{d} is not a trading session; shift by +/-1 to snap")
            return d
        step = 1 if n > 0 else -1
        span = dt.timedelta(days=abs(n) * 3 + 30)
        lo, hi = (d, d + span) if n > 0 else (d - span, d)
        sess = self.sessions(lo, hi)
        if n > 0:
            after = [s for s in sess if s > d]
            if len(after) < n:
                raise ValueError(f"cannot shift {d} forward {n} sessions within {span.days}d")
            return after[n - 1]
        before = [s for s in sess if s < d]
        if len(before) < -n:
            raise ValueError(f"cannot shift {d} back {-n} sessions within {span.days}d")
        return before[n]

    def is_half_day(self, day: dt.date | str) -> bool:
        """Whether ``day`` closes early relative to the spec's normal close.

        A-shares have no regular half days, but the schedule has carried
        one-off early closes, and a slot grid that assumes a full day would
        manufacture bars after the close.
        """
        d = _as_date(day)
        if not self.is_session(d):
            return False
        normal = self.session.intervals[-1][1]
        return bool(self._local_close(d) < normal)

    # -- intraday grid ---------------------------------------------------

    def slot_grid(
        self,
        day: dt.date | str,
        freq: str = "1min",
        *,
        include_night: bool = False,
        include_close_auction: bool = True,
    ) -> pl.Series:
        """Right-labelled slot boundaries for one trading date.

        Each returned timestamp ``t`` labels the slot covering ``(t - freq, t]``.
        The first label of a session is therefore ``open + freq``, never the
        open itself -- at the open no time has elapsed and there is nothing to
        aggregate.

        Parameters
        ----------
        include_night:
            Prepend the night session belonging to this trading date. Those
            timestamps fall on the *previous* calendar evening.
        include_close_auction:
            Append a single slot labelled at the close-auction end. The auction
            is one price, not a series of slots, so it contributes exactly one
            label regardless of ``freq``.

        Raises
        ------
        ValueError
            If ``day`` is not a trading session, or ``freq`` does not divide a
            session interval evenly. A ragged final slot silently shortens the
            last bar of each session, which biases any open/close feature.
        """
        d = _as_date(day)
        if not self.is_session(d):
            raise ValueError(f"{d} is not a trading session on {self.exchange}")
        step = parse_freq(freq)
        if step >= dt.timedelta(days=1):
            return pl.Series(
                "ts",
                [dt.datetime.combine(d, self.session.intervals[-1][1])],
                dtype=pl.Datetime("ns"),
            )

        out: list[dt.datetime] = []

        if include_night and self.session.night:
            prev_eve = d - dt.timedelta(days=1)
            for lo, hi in self.session.night:
                base = dt.datetime.combine(prev_eve, lo)
                end = dt.datetime.combine(prev_eve if _secs(hi) > _secs(lo) else d, hi)
                out.extend(_tile(base, end, step, freq, f"night {lo}-{hi}"))

        last_close = self._effective_close(d)
        for lo, hi in self.session.intervals:
            hi = min(hi, last_close)
            if lo >= hi:
                continue
            out.extend(
                _tile(
                    dt.datetime.combine(d, lo),
                    dt.datetime.combine(d, hi),
                    step,
                    freq,
                    f"session {lo}-{hi}",
                )
            )

        if include_close_auction and self.session.close_auction is not None:
            ca_end = self.session.close_auction[1]
            if ca_end <= last_close:
                out.append(dt.datetime.combine(d, ca_end))

        return pl.Series("ts", out, dtype=pl.Datetime("ns"))

    def _local_close(self, d: dt.date) -> dt.time:
        """Exchange-local close time for ``d``.

        ``exchange_calendars`` returns tz-aware UTC timestamps. Taking
        ``.time()`` off one directly yields 07:00 for a 15:00 Shanghai close,
        which is *before* the 09:30 open -- every session then clips to an
        empty grid and every day looks like a half day. Convert first.
        """
        ts = self._cal.session_close(str(d))
        tz = getattr(self._cal, "tz", None)
        if getattr(ts, "tzinfo", None) is not None and tz is not None:
            ts = ts.tz_convert(tz)
        return ts.time()

    def _effective_close(self, d: dt.date) -> dt.time:
        """Normal close, or the early close on a half day."""
        normal = self.session.intervals[-1][1]
        if self.session.close_auction is not None:
            normal = max(normal, self.session.close_auction[1])
        actual = self._local_close(d)
        return min(normal, actual) if actual < normal else normal

    def slots(
        self,
        start: dt.date | str,
        end: dt.date | str,
        freq: str = "1min",
        **kwargs: bool,
    ) -> pl.Series:
        """Concatenated slot grid across every session in ``[start, end]``.

        Materialises the whole range. At 3s that is ~4800 labels per day, so a
        year is ~1.2M timestamps -- fine as a Series, but do not cross-join it
        against 5000 symbols and expect to hold the result.
        """
        parts = [self.slot_grid(d, freq, **kwargs) for d in self.sessions(start, end)]
        if not parts:
            return pl.Series("ts", [], dtype=pl.Datetime("ns"))
        return pl.concat(parts)

    def iter_sessions(
        self, start: dt.date | str, end: dt.date | str
    ) -> Iterator[tuple[dt.date, pl.Series]]:
        """Yield ``(date, slot_grid)`` one session at a time.

        The streaming counterpart to :meth:`slots`. Tick pipelines should drive
        off this rather than building a range-wide grid.
        """
        for d in self.sessions(start, end):
            yield d, self.slot_grid(d)

    def n_slots(self, freq: str = "1min", **kwargs: bool) -> int:
        """Slots in a full (non-half) session at ``freq``."""
        ref = self.sessions(dt.date(2023, 3, 1), dt.date(2023, 3, 31))[0]
        return len(self.slot_grid(ref, freq, **kwargs))

    # -- auctions --------------------------------------------------------

    def in_open_auction(self, ts: dt.datetime) -> bool:
        """Whether ``ts`` falls in the open call auction window."""
        return _in_window(ts, self.session.open_auction)

    def in_close_auction(self, ts: dt.datetime) -> bool:
        """Whether ``ts`` falls in the close call auction window."""
        return _in_window(ts, self.session.close_auction)

    def in_continuous(self, ts: dt.datetime) -> bool:
        """Whether ``ts`` falls in a continuous-trading interval.

        Orders outside continuous trading cannot be assumed to fill at the
        prevailing quote, so cost models must branch on this.
        """
        t = ts.time()
        return any(lo <= t < hi for lo, hi in (*self.session.intervals, *self.session.night))

    def __repr__(self) -> str:
        return f"TradingCalendar(exchange={self.exchange!r}, session={self.session.name!r})"


def _in_window(ts: dt.datetime, window: tuple[dt.time, dt.time] | None) -> bool:
    if window is None:
        return False
    lo, hi = window
    return lo <= ts.time() < hi


def _tile(
    start: dt.datetime, end: dt.datetime, step: dt.timedelta, freq: str, what: str
) -> list[dt.datetime]:
    """Right-labelled boundaries tiling ``(start, end]``, refusing a ragged tail."""
    span = end - start
    if span.total_seconds() % step.total_seconds() != 0:
        raise ValueError(
            f"freq {freq!r} does not divide {what} evenly "
            f"({span} / {step}); a ragged final slot would bias open/close features"
        )
    n = int(span / step)
    return [start + step * (i + 1) for i in range(n)]


def _as_date(d: dt.date | str) -> dt.date:
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return dt.date.fromisoformat(str(d))


def sessions_between(
    cal: TradingCalendar, start: dt.date | str, end: dt.date | str
) -> Sequence[dt.date]:
    """Free-function alias for :meth:`TradingCalendar.sessions`."""
    return cal.sessions(start, end)
