"""The raw exchange market-data standard. This is crucible's data contract.

Authority
---------
Field names, types, scaling and enumerations here follow the **exchange**
specifications, not any vendor's encoding of them:

* SSE -- 上海证券交易所行情网关 BINARY 数据接口规范, IS120 v0.61 (2026-06-12)
* SZSE -- 深圳证券交易所 Binary 行情数据接口规范

Vendor feeds (feitu capnp cold storage, XTP, CTP, Wind TDF) are *adapters onto*
this standard, never the definition of it. That direction matters: every vendor
makes lossy choices, and if the vendor's shape becomes the internal shape those
losses become permanent and invisible. The feitu capnp archive, for instance,
stores ``price`` as ``Float64`` -- see :func:`from_vendor_float_price` for why
that is a real loss and not pedantry.

The three traps this module exists to prevent
---------------------------------------------

**1. The two exchanges use different numeric scales.**

    ============  ==================  =================  ==================
    field         SSE                 SZSE               consequence if crossed
    ============  ==================  =================  ==================
    price         ``N13(5)`` -> 1e5   ``N13(4)`` -> 1e4  **10x price error**
    quantity      ``N15(3)`` -> 1e3   ``N15(2)`` -> 1e2  **10x size error**
    amount        ``N16(2)`` -> 1e2   ``N18(4)`` -> 1e4  **100x notional error**
    ============  ==================  =================  ==================

Applying one venue's scale to the other yields prices that are wrong by an
order of magnitude but still *plausible* -- a 18.64 CNY stock reads as 186.40.
Nothing raises. Scaling is therefore never a module-level constant; it is
always looked up from the record's own exchange.

**2. Cancels arrive on different streams.**

    * SSE  -- cancellation is an **order** record, ``ExecType='4'`` (删除委托订单).
    * SZSE -- cancellation is a **trade** record, ``ExecType='4'`` (撤销), carrying
      ``BidApplSeqNum`` / ``OfferApplSeqNum`` back to the original order.

Code that reconstructs a book by "apply orders, then apply trades" will
double-count cancels on one venue and drop them on the other.
:class:`RecordType` normalises both into one vocabulary so downstream logic
never branches on venue by accident.

**3. Sequence, not timestamp, is the replay order.**

``TransactTime`` is millisecond-resolution and ties constantly -- hundreds of
records can share one millisecond. The exchange's own ordering key is
``(ChannelNo, ApplSeqNum)``, contiguous from 1 within a channel. Sorting a tick
stream by timestamp silently reorders same-millisecond events, which changes
queue position and therefore changes every fill in a microstructure backtest.
:func:`replay_key` is the only ordering crucible uses, and gaps in ApplSeqNum
are packet loss that must be detected rather than interpolated over.

Timestamps and the point-in-time rule at tick frequency
-------------------------------------------------------
``TransactTime`` is the **exchange generation** time. It is not when you could
have acted on the record. A feature evaluated at decision time ``t`` may only
read records whose *arrival* time at your colo box is ``<= t``; using
generation time assumes zero wire latency and back-dates every observation by
the one-way delay. At daily frequency this is invisible; at 3-second slots it
is decisive, and at tick level it is the whole game.

crucible therefore carries up to three timestamps per record and is explicit
about which one gates a feature -- see :class:`Timestamps` and
:data:`PIT_TIMESTAMP`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Final

__all__ = [
    "Exchange",
    "Scaling",
    "SSE_SCALING",
    "SZSE_SCALING",
    "scaling_for",
    "Side",
    "RecordType",
    "TradingPhase",
    "PhaseFlags",
    "Timestamps",
    "PIT_TIMESTAMP",
    "decode_sse_time",
    "decode_szse_time",
    "parse_sse_phase",
    "parse_szse_phase",
    "replay_key",
    "find_sequence_gaps",
    "merged_sequence_gaps",
    "normalize_sse_order",
    "normalize_szse_trade",
    "from_vendor_float_price",
    "to_price",
    "to_quantity",
    "to_amount",
]


class Exchange(Enum):
    """Trading venue, keyed by ISO 10383 MIC."""

    XSHG = "XSHG"
    """Shanghai Stock Exchange."""
    XSHE = "XSHE"
    """Shenzhen Stock Exchange."""

    @property
    def wind_suffix(self) -> str:
        """Wind-style code suffix, ``.SH`` / ``.SZ``."""
        return ".SH" if self is Exchange.XSHG else ".SZ"

    @classmethod
    def of_symbol(cls, symbol: str) -> Exchange:
        """Infer the venue from a 6-digit code.

        Uses the exchange code-range allocations. Note ``000xxx`` is genuinely
        ambiguous -- Shenzhen equities share the range with Shanghai indices --
        so an explicit venue always wins over inference. Callers holding a real
        feed should use the record's own venue and never call this.
        """
        s = str(symbol)
        if s.startswith(("5", "6", "9")):
            return cls.XSHG
        return cls.XSHE


@dataclass(frozen=True)
class Scaling:
    """Fixed-point scale factors for one venue.

    The exchange transmits prices, quantities and amounts as **integers** with
    an implied decimal position (``N13(4)`` means 13 digits, 4 of them after
    the point). crucible keeps them as integers for as long as possible: an
    integer tick compares exactly against a limit band, and a float does not.
    """

    price: int
    quantity: int
    amount: int
    price_decimals: int
    quantity_decimals: int
    amount_decimals: int


SSE_SCALING: Final = Scaling(
    price=10**5, quantity=10**3, amount=10**2,
    price_decimals=5, quantity_decimals=3, amount_decimals=2,
)
"""SSE: Price ``N13(5)``, OrderQty ``N15(3)``, TotalValueTraded ``N16(2)``."""

SZSE_SCALING: Final = Scaling(
    price=10**4, quantity=10**2, amount=10**4,
    price_decimals=4, quantity_decimals=2, amount_decimals=4,
)
"""SZSE: Price ``N13(4)``, Qty ``N15(2)``, Amt ``N18(4)``."""


def scaling_for(exchange: Exchange) -> Scaling:
    """Scale factors for ``exchange``.

    Always call this per record. A module-level scale constant is how a 10x
    price error gets shipped.
    """
    return SSE_SCALING if exchange is Exchange.XSHG else SZSE_SCALING


def to_price(raw: int, exchange: Exchange) -> float:
    """Scaled integer price -> float CNY.

    Lossy by construction; do the comparison against limit bands and tick grids
    on the integer instead wherever it matters.
    """
    return raw / scaling_for(exchange).price


def to_quantity(raw: int, exchange: Exchange) -> float:
    """Scaled integer quantity -> shares (equities) or contracts."""
    return raw / scaling_for(exchange).quantity


def to_amount(raw: int, exchange: Exchange) -> float:
    """Scaled integer turnover -> CNY."""
    return raw / scaling_for(exchange).amount


def from_vendor_float_price(price: float, exchange: Exchange) -> int:
    """Recover the exchange's integer price from a vendor's ``Float64``.

    Cold-storage archives (the feitu capnp set among them) re-encode price as
    ``Float64``, and a binary float cannot represent most 2-decimal CNY prices
    exactly. A price of 1.13 comes back as ``1.1299999999999999``, so the naive
    ``int(p * 10**4)`` truncates to ``11299`` instead of ``11300`` -- one tick
    low, silently.

    This is not a rare edge: measured over every 0.01 increment from 1.00 to
    2000.00, **10,456 of 199,900 prices (5.2%) truncate a tick low**. A limit
    band or tick-grid comparison against those is wrong roughly one time in
    twenty. Rounding, not truncation, is required.

    This function is the *only* sanctioned way to cross that boundary, so the
    loss happens in one reviewable place.
    """
    return round(price * scaling_for(exchange).price)


class Side(Enum):
    """Order side.

    SSE and SZSE agree on ``'1'``/``'2'``. SZSE additionally uses ``'G'`` and
    ``'F'`` for the securities-lending (转融通) stream, which are *not* ordinary
    buys and sells and must not be folded into them.
    """

    BUY = "1"
    SELL = "2"
    BORROW = "G"
    """SZSE 借入 -- securities lending, not a directional trade."""
    LEND = "F"
    """SZSE 出借 -- securities lending, not a directional trade."""

    @property
    def is_directional(self) -> bool:
        """Whether this side represents ordinary buying or selling pressure."""
        return self in (Side.BUY, Side.SELL)


class RecordType(Enum):
    """Venue-neutral record vocabulary.

    Both exchanges are normalised into this, because their raw encodings
    disagree in a way that breaks book reconstruction:

    ==================  =========================  ==========================
    meaning             SSE                        SZSE
    ==================  =========================  ==========================
    new order           order,  ``ExecType='0'``   order record
    cancel              order,  ``ExecType='4'``   **trade**, ``ExecType='4'``
    execution           trade,  ``ExecType='F'``   trade,  ``ExecType='F'``
    ==================  =========================  ==========================
    """

    ORDER_ADD = "add"
    ORDER_CANCEL = "cancel"
    TRADE = "trade"
    STATUS = "status"
    """SSE product-status records that ride the order stream."""


class TradingPhase(Enum):
    """First character of ``TradingPhaseCode`` -- the session phase.

    Taken from the SSE spec's ``SecurityType=1, MDStreamID=MD002`` table. SZSE
    uses a compatible vocabulary for its equity phase codes.
    """

    START = "S"
    """启动 -- pre-open."""
    OPEN_AUCTION = "C"
    """开盘集合竞价."""
    CONTINUOUS = "T"
    """连续交易."""
    CLOSED = "E"
    """闭市."""
    SUSPENDED = "P"
    """产品停牌 -- the authoritative suspension flag."""
    BREAK_RECOVERABLE = "M"
    """可恢复熔断 (盘中集合竞价)."""
    BREAK_TERMINAL = "N"
    """不可恢复熔断 -- halted until close."""
    CLOSE_AUCTION = "U"
    """收盘集合竞价."""
    UNKNOWN = "?"

    @classmethod
    def parse(cls, ch: str) -> TradingPhase:
        try:
            return cls(ch)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class PhaseFlags:
    """Decoded ``TradingPhaseCode``: the authoritative tradability state.

    This is strictly better than inferring suspension from ``volume == 0``.
    Zero volume conflates a suspended name, a halted name, and a name that
    simply had no trades in the interval -- three different states with three
    different correct handlings. The exchange tells you which; use it.

    Attributes
    ----------
    phase:
        Session phase (first character).
    can_trade:
        Second character. ``'1'`` means the product may trade normally. Note
        the spec's subtlety: after close this **retains** the product's
        pre-close value, so it describes the session that just ended.
    is_listed:
        Third character. ``'0'`` means not yet listed.
    accepts_orders:
        Fourth character. Only meaningful during a trading session.
    """

    phase: TradingPhase
    can_trade: bool
    is_listed: bool
    accepts_orders: bool
    raw: str

    @property
    def is_halted(self) -> bool:
        """Suspended or in a terminal circuit break."""
        return self.phase in (TradingPhase.SUSPENDED, TradingPhase.BREAK_TERMINAL)

    @property
    def is_auction(self) -> bool:
        """In a call auction, where fills are single-price rather than continuous."""
        return self.phase in (
            TradingPhase.OPEN_AUCTION,
            TradingPhase.CLOSE_AUCTION,
            TradingPhase.BREAK_RECOVERABLE,
        )

    @property
    def is_tradable(self) -> bool:
        """Listed, permitted to trade, and in a phase where trading happens."""
        return (
            self.is_listed
            and self.can_trade
            and not self.is_halted
            and self.phase in (TradingPhase.CONTINUOUS, TradingPhase.OPEN_AUCTION, TradingPhase.CLOSE_AUCTION)
        )


def parse_sse_phase(code: str) -> PhaseFlags:
    """Parse an SSE 8-character ``TradingPhaseCode``.

    Undefined positions are spaces per the spec, so short or padded strings are
    tolerated rather than raising -- a malformed phase code should degrade to
    "not tradable", never to an exception mid-replay.
    """
    s = (code or "").ljust(4)
    return PhaseFlags(
        phase=TradingPhase.parse(s[0]),
        can_trade=s[1] == "1",
        is_listed=s[2] != "0",
        accepts_orders=s[3] == "1",
        raw=code or "",
    )


def parse_szse_phase(code: str) -> PhaseFlags:
    """Parse an SZSE ``TradingPhaseCode``.

    SZSE's equity phase vocabulary matches SSE's in the first character. The
    remaining positions differ in meaning between the venues, so only the
    fields crucible relies on are decoded; the original is kept in ``raw``.
    """
    s = (code or "").ljust(4)
    phase = TradingPhase.parse(s[0])
    return PhaseFlags(
        phase=phase,
        can_trade=phase not in (TradingPhase.SUSPENDED, TradingPhase.BREAK_TERMINAL),
        is_listed=True,
        accepts_orders=phase in (TradingPhase.CONTINUOUS, TradingPhase.OPEN_AUCTION, TradingPhase.CLOSE_AUCTION),
        raw=code or "",
    )


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------

PIT_TIMESTAMP: Final = "arrival_ts"
"""Column that gates point-in-time feature access.

**Not** ``exchange_ts``. A feature at decision time ``t`` may read a record only
when it had physically arrived by ``t``; keying on exchange generation time
assumes zero wire latency and back-dates every observation by the one-way
delay. See the module docstring.
"""


@dataclass(frozen=True)
class Timestamps:
    """The timestamps a record can carry, in causal order.

    ``exchange_ts <= broker_ts <= arrival_ts``. The broker sits **upstream** of
    the capture box -- data leaves the exchange, reaches the broker's server,
    and only then reaches us -- so arrival is last, not middle. A violation
    means clock skew and must be surfaced rather than silently sorted away;
    :meth:`is_causal` exists for that check.

    Measured on 600,000 real records (feitu v2 order + transaction,
    2026-04-21 10:30): ``broker_ts <= arrival_ts`` held on 100.0% of records,
    and ``exchange_ts -> broker_ts`` was **exactly 0 ms at every percentile
    including max**. In that archive ``serverTs`` is a copy of the exchange
    stamp rather than an independent measurement, so the two legs cannot be
    decomposed -- only total wire time is observable. Do not read a
    ``broker_ts`` equal to ``exchange_ts`` as "zero broker latency"; read it as
    "not measured".

    Attributes
    ----------
    exchange_ts:
        Generation time at the exchange (``TransactTime`` / ``LastUpdateTime``),
        nanoseconds. Millisecond resolution in practice.
    broker_ts:
        Receipt time at the broker server (``serverTs``).
    arrival_ts:
        Capture time at our box (the vendor archive's ``spiderTs``).
        **This is the point-in-time gate** -- the last point in the chain, and
        the only instant at which the record is genuinely knowable to us.
    """

    exchange_ts: int
    arrival_ts: int | None = None
    broker_ts: int | None = None

    @property
    def is_causal(self) -> bool:
        """Whether the stamps are ordered as physics requires."""
        if self.arrival_ts is None:
            return True
        if self.arrival_ts < self.exchange_ts:
            return False
        if self.broker_ts is not None and not (
            self.exchange_ts <= self.broker_ts <= self.arrival_ts
        ):
            return False
        return True

    @property
    def wire_latency_ns(self) -> int | None:
        """Exchange-to-colo one-way latency, or ``None`` if not captured.

        Worth monitoring as a time series: a shift in its distribution usually
        means a network change that also shifted your fill rates.
        """
        return None if self.arrival_ts is None else self.arrival_ts - self.exchange_ts

    def pit(self) -> int:
        """The timestamp a feature must be gated on.

        Falls back to ``exchange_ts`` when no capture time exists, which is the
        best available answer but **optimistic** -- it assumes instant delivery.
        """
        return self.arrival_ts if self.arrival_ts is not None else self.exchange_ts


def decode_sse_time(trade_date: int, hhmmssmmm: int) -> dt.datetime:
    """SSE ``TradeDate`` + ``HHMMSSsss`` -> a naive exchange-local datetime.

    SSE transmits time-of-day only (``uint32``, ``N9``); the date lives in a
    separate ``TradeDate`` field on the message. Reconstructing the full
    timestamp requires both, which is why they travel together here rather than
    being decoded independently.
    """
    y, m, d = trade_date // 10000, trade_date // 100 % 100, trade_date % 100
    ms = hhmmssmmm % 1000
    hh = hhmmssmmm // 10_000_000
    mm = hhmmssmmm // 100_000 % 100
    ss = hhmmssmmm // 1000 % 100
    return dt.datetime(y, m, d, hh, mm, ss, ms * 1000)


def decode_szse_time(local_timestamp: int) -> dt.datetime:
    """SZSE ``LocalTimeStamp`` (``YYYYMMDDHHMMSSsss``) -> naive datetime.

    Self-contained, unlike SSE: the date is part of the field.
    """
    ms = local_timestamp % 1000
    rest = local_timestamp // 1000
    ss, rest = rest % 100, rest // 100
    mm, rest = rest % 100, rest // 100
    hh, date = rest % 100, rest // 100
    return dt.datetime(
        date // 10000, date // 100 % 100, date % 100, hh, mm, ss, ms * 1000
    )


# --------------------------------------------------------------------------
# sequencing
# --------------------------------------------------------------------------


def replay_key(channel_no: int, appl_seq_num: int) -> tuple[int, int]:
    """The canonical replay ordering key.

    ``(ChannelNo, ApplSeqNum)``, not timestamp. ``TransactTime`` has
    millisecond resolution and ties constantly; sorting by it reorders
    same-millisecond events and therefore changes queue position and every
    resulting fill.
    """
    return (channel_no, appl_seq_num)


def find_sequence_gaps(
    seq_nums: list[int], *, start: int = 1
) -> list[tuple[int, int]]:
    """Missing ``ApplSeqNum`` ranges within one channel.

    ``ApplSeqNum`` is contiguous from 1 within a channel and covers orders and
    trades **jointly**, so a gap is packet loss, not a quiet period. Returning
    the gaps rather than repairing them is deliberate: an interpolated book is
    a fabricated book, and a tick backtest run over one is meaningless.

    .. warning::
       ``seq_nums`` must be the **merged** order + trade stream for the channel.
       Passing one stream alone reports the other stream's records as loss.

       Measured on a full minute of real data (2026-04-21 10:30, 14 channels,
       both venues, 2.47M records): per-stream gap counts were 845,453 (order)
       and 1,418,420 (trade), while the **merged** stream had exactly **0 gaps
       on every channel**. Each stream's "missing" numbers were precisely the
       other stream's records. A single-stream check would have condemned a
       flawless capture as ~50% lossy.

    See Also
    --------
    merged_sequence_gaps : takes both streams and does this correctly.

    Returns
    -------
    ``(first_missing, last_missing)`` inclusive ranges.
    """
    if not seq_nums:
        return []
    ordered = sorted(set(seq_nums))
    gaps: list[tuple[int, int]] = []
    if ordered[0] > start:
        gaps.append((start, ordered[0] - 1))
    for a, b in zip(ordered[:-1], ordered[1:]):
        if b > a + 1:
            gaps.append((a + 1, b - 1))
    return gaps


def merged_sequence_gaps(
    order_seq_nums: Iterable[int],
    trade_seq_nums: Iterable[int],
    *,
    start: int = 1,
) -> list[tuple[int, int]]:
    """Sequence gaps for one channel, across both tick streams.

    The correct way to detect packet loss. Orders and trades share a single
    per-channel ``ApplSeqNum`` space, so contiguity is only meaningful once
    both are combined -- see the warning on :func:`find_sequence_gaps` for the
    measured consequence of checking either alone.

    Parameters
    ----------
    order_seq_nums, trade_seq_nums:
        Sequence numbers from the two streams **for the same channel and the
        same time window**. Windows that differ produce spurious gaps at the
        edges purely from the mismatch.

    Returns
    -------
    ``(first_missing, last_missing)`` inclusive ranges; empty when the merged
    stream is contiguous.
    """
    return find_sequence_gaps([*order_seq_nums, *trade_seq_nums], start=start)


# --------------------------------------------------------------------------
# venue normalisation
# --------------------------------------------------------------------------


def normalize_sse_order(exec_type: str) -> RecordType:
    """SSE order-stream ``ExecType`` -> :class:`RecordType`.

    ``'0'`` 新增委托订单, ``'4'`` 删除委托订单. Cancels arrive **here** on SSE,
    unlike SZSE.
    """
    if exec_type == "0":
        return RecordType.ORDER_ADD
    if exec_type == "4":
        return RecordType.ORDER_CANCEL
    return RecordType.STATUS


def normalize_szse_trade(exec_type: str) -> RecordType:
    """SZSE trade-stream ``ExecType`` -> :class:`RecordType`.

    ``'F'`` 成交, ``'4'`` 撤销. Cancels arrive **here** on SZSE, on the trade
    stream, carrying ``BidApplSeqNum``/``OfferApplSeqNum`` back to the order
    being withdrawn. Treating every trade-stream record as an execution
    inflates SZSE volume by the cancel count.
    """
    if exec_type == "F":
        return RecordType.TRADE
    if exec_type == "4":
        return RecordType.ORDER_CANCEL
    return RecordType.STATUS
