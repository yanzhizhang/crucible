"""Market-by-order book reconstruction from the order and trade streams.

Rebuilds the full limit order book by replaying every add, cancel and execution
in the exchange's own event order. That recovers everything the missing
quotation stream would have given -- spread, depth, queue position, microprice --
and more besides, because an MBO book knows *which* order sits where, not just
how much size is at a level.

Replay order
------------
Events are merged across both streams and sorted by ``(channelId, seqId)``,
never by timestamp. ``TransactTime`` is millisecond-resolution and ties
constantly; sorting by it reorders same-millisecond events, which changes queue
position and therefore changes every fill. The two streams share one
per-channel sequence space (measured: merged gaps were zero on every channel),
so the merge reproduces the exchange's exact sequence.

Venue divergence, measured not assumed
--------------------------------------
Verified on real data (2026-04-21 10:30) by ``probe_mbo_linkage.py``:

============  =========================  ==========================
concern       SSE (XSHG)                 SZSE (XSHE)
============  =========================  ==========================
order key     ``orderId``                ``seqId``
trade ref     -> ``orderId``  (25%)      -> ``seqId``  (50%)
wrong key     ``seqId`` gives 1.4%       ``orderId`` gives 0.0%
cancels       order stream, ``updateType=2``  trade stream, ``tradeType=2``
``orderType`` not populated (all 0)      limit/market/best-price
============  =========================  ==========================

Using one venue's key on the other resolves ~0% of references. The book still
builds -- orders accumulate and nothing raises -- it is simply wrong.

Unknown-order references
------------------------
A trade may reference an order added before the replay window opened. Those are
counted in :attr:`Book.unresolved` rather than silently ignored, because a high
unresolved rate means the book is being built from a partial history and its
depth is understated. Warm the replay from the session open to drive it to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

__all__ = ["Book", "BookSnapshot", "replay_events"]

_SSE: Final = "XSHG"


@dataclass(frozen=True)
class BookSnapshot:
    """Top-of-book and depth state at one instant."""

    symbol: str
    best_bid: float
    best_ask: float
    bid_qty: int
    ask_qty: int
    bid_depth5: int
    ask_depth5: int
    n_bid_orders: int
    n_ask_orders: int
    bid_levels: int
    ask_levels: int

    @property
    def spread(self) -> float:
        """Quoted spread. ``nan`` when either side is empty."""
        if self.best_bid <= 0 or self.best_ask <= 0:
            return float("nan")
        return self.best_ask - self.best_bid

    @property
    def mid(self) -> float:
        """Arithmetic mid."""
        if self.best_bid <= 0 or self.best_ask <= 0:
            return float("nan")
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def microprice(self) -> float:
        """Size-weighted mid: ``(P_b*Q_a + P_a*Q_b) / (Q_a + Q_b)``.

        Leans toward the side with *less* size, because that is the side likely
        to be consumed next. Consistently the strongest single short-horizon
        predictor in the microstructure literature, and unavailable without a
        book -- which is the main reason to rebuild one.
        """
        if self.best_bid <= 0 or self.best_ask <= 0:
            return float("nan")
        tot = self.bid_qty + self.ask_qty
        if tot <= 0:
            return self.mid
        return (self.best_bid * self.ask_qty + self.best_ask * self.bid_qty) / tot

    @property
    def imbalance_l1(self) -> float:
        """Top-of-book size imbalance in ``[-1, 1]``."""
        tot = self.bid_qty + self.ask_qty
        return (self.bid_qty - self.ask_qty) / tot if tot > 0 else 0.0

    @property
    def imbalance_l5(self) -> float:
        """Five-level depth imbalance; steadier than L1, slower to react."""
        tot = self.bid_depth5 + self.ask_depth5
        return (self.bid_depth5 - self.ask_depth5) / tot if tot > 0 else 0.0


@dataclass
class Book:
    """One symbol's limit order book, keyed by resting order.

    ``orders`` maps the venue-appropriate order key to
    ``[price, side, remaining_qty]``. Price levels are aggregated on demand;
    keeping only the order map avoids maintaining two structures that can drift
    apart.
    """

    symbol: str
    exchange: str
    orders: dict[int, list] = field(default_factory=dict)
    unresolved: int = 0
    trades: int = 0
    traded_qty: int = 0
    buy_traded: int = 0
    sell_traded: int = 0

    def add(self, key: int, price: float, side: str, qty: int) -> None:
        """Insert a resting order. Zero-price or zero-qty orders are ignored.

        A zero price on an add is a market or best-price order that never
        rested; treating it as a limit at 0.00 would create a permanent phantom
        bid at the bottom of the book.
        """
        if qty <= 0 or price <= 0 or side not in ("buy", "sell"):
            return
        self.orders[key] = [price, side, qty]

    def cancel(self, key: int, qty: int = 0) -> bool:
        """Remove or reduce a resting order. Returns whether it was found."""
        o = self.orders.get(key)
        if o is None:
            self.unresolved += 1
            return False
        if qty <= 0 or qty >= o[2]:
            del self.orders[key]
        else:
            o[2] -= qty
        return True

    def execute(self, buy_key: int, sell_key: int, qty: int, side: str | None) -> None:
        """Apply a trade against both resting orders."""
        self.trades += 1
        self.traded_qty += qty
        if side == "buy":
            self.buy_traded += qty
        elif side == "sell":
            self.sell_traded += qty
        for key in (buy_key, sell_key):
            if not key:
                continue
            o = self.orders.get(key)
            if o is None:
                self.unresolved += 1
                continue
            o[2] -= qty
            if o[2] <= 0:
                del self.orders[key]

    def snapshot(self, levels: int = 5) -> BookSnapshot:
        """Aggregate the order map into top-of-book and depth."""
        bids: dict[float, int] = {}
        asks: dict[float, int] = {}
        nb = na = 0
        for price, side, qty in self.orders.values():
            if qty <= 0:
                continue
            if side == "buy":
                bids[price] = bids.get(price, 0) + qty
                nb += 1
            else:
                asks[price] = asks.get(price, 0) + qty
                na += 1

        bb = sorted(bids, reverse=True)[:levels] if bids else []
        aa = sorted(asks)[:levels] if asks else []
        return BookSnapshot(
            symbol=self.symbol,
            best_bid=bb[0] if bb else 0.0,
            best_ask=aa[0] if aa else 0.0,
            bid_qty=bids.get(bb[0], 0) if bb else 0,
            ask_qty=asks.get(aa[0], 0) if aa else 0,
            bid_depth5=sum(bids[p] for p in bb),
            ask_depth5=sum(asks[p] for p in aa),
            n_bid_orders=nb,
            n_ask_orders=na,
            bid_levels=len(bids),
            ask_levels=len(asks),
        )

    def reset_flow(self) -> None:
        """Clear per-slot flow counters after a snapshot is taken."""
        self.trades = self.traded_qty = self.buy_traded = self.sell_traded = 0


def replay_events(events: list[tuple], slot_ns: int, levels: int = 5) -> list[dict]:
    """Replay merged events and snapshot the book at each slot boundary.

    Parameters
    ----------
    events:
        Tuples of ``(channel, seq, arrival_ts, symbol, exchange, kind, price,
        qty, side, key, buy_key, sell_key)`` where ``kind`` is ``add`` /
        ``cancel`` / ``trade``. **Must already be sorted by
        ``(channel, seq)``** -- see the module docstring for why timestamp order
        is not a substitute.

    Returns
    -------
    One dict per ``(slot, symbol)`` with book state and the flow that occurred
    within the slot.

    Notes
    -----
    Snapshots are taken at the slot's **closing** edge and describe the book as
    of that instant, so the row labelled ``t`` is knowable at ``t`` and may be
    read by a feature at ``t``.
    """
    books: dict[str, Book] = {}
    out: list[dict] = []
    cur_slot: int | None = None

    def flush(slot: int) -> None:
        for sym, bk in books.items():
            if not bk.orders and bk.trades == 0:
                continue
            s = bk.snapshot(levels)
            out.append({
                "slot": slot,
                "symbol": sym,
                "exchange": bk.exchange,
                "best_bid": s.best_bid,
                "best_ask": s.best_ask,
                "spread": s.spread,
                "mid": s.mid,
                "microprice": s.microprice,
                "bid_qty": s.bid_qty,
                "ask_qty": s.ask_qty,
                "bid_depth5": s.bid_depth5,
                "ask_depth5": s.ask_depth5,
                "imbalance_l1": s.imbalance_l1,
                "imbalance_l5": s.imbalance_l5,
                "n_bid_orders": s.n_bid_orders,
                "n_ask_orders": s.n_ask_orders,
                "bid_levels": s.bid_levels,
                "ask_levels": s.ask_levels,
                "slot_trades": bk.trades,
                "slot_qty": bk.traded_qty,
                "slot_buy_qty": bk.buy_traded,
                "slot_sell_qty": bk.sell_traded,
                "unresolved": bk.unresolved,
            })
            bk.reset_flow()

    for ev in events:
        (ch, seq, ts, sym, ven, kind, price, qty, side, key, bkey, skey) = ev
        slot = ((ts // slot_ns) + 1) * slot_ns
        if cur_slot is None:
            cur_slot = slot
        elif slot != cur_slot:
            flush(cur_slot)
            cur_slot = slot

        bk = books.get(sym)
        if bk is None:
            bk = books[sym] = Book(symbol=sym, exchange=ven)

        if kind == "add":
            bk.add(key, price, side, qty)
        elif kind == "cancel":
            bk.cancel(key, qty)
        else:
            bk.execute(bkey, skey, qty, side)

    if cur_slot is not None:
        flush(cur_slot)
    return out
