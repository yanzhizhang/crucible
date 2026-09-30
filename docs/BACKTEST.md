# Stage 7 -- high-frequency backtest (`research/bt/`)

A market-by-order replay that answers one question: **where does a signal's paper PnL go** once
it meets latency, the spread, the queue, fees and T+1? It does not depend on the PM's formulas;
the signal is a placeholder (1-minute flow-imbalance momentum, `bt/strategy.py`) until stage 6
produces real predictions.

## Pipeline

| step | module | output |
|---|---|---|
| raw order / transaction / quotation -> one MBO event stream per stock, venue-normalised, SZSE market orders tagged | `bt/events.py` | `/work/crucible_data/store/bt_events/date=D/{events,limits}.parquet` |
| replay the book, simulate trips (numba, one stock per call, GIL released) | `bt/engine.py` | fills matrix |
| fills -> CNY, fees (`toll.CostModel`), per-fill PnL decomposition | `bt/pnl.py` | |
| realism ladder + latency sweep over a universe, thread pool | `bt/run.py` | `/work/crucible_data/store/bt/date=D/{fills,summary,sweep}.parquet` |
| replay audit: rebuilt level 1 vs exchange snapshots | `bt/audit_book.py` | match rates |
| venue / instrument rules | `bt/venues.py` | |

## Venue semantics (why SSE and SZSE replay differently)

| | SSE (XSHG) | SZSE (XSHE) |
|---|---|---|
| order key | `order_id` | `seq_id` |
| cancels ride | the order stream (`update_type 2`) | the trade stream (`trade_type 2`) |
| marketable orders | published only *after* matching, as the resting remainder | published at entry, *before* their trades |
| market orders | never appear (only their remainder, priced) | `orderType 1`, four official types merged into one |
| trade aggressor | exchange BS flag | leg with the larger sequence number |
| replay rule | apply records as published | hold a marketable order out of the book until its own trades/cancel end, then rest the remainder |

SZSE market orders get a subtype tag from a vectorised pre-pass (`szse_market_tags`, mirroring
the prototype's `classify_szse_core`, 100 ms auto-cancel window): FAK and zero-fill FOK never rest,
counterparty-best remainders rest at their fill price. SSE records carry nothing to deduce a
FAK / FOK from, so they are replayed as they are -- the same split the prototype makes.

**Continuous trading only.** Both venues run continuous matching 09:30-11:30 and 13:00-14:57.
Outside it the replayed book holds call-auction orders and is crossed. The engine never matches
against it: an entry arriving outside a window is skipped (`skip_session`), an exit or a passive
cross waits for the next window (`defer_session`), and a trip with no window left is marked to
the closing mid. `_walk` additionally refuses a crossed true book (`walk_crossed`, expected 0).

Auction tie-break (only matters once simulated orders join an auction): SZSE takes the tick
nearest the reference price over the whole grid; SSE takes the midpoint of the tied submitted
prices, rounded up (prototype `me_types.hpp`, `auction_price_rule`).

Extending to ETFs / convertible bonds / derivatives = a new `Rules` entry (tick, lot, T+0/T+1,
stamp duty, sessions) and, if the flow differs, a new engine dispatch code. ETF and CB entries
are registered with their to-do list and refuse to run until implemented.

## Replay audit

Rebuilt best bid/ask vs the exchange's continuous-session snapshots, allowing for snapshot lag
(SZSE snapshots are stamped to the second and trail the book by up to ~2 s; SSE within 1 s).
20260615: 600519 100 %, 601318 99.9 %, 000001 100 %, 300750 99.96 %, 0 unresolved references.

## Trip model

intent (decision time = bar `available_ts`) -> + `lat_sig` -> order reaches the exchange after
`lat_send` -> entry (aggressive walk, max 10 ticks, or passive join at the own best) -> hold 5 min
after the fill is *known* (`lat_report`) -> aggressive exit, retried each second while liquidity
is missing. One open trip per stock. Orders are virtual: they never move the book.

Two books exist at every instant: the **known** book (what our process had received by then, by
arrival time) and the **true** book (the exchange's state when our order arrives). Decisions use
the known book, fills use the true one.

Passive queue: joining behind everything already resting at the price; trades at that price
consume the queue ahead first, cancels of orders ahead shrink it, a trade through our price fills
us. Unfilled after 30 s -> cross the rest.

## PnL decomposition (exact, per fill)

For a fill of signed side `s`, qty `q`, price `p`, known mid at decision `m0`, true mid on arrival
`m1`, mid at trip end `m_end`:

```
total   = s*q*(m_end - p) - fee
signal  = s*q*(m_end - m0)        what the idea was worth
latency = -s*q*(m1 - m0)          the market moved while we decided and sent
spread  = -s*q*(p - m1)           crossing / queue (positive only for passive fills)
fees    = -fee
```

`signal + latency + spread + fees == total` to the cent; restore legs (T+1 close-out) are their own bucket.

## Realism ladder

| config | fill | latency | queue | fees | T+1 |
|---|---|---|---|---|---|
| 1_ideal_mid | known mid | 0 | -- | -- | -- |
| 2_cross_spread | known touch | 0 | -- | -- | -- |
| 3_latency_true_book | walk true book | 1 + 3 + 3 ms | -- | -- | -- |
| 4_passive_queue | passive join, cross after 30 s | 1 + 3 + 3 ms | yes | -- | -- |
| 5_fees | as 4 | as 4 | yes | yes | -- |
| 6_t1_restore | as 4 | as 4 | yes | yes | sell legs from base, close-out at 14:56 |

Configs 1-2 ignore liquidity entirely (they fill at a quote even when nobody is there), so they
can make trips the exchange model cannot -- e.g. buying back a short on a stock locked at limit-up.

## Results (top 30 CSI 300 names by traded amount, 100k CNY per trip, placeholder signal)

CNY. `walk_crossed` = 0 and `unresolved_passive` = 0 in every config on both days.

| config | 20260615 total | signal | latency | spread | fees | 20260805 total | signal | latency | spread | fees |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1_ideal_mid | 5,590 | 5,590 | 0 | 0 | 0 | 21,052 | 21,052 | 0 | 0 | 0 |
| 2_cross_spread | -13,936 | 5,590 | 0 | -19,526 | 0 | 2,429 | 21,052 | 0 | -18,623 | 0 |
| 3_latency_true_book | -10,510 | 7,943 | -258 | -18,195 | 0 | 1,048 | 17,339 | 142 | -16,432 | 0 |
| 4_passive_queue | -4,439 | 3,198 | -349 | -7,287 | 0 | 4,972 | 10,611 | -23 | -5,617 | 0 |
| 5_fees | -40,878 | 3,198 | -349 | -7,287 | -36,439 | -23,339 | 10,611 | -23 | -5,617 | -28,311 |
| 6_t1_restore | -39,838 | 3,556 | -349 | -6,929 | -36,115 | -23,336 | 8,437 | -23 | -4,514 | -27,237 |

Trips: 412 / 350 / 349 / 349 / 346 (0615), 328 / 283 / 282 / 282 / 272 (0805). Passive entries:
~73 % filled in the queue within 30 s, the rest crossed.

What it says:

- **The spread is the first killer, fees the second.** Crossing costs ~19k a day on ~70M CNY
  of turnover (~2.7 bp); fees are ~5 bp of turnover (stamp duty on the sell half + commission),
  which is several times the signal's gross edge. A 1-minute flow signal held 5 min does not pay
  for itself; the PM's model has to clear ~7-8 bp per round trip before it is worth trading.
- **Passive entries halve the spread cost but give back signal** (0805: signal 17.3k -> 10.6k):
  the orders that fill are the ones the market came to -- adverse selection, visible directly in
  the decomposition.
- **Latency does not matter at this horizon**: 0-300 ms moves the total by ~1k, inside the noise.
  It will matter for a signal that lives seconds, not minutes.
- **T+1 binds on 0805** (40 intents skipped: the sell legs would exceed today's base), and the
  restore bucket is 0 because the hold always ends before the 14:56 close-out.

## Bugs found by the ladder

- **Positive spread on aggressive fills** (config 3 showed +14.8k on 5 stocks): exits scheduled
  after 14:57 walked the closing-auction book, which is crossed (sell orders at the down limit
  under a limit-up bid). Fixed by the continuous-session gate above.
- SZSE transient crossing: marketable orders are published before their trades; placing them on
  arrival offered virtual orders liquidity that never existed. Fixed by holding them out of the
  book until their burst ends.
