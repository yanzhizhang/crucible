# run_s -- PM's runS, reconstructed

Our rewrite lives in sht-demo branch `feat/run-s`, `chrono/cpp/strategy_plugins/run_s/run_s.cpp` (worktree
`~/Repo/sht-demo-runs`, local only). This note gives the evidence, the filled-in configuration, the reconstructed rule,
and the finance behind it. **Evidence** marks what was read from the binary or fitted on PM's logs. **Guess** marks a
hypothesis.

## 1. What runS is

| | evidence |
|---|---|
| chrono strategy plugin, type `runS`, class `cts::qlive::runS` | `chrono_plugin` data symbol + `describe<cts::qlive::runS>` (so_probe stage 1) |
| hooks: initialize, on_lv2, on_signal, on_trade_event, on_order_changed, on_error_cancel, on_position | vtable, 12 words, same order as today's `strategy.hpp` (R134) |
| built against chrono's own headers | its log overflow text is chrono's `log_sink` "plugin log line dropped: args exceed {} bytes" |
| gcc 11.5 -O3, C++17, no DWARF for PM code | R135 |
| pipeline: on_signal -> signal2alpha -> Engine (own thread, TimerLoop) -> qAlpEngine -> sol::Alg::onInnerMsg -> LiveEs::PlaceExOrd -> runS::place_order -> chrono | class map + call sites (stage 1, R134) |

## 2. Configuration, filled in

`config2_HS300.json` (read on 97, R133) plus the keys `AlgW::loadCfgs` reads (R137):

| key | value | meaning |
|---|---|---|
| `date` | 20261116 | trading day |
| `univ` | HS300 | universe |
| `NumLevels` / `depthLevel` | 10 | L2 depth used |
| `prod` | true | live, not sim |
| `dataPath` | /tmp/s/stsfile | daily static files: index / ex / sid2symbol / stats/sod (R33) |
| `sim_st` | 93000000 | start of trading, HHMMSSmmm = 09:30:00.000 |
| `sim_et` | 145900000 | end, 14:59:00.000 (chrono's own risk window stops at 14:55) |
| `cr` | 0.0004 | **cost rate per side, 4 bp**. Guess: commission 1.5 bp + transfer 0.64 bp + half of the 5 bp sell stamp = 4.6 bp per side, or about 9 bp round trip. |
| `output_dir` | .../sim_example/HS300 | where runS writes its own logs / dumps |
| `alg_cfg` | .../sim_example/sc.p3.json | the sizing parameters; **not on 97**, so its values are fitted below |
| `ahubs` | absent -> default | **guess**: "alpha hubs", extra alpha sources merged with `fan`; empty = only the configured alpha |
| `fan` / `faw` | ["pqAlpha"] / [1.0] | one alpha, weight 1 |
| `fac` | {alpha_name "alpha.1", alpha_dur 0.03} | the alpha's name in the signal, and its horizon. Guess: 0.03 of a 4 h trading day = about 7 minutes. |

The sizing parameters, from the binary (R133 .rodata) and PM's decision logs on 224 (R33, R72):

| PM name | value | source |
|---|---|---|
| participation cap | 0.15 | `0.15` and `/0.85` in `Alg::onInnerMsg`; winsdQty / emsMktVol = 3/17 exactly (R33) |
| `winsdQty` | emsMktVol x 0.15/0.85 | the most we may trade in the window and stay at 15 % of volume |
| `asf` | 0.8 | constant on every decision (R33) |
| `asq` | asf x winsdQty - emsOurVol, in lots | p50 100, p90 3900 shares (R33) |
| `lthr` | -0.000182 - 0.2208 x hs | fit on 224 (R72); 0 in the first minute (09:30-09:31, R33) |
| `hs` | half spread / mid | "~half a tick over the price" (PM_REPRO) |
| `pos`, `pp`, `cpos` | holding - base, pending, pos + pp | exact for 99.9 % / 100 % (R33) |
| `vpen` | 1e-07 .. 1e-03 through the day | log p1..p99 (R72); `PosClzer::GVpen` constants 0.001 and 1 (R133) |
| EMS windows | 20 / 60 / 30 s | constants in `ct::EMA`, `Alg::onInnerMsg` (R133); which window is which is a guess |
| cancel | ~6 s, retry on failure | placed 09:30:00.45 -> cancel acknowledged 09:30:06.5 (R33); OnFCM retries |
| stale book | skip if book age > 10 s | "book->epoch {} - book->srcEpoch {} > 10" (R136) |
| odd lots / rounding | 100 | `round100`, `roundBy` (stage 1) |

## 3. The rule

For each prediction `fa` on stock i, at time t inside [sim_st, sim_et]:

```
mp   = (bid1 + ask1) / 2
hs   = (ask1 - bid1) / 2 / mp
lthr = 0                          for the first minute
     = -0.000182 - 0.2208 * hs    afterwards                (<= 0)
gap  = holding - base             (PM pos)
cgap = gap + pending              (PM cpos)
asq  = lots( 0.8 * emsMktVol * 0.15/0.85 - emsOurVol )      (capacity left in the window)
cap  = base                       (may sell the whole base, or buy up to one more base)

if   fa >  -lthr and fa >  hs and cgap <  cap :  buy   min(asq, cap - cgap)   at ask1
elif fa <   lthr and fa < -hs and cgap > -cap :  sell  min(asq, cgap + cap, sellable)   at bid1
elif cgap != 0 and |fa| < vpen(t)             :  trade -sign(cgap) * min(asq, |cgap|)  (walk the gap back to base)
after 14:45: only trades that shrink |gap|
cancel anything still working after 6 s
```

PM's logs show these properties. The rule reproduces them by construction:
- Orders are bang-bang at `executp` ~ +-asq, mostly at `lthr` = 0 near the open.
- Order size follows the inventory gap, not the size of `fa` (corr 0.13).

What it does not reproduce exactly: PM's `executp` is a continuous number (R^2 0.29 for the best linear fit), so PM
computes a target from `fa`, `vpen`, `vr`, `poscst` and then caps it. The `vr` (a volatility ratio, p50 0.18) and
`poscst` (position cost) terms are not in the rewrite. Once sc.p3.json is available they replace the simplified
`|fa| < vpen` reversion test.

## 4. The finance: why this algorithm exists

**It is T+0 trading around a base position (底仓做T).**
- The fund holds a CSI 300 base: 9M of 10M CNY, at index weights, rolled daily by `base_position.py`.
- A-shares settle T+1, so shares bought today cannot be sold today. A pure intraday long-short book is impossible in cash equity.
- The workaround: own the shares already, and treat the base as an inventory.
  - When the short-horizon model says a stock will fall, **sell base shares now**.
  - When it turns positive (or by the close), **buy them back**.
  - The day ends near the base again. The P&L is the round-trip spread captured, net of costs, on top of the base's index exposure. That exposure is hedged or simply held.

Each piece of the rule follows from that:
1. **The threshold `lthr` is a cost hurdle.** A round trip costs about 9 bp in fees (`cr` 4 bp per side, stamp on the sell) plus crossing half the spread twice. Trading pays only if the predicted move beats the spread and fees, so the threshold widens with the spread (`-0.22 x hs`) and carries a fixed fee floor (`-1.8 bp`).
   - At the open the spread and the alpha are both largest, and PM drops the threshold to 0. That is where most of its orders happen.
2. **The 15 % participation cap is for market impact and capacity.**
   - Impact grows with the share of volume you take (the square-root law). Capping at 15 % of a moving window keeps impact roughly constant and below the alpha.
   - Counting our own volume (0.15 / 0.85, not 0.15) stops the strategy from feeding on its own prints.
   - `asf` 0.8 is a safety margin under that cap.
3. **The inventory penalty is for overnight and T+1 risk.**
   - A gap left at the close is an unintended position: sold shares can only be re-bought today, and bought shares are locked until tomorrow.
   - So the penalty on carrying a gap is tiny in the morning (vpen 1e-7) and rises into the close (to 1), forcing the book back to its base.
4. **Bang-bang at the cap** is what an alpha that is a sign more than a size implies.
   - The model predicts a few bp over minutes (PM's alpha std is 3.4e-4).
   - Once the edge clears the cost, the best response is to trade as much as capacity allows, not proportionally to `fa`.
5. **Fast cancels (about 6 s)** follow from the alpha horizon (`alpha_dur`, minutes): an order that has not filled within seconds is trading on stale information.
6. **It works on CSI 300** because those names are liquid enough for 15 % of their volume to be a meaningful trade, and because their high retail participation makes their short-term moves predictable.
   - crucible's measurement agrees: at tick scale, 9-second moves continue over the next 30 seconds.

**Differences from PM, on purpose:**
- Everything runs on chrono's core thread. PM's `LiveEs` places orders from its own `Engine` thread, which races with chrono's core-thread tables (the chrono session's finding).
- No disk reads in `initialize()`. PM sleeps 2 s and loads stsfile; our plugin needs only chrono's position, book and signal feeds.

## 5. Open items

- sc.p3.json: the real values of the sizing parameters (vpen schedule, vr / poscst terms, EMS windows). This needs your decision on how it reaches 97.
- `ahubs` meaning: ask PM.
- The `executp` target formula. With sc.p3.json and the decision logs it can be fitted directly. The logs give `fa`, `vpen`, `vr`, `poscst`, `cpos`, `asq` and `hs` per decision.
- A replay test of `run_s` against PM's day: same signals, compare order times, sides and sizes with PM's `place_order` lines on 224.
