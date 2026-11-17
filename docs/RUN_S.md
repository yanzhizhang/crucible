# run_s -- PM's runS, reconstructed

This is crucible's research record of the PM deep dive: the evidence, the filled-in configuration, the reconstructed
rule, and the finance behind it. The production rewrite lives in **sht-demo `main`** at
`chrono/cpp/strategy_plugins/run_s/run_s.cpp` (pushed 2026-11-16, commit 2e376ac). Change the plugin there; keep this
note in step with it. **Evidence** marks what was read from the binary or fitted on PM's logs. **Guess** marks a
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
| `sim_et` | 145900000 | end, 14:59:00.000 (chrono's own risk window stops at 14:55). From the 97 config2 (/data/sht-stg-lib, 11-16); the newer 224 copy (11-17, different md5) is unread |
| `cr` | 0.0004 | **cost rate per side, 4 bp**. Guess: commission 1.5 bp + transfer 0.64 bp + half of the 5 bp sell stamp = 4.6 bp per side, or about 9 bp round trip. |
| `output_dir` | .../sim_example/HS300 | where runS writes its own logs / dumps |
| `alg_cfg` | .../sim_example/sc.p3.json | the sizing parameters. **Read on 224 (2026-11-17)**: see section 2b |
| `ahubs` | the top-level key of sc.p3.json | a list of per-alpha parameter sets (one entry, for alpha.1); not a trader key |
| `fan` / `faw` | ["pqAlpha"] / [1.0] | one alpha, weight 1 |
| `fac` | {alpha_name "alpha.1", alpha_dur 0.03} | the alpha's name in the signal, and its horizon. Guess: 0.03 of a 4 h trading day = about 7 minutes. |

The sizing parameters, from the binary (R133 .rodata) and PM's decision logs on 224 (R33, R72):

| PM name | value | source |
|---|---|---|
| participation cap | 0.15 | `0.15` and `/0.85` in `Alg::onInnerMsg`; winsdQty / emsMktVol = 3/17 exactly (R33) |
| `winsdQty` | emsMktVol x 0.15/0.85 | the most we may trade in the window and stay at 15 % of volume |
| `asf` | 0.8 | constant on every decision (R33) |
| `asq` | asf x winsdQty - emsOurVol, in lots | p50 100, p90 3900 shares (R33) |
| `lthr` | -0.000182 - 0.2208 x hs | fit on 224 (R72); 0 in the first minute (09:30-09:31, R33). Also fits -614 x vpen^2 (R^2 0.93, stronger); vpen's ingredients are unknown, so run_s uses the hs form |
| size | 0.87 x (base - qty) + 0.07 x fa x winsdQty | fit on 224 after 09:31 (R^2 0.87): a continuous pull to base plus an alpha tilt, capped at asq |
| price | ~64 % far touch, rest near touch / inside | 224 orders; the split rule is unknown |
| `hs` | half spread / mid | "~half a tick over the price" (PM_REPRO) |
| `pos`, `pp`, `cpos` | holding - base, pending, pos + pp | exact for 99.9 % / 100 % (R33) |
| `vpen` | **1e-07 before 13:00, then 0.001 x sqrt((t-13:00)/(14:59-13:00))** | identity from the decision logs (R150 / R151): exactly the cfg value before rvts, same for every stock each second, sqrt ramp matches all 23 five-minute medians to 0.1 % |
| EMS windows | 20 / 60 / 30 s | constants in `ct::EMA`, `Alg::onInnerMsg` (R133); which window is which is a guess |
| cancel | ~5.3 s median, retry on failure | 224 orders (6 s was one sample in R33); OnFCM retries |
| re-place | ~0.7 s after the cancel, 95 % same side, often before the ack | 224 orders |
| send gate | only ~2.6 % of gated rows send (classifier AUC 0.65) | rule unknown |
| stale book | skip if book age > 10 s | "book->epoch {} - book->srcEpoch {} > 10" (R136) |
| odd lots / rounding | 100 | `round100`, `roundBy` (stage 1) |

## 2b. PM's alg_cfg, sc.p3.json (read on 224)

`/tmp/zyq/sht-stg-lib/cfg/sim_example/sc.p3.json`, 270 B, mtime 2026-11-17 13:13 (read in place, not copied):

```json
{ "ahubs": [ { "tqub": 10000000, "nmvcst": 10000000, "tpub": 0.09, "adur": 12,
               "vpen": 1e-07, "rvts": 130000000, "asf": 0.8, "neg_nmvcst": false } ] }
```

| key | value | evidence | meaning (status) | run_s |
|---|---|---|---|---|
| `asf` | 0.8 | = the logged asf (R33) | safety factor on the participation cap (**known**) | used |
| `vpen` | 1e-07 | = the logged vpen before 13:00 (R150) | inventory-penalty base; from `rvts` it ramps to 0.001 by `sim_et` (**known**, R151) | used (bang mode) |
| `tpub` | 0.09 | = the 0.09 stored by `Alg::Alg` (R133) | unknown; maybe a turnover / position upper bound | read, logged, unused |
| `tqub` | 1e7 | -- | **guess**: total quantity upper bound (shares per day) | applied as a cap |
| `nmvcst` | 1e7 | = the 10M capital | **guess**: cap on the net market value of the gap (CNY) | applied as a cap |
| `neg_nmvcst` | false | -- | **guess**: whether a net negative value is allowed | read, logged, unused |
| `rvts` | 130000000 | HHMMSSmmm = 13:00:00.000 | start of the vpen ramp (**known**, R150 / R151) | used |
| `adur` | 12 | `Alg` logs "alpha adur: {}" | alpha duration; units unknown | read, logged, unused |

Both caps only ever reduce trading, and at 1e7 they don't bind for a 10M book, so applying them is safe whatever they
mean exactly.

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
cap  = base                       GUESS (may sell the whole base, or buy up to one more base)

size_mode "fit" (default, fitted on 224):
  trade = clamp( 0.87 * (-cgap) + 0.07 * fa * winsdQty , -asq, +asq )
  send if it closes the gap, or if fa clears the threshold on its side (buy: fa > -lthr, sell: fa < lthr)
size_mode "bang" (first draft, a deliberate simplification):
  fa > -lthr and fa > hs  -> buy  min(asq, cap - cgap);  fa < lthr and fa < -hs -> sell min(asq, cgap + cap)
  else if |fa| < vpen (1e-07) -> walk the gap back by min(asq, |cgap|)
sells are capped at sellable - pending sells (T+1); whole lots except a sale that empties the line
after 14:45: only trades that shrink |gap|
price: far touch if fa beyond the threshold >= cross_k * hs (GUESS, default 1), else the near touch
cancel at 5.3 s, then re-decide (the re-place chain):
  replace_mode "ack" (default): at max(cancel + 0.7 s, the old order's terminal report). chrono keeps the old order
    fully reserved until that report, so an earlier same-side re-place, and any re-sell of the same shares, would be
    rejected. The ack normally arrives within tens of ms, so this costs almost nothing.
  replace_mode "fixed" (PM-faithful): at cancel + 0.7 s, before the ack; a rejected re-place retries on the old
    order's terminal report rather than ending the chain
  each chain step costs 2 (cancel + insert) on chrono's order_rate and daily_order_limit: size the limit for it
at most one decision per stock per 3 s (GUESS: PM's send gate is unknown)
```

The fitted size explains two things in PM's logs:
- Sizes are often +-asq only because the gap is usually large.
- Order size follows the gap, not the prediction: the alpha tilt 0.07 x fa x winsdQty is a few shares to a few hundred.

**What is a guess** (each is a named parameter, to be replaced by sc.p3.json or a fit):
- the send gate (`min_order_gap_ms`). This is the biggest behavioural difference: PM sends on only ~2.6 % of the rows that pass its gate.
- the cross / join split (`cross_k`)
- the meaning of sc.p3.json's `tqub` / `nmvcst` (applied as caps) and of `tpub`, `adur`, `rvts`, `neg_nmvcst` (unused)
- cap = base (`gap_frac`)
- the EMS window (`ems_tau_s`)
- the hs form of `lthr` instead of PM's vpen^2 form

Not modelled: PM's `vr` (a volatility ratio, p50 0.18) and `poscst` (position cost) terms.

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
   - PM prices that risk with a penalty coefficient `vpen` (1e-07, a constant in sc.p3.json); the logged vpen still varies (to ~1e-3), so the logged value is probably the coefficient after `PosClzer::GVpen` scales it (by the position or time) -- an inference, not yet checked. What pulls the book back by the close is the gap-closing size rule and the end-of-day unwind; `rvts` (13:00) may switch on an afternoon mode.
4. **The size is mostly a pull back to base, with a small alpha tilt.**
   - The fitted rule is 0.87 x (base - qty) + 0.07 x fa x winsdQty.
   - The model predicts a few bp over minutes (PM's alpha std is 3.4e-4). So the alpha decides *when* and *which side* the gap is worked (through the threshold and the gate) far more than *how much*.
   - The base target, which moves daily with the index weights, sets how much.
   - In effect the strategy executes the daily rebalance toward base, and times it with the alpha.
5. **Fast cancel-and-re-place (about 5.3 s, then 0.7 s)** follows from the alpha horizon (`alpha_dur`, minutes). An order that hasn't filled within seconds is trading on stale information, so it is pulled and re-priced from the current book.
6. **It works on CSI 300** because those names are liquid enough for 15 % of their volume to be a meaningful trade, and because their high retail participation makes their short-term moves predictable.
   - crucible's measurement agrees: at tick scale, 9-second moves continue over the next 30 seconds.

**Differences from PM, on purpose:**
- Everything runs on chrono's core thread. PM's `LiveEs` places orders from its own `Engine` thread, which races with chrono's core-thread tables (the chrono session's finding).
- No disk reads in `initialize()`. PM sleeps 2 s and loads stsfile; our plugin needs only chrono's position, book and signal feeds.

## 5. Open items

- sc.p3.json is now known (section 2b). Still open: what `tpub`, `adur`, `rvts`, `neg_nmvcst` do, and the exact meaning of `tqub` / `nmvcst`.
- The `vr` / `poscst` terms (PM logs them per decision); with vpen known they can be fitted from the 224 decision logs.
- The `executp` target formula. With sc.p3.json and the decision logs it can be fitted directly. The logs give `fa`, `vpen`, `vr`, `poscst`, `cpos`, `asq` and `hs` per decision.
- A replay test of `run_s` against PM's day: same signals, compare order times, sides and sizes with PM's `place_order` lines on 224.
