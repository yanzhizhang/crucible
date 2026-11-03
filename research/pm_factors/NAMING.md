# PM naming: what the letters seem to mean

`ZZ` is a common prefix on every name (meaning unknown). Columns: what the data shows, the reading of the two letters
(pinyin initials or English), confidence. An empty reading = no idea yet. A trailing digit (`zzro2`, `zzgh2`) is a variant.

## Letter conventions inside a family

| letter | meaning | confidence |
| --- | --- | --- |
| `n` / `q` | notional (CNY) / quantity (shares) | high (ZZGW, ZZXC, ZZMS, ZZTS) |
| `a` / `b` | ask (sell) side / bid (buy) side | medium |
| `ab`, `cd`, `ef` | sum of the pair | high (ZZGW identities) |
| `c`/`d`, `e`/`f` | second / third side pair at a higher threshold | medium (ZZUG) |
| `m` / `s` | mean / std | low |
| `c h l o` | close high low open | high (ZZDS) |
| `p` / `r` (engine) | price-type / range- or return-type average | medium |
| `S` / `T` (engine) | snapshot clock / trade clock | high |
| `f` / `s` (zzup) | fast (300 s) / slow (600 s) | high |
| `.t` | time of the last update | high |

## Day-end state families (engine)

| name | what it is (data) | reading | confidence |
| --- | --- | --- | --- |
| zzpk | day high / low / open, price EMA 300 s | peak | high |
| zzvx | true-range EMAs 30 s, rEma = 1 | volatility (VIX-like) | high |
| zzbd | price / return EMA on snapshot (S) and trade (T) clocks | 波动 bodong | medium |
| zzup | price EMA fast 300 s / slow 600 s | up (trend) | low |
| zzqj | volume, turnover, running VWAP | 区间 / 全价 ? | low |
| zzmh | highest snapshot price, distance from it squared | max high | medium |
| zzba | both-clock EMA of the snapshot price, range | bid-ask ? | low |
| zzxu | price EMA 300 s, range EMA | | |
| zzeb | mid EMA 100 s, snapshot-gap EMA | | |
| zzab2 | price EMA 300 s | | |
| zzcb | spread-like, falls with price | 差 / 成本 ? | low |
| zztx | trade-clock EMA 100 s | | |
| zzfb | 20 s grid of cumulative amount | 分笔 fenbi | medium |
| zzgh | 15 s grid, buy-sell imbalance in 5 codes; nb = net buy | | |
| zzjb | 15 s grid flag: price 3.04% below previous close | 警报 jingbao ? | low |
| zzhj | quarter-hour mids; daily EMA of a volume-like value | | |
| zzxo / zzui | 10 s price grids to 14:57 | | |
| zzvm | volume and turnover | volume ... | low |
| cdc.lp | last price | last price | high |

## Model feature families (lgb_fn_v1_209)

| name | what it is (distribution / IC match) | reading | confidence |
| --- | --- | --- | --- |
| zzro / zzro2 | signed returns, three horizons a / b / c; largest gain | return ... / rate of change | low |
| zzeu / zzeu2 | bounded ±1, level-1 book imbalance | | |
| zzef | 10-30 s traded-amount imbalance | | |
| zzpe | 30 s traded-amount imbalance | | |
| zzie3 | 30-60 s volume imbalance | imbalance ... | low |
| zzyg2 | level-1 bid / ask imbalance | 一挂 yigua ? | low |
| zzjx3 | -log(price / VWAP) | 均线 junxian | medium |
| zzzl2 | return against the previous close | 涨落 / 涨率 zhang | medium |
| zzfe | 600 s trade-count imbalance | | |
| zzfb2 | abs full-depth book imbalance | 分笔 ? | low |
| zzeg | ratio near 1 | | |
| zzel | | | |
| zzxo | sits at -15.5 | | |

## samplerS / samplerR families (hstats)

Mechanism: each 5-minute cell = median over the last N trading days (N > 20), unscaled.

| name | what it is (data) | reading | confidence |
| --- | --- | --- | --- |
| ZZTS | q = median buy-initiated amount per bar | taker side ? | low |
| ZZXC | n / q = median amount / volume per bar | 成交 ? / execution | low |
| ZZMS | n / q ≈ cumulative amount / volume; bl / bm / bs unknown | money by size ? | low |
| ZZAL | an / bn ≈ ask / bid total book amount; da / db imbalance | ask levels | medium |
| ZZDS | c h l o ≈ VWAP / high / low (daily); mb / ms / msr / vr | daily stats | medium |
| ZZGW | na / nb / qa / qb new-order notional / quantity, ab = a + b | 挂委 guawei ? | low |
| ZZQO | q ≈ order volume (ratio 1.4-2.2) | quantity of orders | low |
| ZZUG / ZZUG2 | large-order thresholds (size quantiles per side) | | |
| ZZUGSS | summary stats of ZZUG (m, s) | | |
| ZZQI | scale-free, cumulative | queue imbalance ? | low |
| ZZVC | a-d size stats; e = 25 x snapshot count, capped 10000 | | |
| ZZWA | an / bn ≈ 0.003 (relative-spread scale), aq / bq ≈ 0.3 | weighted ask ? | low |
| ZZHL | n ≈ SSE order count | | |
| ZZSQ | daily, a / ab / b | | |
| ZZAC | samplerR: as / bs size quantiles per 5-minute bar | | |
| ZZCR | samplerR: as / ps / ab quantiles per day | | |
