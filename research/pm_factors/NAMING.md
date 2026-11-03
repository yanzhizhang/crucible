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

## Model feature families (lgb_fn_v1_209: 209 features, 68 families)

"data" = what the state rule, the training CDF or the IC match shows; "name only" = a reading of the letters with no
data behind it. Gain = share of the model's split gain (202509 worklist), where photographed.

| family | features | data | reading | confidence |
| --- | --- | --- | --- | --- |
| ro | ro.a/b/e, ro2.a/b/c | signed returns, several horizons (ro2.a ~45 s micro-price, ro2.c ~240 s); gain 0.129 | return | low |
| qf | qf, qf2-qf10 x q.a/b/c1-c4 | unknown; gain 0.059; a window index 2-10 times a cut c1-c4 | | |
| zl | zl.d/f, zl2.b-i | return against the previous close; gain 0.056 | 涨落 / 涨率 | medium |
| jx | jx.a, jx2.a, jx3.a/b | -log(price / VWAP) | 均线 | medium |
| eu | eu, eu2 | level-1 book imbalance, bounded +-1 | | |
| eg | eg, eg2 | ratio near 1 | | |
| ef | ef, ef2 | 10-30 s traded-amount imbalance | | |
| pe | pe.a-e | 30 s traded-amount imbalance (pe.c) | | |
| ie | ie.c, ie2-ie5.c | 30-60 s volume imbalance (ie3.c) | imbalance | low |
| fe | fe, fe2 | 600 s trade-count imbalance | | |
| yg | yg.0.a, yg.1.a, yg2.0.b | level-1 bid / ask imbalance; .0 / .1 look like a book level | | |
| up | up | price / fast EMA (zzup.pf) | up | low |
| pk | pk | peak values (high / low / open) | peak | high |
| vx | vx.a | >= 1, p10 = 1: a range ratio, unsolved | volatility | medium |
| xu | xu.a/b | price / range EMA state | | |
| mh | mh | highest snapshot price, distance from it | max high | medium |
| ba | ba | both-clock EMA of the snapshot price | | |
| ab | ab, ab2 | price EMA 300 s state | | |
| cb | cb | spread-like, falls with price | | |
| fb | fb.b, fb2.a | fb: 20 s cumulative-amount grid; fb2.a = abs full-depth imbalance | 分笔 | low |
| gh | gh.b, gh2.b | integer count (thresholds ~ trades a day); 15 / 10 s imbalance codes | | |
| jb | jb.b, jb2.b | 3.04 % drop flag grid | | |
| hj | hj | quarter-hour mid samples | | |
| xo | xo | sits at -15.5 | | |
| vm | vm, vm2 | ratio near 1; volume / turnover state | | |
| al | al.a-h | samplerS ZZAL (book amount profile); al.d ~ 600 s volume imbalance | ask levels | medium |
| wa | wa.a-k | samplerS ZZWA (relative-spread scale) | | |
| ts | ts | samplerS ZZTS (median buy-initiated amount per bar) | | |
| qo | qo.a-f | samplerS ZZQO | | |
| xc | xc.a-d | samplerS ZZXC (median amount / volume per bar) | | |
| yj | yj.c/d, yj2.a/c, yj3.c, yj4.a/c, yj5.a/b/c | | 溢价 ? (name only) | low |
| hi | hi.b/h, hi2.b/c/e/f/h/i/l | | | |
| ag | ag.a/c, ag2.a/c/e/f | | | |
| bt | bt.a/b, bt2.a/h | | | |
| op | op.a/i/j, op2.b | | open ? (name only) | low |
| sf | sf.c/d | | | |
| pp | pp.a/b/e, pp2.d/e | | | |
| pz | pz.a-d | | 盘整 ? (name only) | low |
| wj | wj.a-d | | 委价 ? (name only) | low |
| xt | xt, xt2/3/5/6 | | 形态 ? (name only) | low |
| sa | sa.a/b, sa2.a/b | | | |
| uc | uc.c/d/f | | | |
| lk | lk, lk2, lk3 | | | |
| ri | ri.f, ri2.a/f | | 日内 ? (name only) | low |
| el | el.a/b | | | |
| aa | aa.a/b | | | |
| av | av, av2 | | average ? (name only) | low |
| bo | bo.a/c | | | |
| bs | bs.b, bs2.b | | buy / sell ? (name only) | low |
| fq | fq.b/e | | | |
| ho | ho, ho2 | | | |
| or | or.a, or2.a | | | |
| dw | dw | | 档位 ? (name only) | low |
| hk | hk.d | | 回看 ? (name only) | low |
| ht | ht | | 回调 ? (name only) | low |
| hx | hx.d | | | |
| ja | ja | | | |
| ko | ko | | | |
| kp | kp | | 开盘 ? (name only) | low |
| lf | lf | | | |
| mo | mo | | momentum ? (name only) | low |
| oe | oe | | | |
| ow | ow2 | | | |
| rb | rb | | | |
| rg | rg | | | |
| so | so | | | |
| wf | wf | | | |
| zs | zs.a | | 涨速 ? (name only) | low |

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
