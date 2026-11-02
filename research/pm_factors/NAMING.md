# PM naming: what the letters seem to mean

Status per row: **confirmed** (an identity or an exact match), **likely** (fingerprint and a near match agree), **guess**.

## Column letters (samplerS V names)

| letter | meaning | status | evidence |
| --- | --- | --- | --- |
| `n` | notional, CNY amount | confirmed | ZZGW na/nb, ZZXC n, ZZMS n are amount-type (fingerprint ba≈1, bp≈0) |
| `q` | quantity, shares | confirmed | ZZGW qa/qb, ZZXC q, ZZMS q, ZZQO q are share-type (bp≈-1) |
| `a` / `b` | ask (sell) side / bid (buy) side | likely | ZZAL an≈ask resting amount, bn≈bid; ZZWA an/aq/bn/bq |
| `ab`, `cd`, `ef` | sum of the pair | confirmed | ZZGW qab = qa + qb, nab = na + nb exactly |
| `c`/`d`, `e`/`f` | second / third (ask, bid) pair, a higher threshold | likely | ZZUG a/b ≈ 0.58 x q90 trade size, c/d ≈ 1.16 x q95 |
| `m` / `s` | mean / std (sigma) | guess | ZZQI, ZZUGSS carry m and s next to the a..d pairs |
| `l` / `m` / `s` after `b` | large / medium / small | guess | ZZMS bl > bm > bs in level; classic money flow by order size |
| `c h l o` | close high low open | confirmed shape | ZZDS: price-type, rank 0.999 with our last/high/low/open |
| `da` / `db` | delta (imbalance) on the ask / bid side | likely | ZZAL da/db in [-1, 1] |
| `vr` | volatility / range | guess | ZZDS vr scales with price |
| `mb`, `ms`, `msr` | mean book, mean spread, mean spread ratio | guess | ZZDS: mb ≈ 5-level book volume, ms ≈ 0.006 scale-free |

## Family names (samplerS)

| family | reading | status |
| --- | --- | --- |
| ZZMS | Money flow by Size (n, q totals; bl/bm/bs size buckets) | guess, n/q confirmed |
| ZZAL | Ask/bid Levels (book amount per side, da/db imbalance) | likely |
| ZZGW | order flow notional / quantity per side | confirmed (identities) |
| ZZDS | Daily Summary (OHLC, book, spread, range) | likely |
| ZZQO | Quantity of Orders (new-order volume) | likely |
| ZZUG | large-order thresholds (size quantiles per side) | likely |
| ZZUGSS | summary stats of ZZUG (m, s) | guess |
| ZZWA | Weighted Ask/bid distance (an ≈ 0.003 = relative spread scale) | guess |
| ZZXC | eXeCution volume / amount per bar | likely |
| ZZTS | Trade Size / time-and-sales (q amount-type) | guess |
| ZZQI | Queue Imbalance (scale-free, cumulative) | guess |
| ZZVC | Volume Cluster / cancel (e capped at 10000) | guess |
| ZZHL | ? (n ≈ SSE order count) | open |
| ZZSQ | ? (daily, 3 V) | open |

## Engine keys (day-end state)

| key | reading | status |
| --- | --- | --- |
| `zzbd.pEmaS / pEmaT / rEmaS / rEmaT` | price / return EMA on the Snapshot / Trade clock | confirmed |
| `zzup.pf / ps` | price EMA fast (300 s) / slow (600 s) | confirmed |
| `zzpk.high / low / open` | peak values of the day | confirmed |
| `zzvx.tr*` | volatility: true range EMAs | confirmed (rule) |
| `zzqj.v / tpv / r` | volume, turnover (price x volume), running VWAP | confirmed |
| `zzgh.nb` | net buy share (buy - sell) / (buy + sell) | confirmed shape, 1% off |
| `zzmh.hm` | highest snapshot price | confirmed |
| `cdc.lp` | last price | confirmed |
