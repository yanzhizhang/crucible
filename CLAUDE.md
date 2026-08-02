# CLAUDE.md — working in crucible

Operating guide for this repo, plus the research workflow it encodes. Read the
first half to work here; read the second half if you are learning the
researcher side of the quant workflow.

---

# Part 1 — the repo

## What crucible is

The Python **research** layer of a Chinese A-share / futures system. It is a
_consumer_: factors are computed in C++ (`prism`) and dumped as Parquet. Python
evaluates, screens, models and sizes. It does not compute production factors.

Two frequencies are supported end to end: **daily** (from Parquet dumps) and
**3-second microstructure** (from raw exchange tick streams, with the limit
order book reconstructed by market-by-order replay).

## Current state

- 12 packages, **245 tests passing**
- Daily pipeline validated on 80 A-shares × 624 sessions, 36 factors
- Tick pipeline validated on real colo dumps: 14.2M-row flow panel, 736k-row
  order-book panel (98.3% two-sided), 31 microstructure factors
- Python 3.14, polars 1.43, pandas 3.0, LightGBM 4.7

## Layout

```
src/           the 12 packages (see README for the table)
research/      prism stand-in -- factor producers, NOT production
tests/         245 tests
docs/          EXTRAS.md (optional dependency growth path)
data/          generated; gitignored
```

## Non-negotiables

These are enforced by code, not convention. Do not "temporarily" relax them.

1. **Point-in-time.** Features read data at or before `t`; labels only after.
   `merge` + `shift` is banned — `shift` means "n rows down the frame", which
   equals "n periods ahead" only if there are no gaps. Suspensions guarantee
   gaps. Join on an explicit slot index instead.
2. **No factor reimplementation.** `forge/` holds _transforms_. Factors live in
   prism. `research/` is a stand-in producer for study only.
3. **Parity gate.** Python PnL must reconcile against the C++ backtester with
   fills held constant. A mismatch is a bug in weights/costs/accounting — never
   widen the tolerance.
4. **Determinism.** Fixed seeds; byte-identical outputs. No wall-clock in any
   computation. `stable_hash`, never `hash()` (salted per process).
5. **Cross-sectional grouping.** All symbols at `t` are ONE sample. Random
   K-fold is a broken validation scheme here, not merely a weak one.

## Conventions

- **polars-native** through IO/alignment/features/labels; convert to pandas
  once at the `kiln`/`ballast`/`herald` edge. Public functions accept either
  flavor and return what they were given.
- **Long format** `(ts, symbol, ...)` sorted canonically. Wide is a
  presentation detail produced at the last moment.
- **Every metric returns a dataclass carrying `n`**, never a bare float. An IC
  of 0.09 over 11 cross-sections and over 2000 are different claims.
- **Masked values become null, not dropped** — keeps the panel rectangular so
  cross-sectional ops still align.
- Errors are raised, never logged-and-continued. A research bug that returns
  plausible numbers is worse than a crash.

## Traps this codebase exists to prevent

Each was a real failure mode; each has a test.

| trap                      | what happens                               | guard                                                 |
| ------------------------- | ------------------------------------------ | ----------------------------------------------------- |
| stale suspension price    | flat 0% return reads as mean-reversion     | `almanac.build_masks`                                 |
| survivorship bias         | today's index members applied historically | `Universe` refuses membership without effective dates |
| `shift` across gaps       | feature paired with wrong future           | explicit slot-index join                              |
| venue scale crossed       | 10× price error, still plausible           | `scaling_for(exchange)` per record                    |
| `symbol=000001` → int `1` | joins match nothing, silently              | hive types pinned VARCHAR                             |
| single-stream seq gaps    | clean capture reads as 50% loss            | `merged_sequence_gaps`                                |
| exchange-time PIT         | look-ahead equal to wire latency           | gate on `arrival_ts`                                  |
| vendor float price        | truncation loses a tick                    | `from_vendor_float_price` rounds                      |
| overlapping labels        | t-stat inflated 2-3×                       | Newey-West lags = horizon − 1                         |
| leakage canary            | perfect backtest, silent                   | `assert_no_feature_leakage`                           |

## Commands

```bash
uv sync --extra dev --default-index https://pypi.tuna.tsinghua.edu.cn/simple
uv run pytest -q
uv run ruff check . && uv run ruff format .
uv run mypy
```

---

# Part 2 — the industry research workflow

What follows is how systematic equity research is organised at most quant
funds. It is a synthesis of published method (López de Prado, Harvey–Liu–Zhu,
Bailey–López de Prado) and common desk practice. Where something is genuinely
contested I say so rather than presenting one school as settled.

## The pipeline

```
hypothesis -> data -> feature -> label -> single-factor eval
    -> screening -> combination -> portfolio -> costs -> backtest
    -> paper -> live -> monitor -> decay/retire
```

crucible implements the middle: `quarry`→`almanac`→`forge`→`horizon`→`assay`
→`sieve`→`kiln`→`ballast`→`toll`→`ledger`→`herald`.

### 1. Hypothesis first

The strongest discipline in the whole process, and the one most often skipped.
State the economic mechanism _before_ looking: who is on the other side, why
they trade suboptimally, and why the effect persists. "Order-flow imbalance
predicts short-horizon returns because informed traders must cross the spread
and inventory-constrained market makers widen against them" is a hypothesis.
"Feature 47 has IC 0.03" is not.

Data mining without a prior is not forbidden — it is how many real factors were
found — but it raises the statistical bar enormously (see multiple testing).

### 2. Data integrity

Nearly all fake alpha is a data bug, not a modelling error. The recurring ones:
point-in-time violations, survivorship bias, look-ahead in
restatements/fundamentals, and index membership applied retroactively.

A vendor "as-of" snapshot is not the same as a PIT database. Fundamentals get
restated; using the restated value at the original date is look-ahead.

### 3. Feature construction

Cross-sectional standardisation within each timestamp; winsorize before
neutralize before standardize. Neutralize against what is already priced —
industry and size at minimum, often the full risk-model factor set.

The order matters: outliers dominate a regression, so winsorize first;
standardising before residualisation is undone by the residualisation.

### 4. Labels

Choose the horizon your holding period will actually be, not the one with the
best IC. VWAP-to-VWAP is the honest default; close-to-close assumes you fill the
whole position at one print.

Entry lag is not pedantry: the price stamped at `t` is already complete and
cannot be traded.

### 5. Single-factor evaluation

The standard battery — and what each is _for_:

| metric                | question it answers                            |
| --------------------- | ---------------------------------------------- |
| IC (rank)             | does it order names correctly?                 |
| ICIR                  | is the ordering stable over time?              |
| positive rate         | is it broad, or carried by a few periods?      |
| quantile monotonicity | is it a real exposure, or an outlier effect?   |
| decay curve           | what holding period can it support?            |
| turnover              | what will it cost to run?                      |
| coverage              | is the universe stable, or is this a data gap? |

**Decay against turnover is the decisive pair.** A factor that turns over faster
than its alpha decays pays costs for nothing. This single comparison kills more
candidate factors than any significance test.

### 6. The multiple-testing problem

The central statistical issue in modern quant research, and the one most
retail-facing material ignores.

- Harvey, Liu & Zhu (2016) surveyed the published factor zoo and argued the
  conventional |t| > 2 hurdle is far too weak given how many factors have been
  tried; they propose roughly **|t| > 3.0** for new candidates.
- Hou, Xue & Zhang (2020) replicated hundreds of published anomalies and found
  a majority failed under uniform methodology.
- Bailey & López de Prado's **Deflated Sharpe Ratio** adjusts for the number of
  trials and the non-normality of returns.
- **PBO** (probability of backtest overfitting) via combinatorially symmetric
  cross-validation estimates how likely your selected configuration is to
  underperform out of sample.

Practical consequence: _record how many things you tried_. A factor found on
the fiftieth attempt needs a far higher bar than the first. crucible's
`sieve` admission log exists partly for this.

### 7. Validation

Random K-fold is invalid on this data for two independent reasons:
cross-sectional dependence (all names at `t` are one observation) and label
overlap (a 10-day label at `t` observes returns through `t+10`).

Standard practice is **purged walk-forward with embargo** (López de Prado):
remove training samples whose label window overlaps the test set, then drop a
further gap after it for serial correlation. **Combinatorial purged CV** gives
more train/test paths from short samples, at the cost of non-independent folds.

A **noise benchmark** — rerun the identical pipeline on phase-randomized
surrogates that preserve the power spectrum — is the cheapest guard against
rediscovering autocorrelation you fed in yourself.

### 8. Combination

Once several factors survive, combine them. In practice:

- **Equal weight** on standardised factors is a strong, hard-to-beat baseline
- **IC/IR weighting** must use _lagged_ IC estimates, or it is look-ahead
- **Gradient boosting** (LightGBM) is the workhorse for tabular cross-sections
  and generally beats deep nets on flat features
- **Sequence models** (GRU/Transformer/GAT) start to pay when there is genuine
  sequence structure — intraday paths, order-book evolution — that a flat
  feature vector discards

The gain from a better model is usually smaller than the gain from a better
label or from removing a data bug.

### 9. Portfolio construction

Where much paper alpha dies. Constraints — name caps, industry caps, turnover
caps — each cost expected return and each is worth it, because an unconstrained
ranking portfolio concentrates in exactly the illiquid, high-idiosyncratic names
where the signal is least reliable.

Never hand a raw sample covariance to an optimizer on a wide cross-section: its
extreme eigenvalues are noise and the optimizer will load onto them
("error maximisation"). Use shrinkage (Ledoit-Wolf) or a factor model.

### 10. Costs and capacity

Costs are not one number. Explicit costs (stamp duty, commission, fees) scale
linearly with turnover; **market impact scales with the square root of
participation**. Only the second imposes a capacity limit, and only the
decomposition tells you which one is binding.

Capacity analysis — at what AUM does impact eat the alpha — is a first-class
research output, not an afterthought.

### 11. Attribution and monitoring

Live performance is decomposed against the same risk model used in
construction. Persistent unexplained residual means the risk model is missing
a factor you are unknowingly exposed to.

Alpha decays. Track IC over time; factors get crowded and die. Retiring a
factor is a normal outcome, not a failure.

## Researcher vs PM

The split varies by shop, but the common division:

|                | **Researcher**                                | **Portfolio Manager**                             |
| -------------- | --------------------------------------------- | ------------------------------------------------- |
| owns           | hypothesis, factor, label, model, evaluation  | capital allocation, risk budget, live book        |
| optimises      | information content per unit of research risk | risk-adjusted return of the whole book            |
| horizon        | weeks–months per idea                         | daily                                             |
| decides        | is this factor real?                          | do we allocate to it, and how much?               |
| typical output | factor spec + tearsheet + admission record    | position limits, hedge policy, retire/scale calls |

**The handoff artifact** is the thing to get right. A PM cannot act on "IC is
0.04". What transfers is: the economic thesis, the decay/turnover pair, the
capacity estimate, the cost decomposition, the exposure profile, and an honest
statement of how many variants were tried. crucible's tearsheets and admission
log are built to be exactly that artifact.

A researcher who reports only the good number is not being helpful — the PM's
job is sizing under uncertainty, and they need the uncertainty.

## Chinese A-share specifics

Meaningfully different from US equity research:

- **T+1 settlement** — a stock bought today cannot be sold today. Intraday
  round trips in cash equity are impossible; this is a hard constraint on any
  intraday equity strategy and pushes intraday work into futures and ETFs.
- **Price limits** — ±10% main board, ±20% STAR/ChiNext, ±5% ST. A limit-locked
  name is untradeable in one direction; one-word boards are untradeable
  entirely.
- **Stamp duty 0.05% sell-side** (halved from 0.10% in Aug 2023) — large
  relative to short-horizon alpha, and asymmetric.
- **High retail participation** — stronger and more persistent short-term
  reversal than developed markets at daily frequency. Note this **inverts at
  tick scale**: measured here, 9-second moves _continue_ over the next 30
  seconds.
- **Shorting is constrained** — securities lending is limited and expensive, so
  most "long-short" A-share books are long cash equity hedged with index
  futures (IF/IC/IM), which introduces basis risk and a small-vs-large tilt if
  the hedge index does not match the book.
- **Suspensions are common** and long, unlike US markets.

## Modern / SOTA additions

Where the field is moving, with an honest note on maturity:

- **Automated alpha mining** — genetic programming and RL search over formulaic
  alpha grammars (alphagen and successors). Real, but generates candidates at a
  rate that makes the multiple-testing problem acute. Screening discipline
  matters _more_, not less.
- **Open frameworks** — `qlib` (Microsoft) is the most complete open-source
  quant platform; its model zoo is genuinely useful. Its data layer and
  backtester overlap what a shop usually owns itself.
- **Agentic research loops** — RD-Agent and similar run hypothesis → implement
  → evaluate automatically. Early; the bottleneck moves to evaluation
  trustworthiness.
- **Deep learning** — established for sequence/microstructure data; contested
  for daily cross-sections, where boosting remains competitive.
- **Deflated metrics and PBO** — increasingly expected in institutional
  research review; still rare in retail-facing material.

## Reading list

- López de Prado, _Advances in Financial Machine Learning_ — purging, embargo,
  CPCV, PBO. The standard reference for the validation half.
- Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns" (2016) —
  multiple testing, the |t| > 3 argument.
- Hou, Xue & Zhang, "Replicating Anomalies" (2020) — the replication crisis.
- Bailey & López de Prado, "The Deflated Sharpe Ratio" (2014).
- Almgren & Chriss, "Optimal Execution of Portfolio Transactions" (2000).
- Grinold & Kahn, _Active Portfolio Management_ — the fundamental law
  (IR ≈ IC × √breadth), still the clearest framing of why breadth matters.

The fundamental law is worth internalising early: information ratio scales with
IC times the square root of breadth. A modest IC across 3,000 names beats a
strong IC across 30. It is also why the 80-name daily study here found nothing
tradable — insufficient breadth, regardless of factor quality.
