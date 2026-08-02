# Growth extras

The classical pipeline — load, align, label, evaluate, screen, model, size,
cost, report — needs none of these. They are optional dependencies that plug
into seams the core already exposes, so you can adopt them one at a time as you
learn each layer rather than inheriting a framework on day one.

```bash
uv sync --extra dev --extra opt      # add the convex optimizer
uv sync --extra dev --extra dl       # add torch rankers
```

## Why extras rather than a framework

`qlib`, `alphagen` and `RD-Agent` each ship their own data layer, expression
engine and backtester. Adopting any of them wholesale collides head-on with
three crucible invariants:

| their component                       | collides with                            | why it matters                                                              |
| ------------------------------------- | ---------------------------------------- | --------------------------------------------------------------------------- |
| qlib `.bin` data layer + `D.features` | `quarry`                                 | two sources of truth for what a price was at `t`                            |
| qlib / alphagen expression engines    | invariant 2 (no factor reimplementation) | a factor computed in Python is a live/research divergence waiting to happen |
| qlib / RD-Agent backtesters           | invariant 3 (C++ parity gate)            | a second fill model makes the parity gate untestable                        |

So crucible borrows the parts that are genuinely additive — model
architectures, search strategies, optimizers — and keeps ownership of data,
factors and PnL.

## `opt` — convex optimization in `ballast`

Adds `cvxpy`. Slots in behind the existing interface:

```python
score_to_weight(panel, "score", method="optimizer", constraints=Constraints(...))
```

Maximises `w'α − λ w'Σw` subject to the same `Constraints` the projection path
already enforces. Pair it with `ballast.ledoit_wolf` or
`ballast.factor_covariance` — never the raw sample covariance, whose extreme
eigenvalues are noise the optimizer will happily load onto.

## `dl` — torch rankers in `kiln`

Adds `torch`. Cross-sectional GRU / Transformer rankers that sit next to
`fit_lgbm` and reuse `PurgedWalkForward` unchanged, because the splitter
operates on timestamps and knows nothing about the model.

Worth knowing before you reach for it: on tabular cross-sectional equity data,
gradient boosting is a genuinely strong baseline and usually wins on a
single-factor panel. Neural rankers start to pay off when you have sequence
structure (intraday paths, order-book snapshots) that a flat feature vector
throws away — which is exactly the tick/3s regime this repo targets, so the
extra is here rather than dismissed.

Architectures worth borrowing from qlib's model zoo (`qlib.contrib.model`):
GATs, TFT, and the `ALSTM` variants. Take the module definitions; leave the
`Dataset`/`Recorder` scaffolding.

## `mining` — candidate alpha search in `sieve`

Adds `gplearn` for genetic-programming search over a formulaic alpha grammar,
in the spirit of alphagen's RL formulation.

**The output is a proposal, not a factor.** Candidates are scored by
`assay.ic` and gated by the existing `FactorPool.propose`, and anything that
survives gets **ported to prism in C++** before it can be traded. A mined
formula running live from Python would violate invariant 2 outright.

Treat mined candidates with more suspicion than hand-built ones: a search over
millions of expressions will find spectacular in-sample results by construction.
`assay.parameter_surface` and `kiln.noise_benchmark` are not optional here —
they are the only thing standing between you and a pool full of noise.

## `agentic` — RD-Agent research loop in `examples/`

Adds `rdagent` and needs LLM API credentials at runtime. Ships as a runnable
example driver rather than library code: it orchestrates
`forge` → `assay` → `sieve` in a hypothesis → implement → evaluate loop.

Every candidate it produces still passes through the same admission gate and
the same leakage assertions. An agent that can propose factors faster than you
can evaluate them makes the screening discipline more important, not less.

## Adoption order

1. **`opt`** — smallest, clearest win, low risk.
2. **`mining`** — teaches you how much the screening gates are doing.
3. **`dl`** — once you have intraday sequence features worth modelling.
4. **`agentic`** — last, once the evaluation harness is something you trust
   without reading its output line by line.
