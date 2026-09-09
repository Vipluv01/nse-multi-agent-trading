# A Multi-Agent LLM Framework for Explainable Algorithmic Trading on NSE-Listed Equities

An attention-augmented deep learning forecaster, an LLM news-sentiment analyst, and a
structured bull/bear debate, combined into one auditable trading decision — and then
evaluated honestly enough to say that **none of it beats buying and holding.**

That negative result is the contribution. The machinery below exists to make it
*trustworthy*: purged walk-forward validation, a full Indian transaction-cost model,
open-to-open execution, block-bootstrap confidence intervals, Holm correction, and a
Deflated Sharpe Ratio that charges the result for every configuration searched.

---

## Contents

- [The claim, and what actually happened](#the-claim-and-what-actually-happened)
- [Results](#results)
- [Does the null hold across market regimes?](#does-the-null-hold-across-market-regimes)
- [How to reproduce](#how-to-reproduce)
- [Live pipeline: the same agents, running on real data today](#live-pipeline-the-same-agents-running-on-real-data-today)
- [Architecture](#architecture)
- [How correctness is verified](#how-correctness-is-verified)
- [Threats to validity](#threats-to-validity)
- [Relation to the literature](#relation-to-the-literature)
- [What I would do next](#what-i-would-do-next)

---

## The claim, and what actually happened

The literature this project builds on reports, in sequence:

1. **PLSTM-TAL** (Heliyon 2024) — a peephole LSTM with temporal attention beats CNN,
   LSTM, SVM and Random Forest at index direction prediction.
2. **Lopez-Lira & Tang** (2023, rev. 2025) — GPT-4 scoring of news headlines predicts
   subsequent returns; small models (GPT-1/2, BERT) cannot do it.
3. **TradingAgents** (arXiv 2412.20138) — LLM agents in specialist roles, with bull/bear
   researchers debating and a risk team overseeing, beat Buy-and-Hold, MACD and KDJ+RSI
   on cumulative return and Sharpe.

This project reimplements all three ideas on **NSE-listed Indian equities**, which the
agentic-trading literature has barely touched, and evaluates them under one shared,
deliberately unforgiving protocol.

**What happened:**

| Claim under test | Result here |
|---|---|
| Peephole cells + temporal attention improve direction prediction | **Not supported.** All four architectures land at 49.96%–50.45% out-of-sample accuracy. The plain LSTM scores highest. No architecture differs from chance (best p = 0.21). |
| Attention-augmented DL beats classical baselines on P&L | **Not supported.** Every learned model loses to Buy&Hold, most of them significantly after Holm correction. |
| A multi-agent system beats Buy-and-Hold | **Not supported** on this data. The best configuration reaches Sharpe 0.34 against Buy&Hold's 0.58. |
| LLM headline sentiment predicts next-day returns | **Not supported at 1.5B.** Event study over 15,200 symbol-days: slope p = 0.986, and the tercile ordering is inverted. A direct replication of the paper's own *small-model* result. |
| Adding agents to a system improves it | **False here.** Adding the sentiment agent is actively harmful — `Tech+Sentiment` is the worst strategy in the study (Sharpe −2.09, 123.9× turnover). Only the regime filter helps, and not enough. |
| Costs are a detail | **Emphatically false.** The mean-reversion baseline has a *gross* Sharpe of **+0.68** and a *net* Sharpe of **−1.41**. Its 223×/year turnover costs 36% of capital annually. |
| Trying harder (more model classes, horizons, universe breadth) would find an edge | **Tested directly — no.** Nine further pre-registered attempts (GBM, wider universe, cross-sectional labels, longer horizons, debiasing) all failed; one briefly looked real before a units bug in the CI was caught and fixed. |
| The null result means there is no edge at all | **Not quite — now quantified.** A power analysis shows this study could only reliably detect a true Sharpe above **~1.0**. The honest claim is "no edge that large was found," not "no edge exists." |
| The risk overlay does nothing since it doesn't beat Buy&Hold | **Wrong mechanism, not no mechanism.** During the 82-day COVID crash it captured a −1.1% drawdown against Buy&Hold's −23.6%. It trades upside for downside protection, and over this window that trade lost on net — a different finding from "no effect." |

The last row is the single most transferable finding. On daily NSE signals, the
round-trip cost floor is **32 bps** (0.1% STT each way, 0.015% stamp duty on the buy,
exchange and SEBI charges, 18% GST on those, and 5 bps of slippage). A strategy needs
to be right by more than that before it is right at all.

---

## Results

Out-of-sample window: **2018-12-07 → 2026-08-27** (1,911 trading days, 19,110
symbol-days), across 10 liquid NSE large/mid caps. Every strategy — learned, classical,
and agentic — is run through the identical execution path, cost model and date range.

### Directional accuracy: the architecture ablation

![Architecture accuracy](results/figures/architecture_accuracy.png)

| Architecture | Accuracy (3 seeds) | AUC | MCC | p vs chance |
|---|---|---|---|---|
| LSTM (plain) | 0.5045 ± 0.0019 | 0.5049 | 0.0076 | 0.21 |
| LSTM-TAL (attention only) | 0.5024 ± 0.0005 | 0.5017 | 0.0056 | 0.51 |
| PLSTM-TAL (the paper's model) | 0.4998 ± 0.0048 | 0.5008 | 0.0007 | 0.96 |
| PLSTM (peephole only) | 0.4996 ± 0.0003 | 0.4999 | −0.0018 | 0.90 |

The between-architecture spread (0.005) is the same size as the between-seed spread,
which is the finding: at this effect size, a single-seed architecture comparison
measures initialisation luck. **This is why the ablation runs 3 seeds and reports the
spread** — the papers being replicated generally do not.

### Cost-adjusted performance

![Gross vs net Sharpe](results/figures/gross_vs_net_sharpe.png)

The gap between the two dots is what Indian transaction costs remove. Note
`MeanReversion`: a genuinely positive gross signal that is deeply unprofitable to trade.

### Is any of it distinguishable from zero?

![Sharpe forest plot](results/figures/sharpe_forest.png)

**No.** Buy&Hold's 0.58 Sharpe has a 95% block-bootstrap interval of [−0.15, +1.34] —
7.7 years is not enough data to establish even a passive equity premium at conventional
significance. Reporting a point estimate without this interval is how backtests get
oversold.

The **Deflated Sharpe Ratio** makes it worse, correctly. Given the 10 configurations
searched in the full agent ablation, the expected maximum Sharpe from pure noise alone is
**1.35** — higher than Buy&Hold's own 0.58, let alone anything the learned strategies
produced. Under that null, no strategy in this study clears the bar. (This number rises
further once the nine additional improvement attempts below are counted; see
[`PREREGISTRATION.md`](PREREGISTRATION.md).)

### Equity curves

![Equity curves](results/figures/equity_curves.png)

`Tech-only` is flat 84% of the time (it trades only on high-confidence days) and
therefore mostly earns the risk-free rate — which the backtest credits, so that being
selective is not artificially punished.

### Threshold sensitivity

![Threshold sensitivity](results/figures/threshold_sensitivity.png)

A result that only appears at one hand-picked decision threshold is an artefact. This
one does not appear at *any* threshold, which is at least consistent.

### Does LLM headline sentiment predict returns?

Tested as a **standalone event study before any strategy is built on it**, because a
sentiment signal with no return predictability cannot acquire it by being placed inside
a multi-agent system.

Corpus: 34,992 headlines (after noise filtering) scored by Qwen2.5-1.5B-Instruct,
aggregated to **15,200 (symbol, trading-day) views**.

![Sentiment event study](results/figures/sentiment_buckets_local.png)

| Test | Result |
|---|---|
| OLS slope, next-day return on sentiment | **+0.05 bps** per unit sentiment (se 3.04), t = 0.018, **p = 0.986**, R² = 0.0000 |
| Positive-minus-negative tercile spread | **−1.01 bps/day**, t = −0.28, **p = 0.782** — the *wrong sign* |
| Mean next-day return, all views | +5.42 bps |

**No predictability, and the tercile ordering is inverted.** This is a direct
replication of Lopez-Lira & Tang's *negative* result — they report that GPT-1, GPT-2 and
BERT fail at this task and that predictability emerges only in more capable models.
Qwen2.5-1.5B behaves exactly like their small-model group.

The label distribution shows why, and is worth reporting on its own:

| P(Good) | P(Bad) | P(Unknown) | Mean signed score |
|---|---|---|---|
| 0.458 | 0.154 | 0.388 | **+0.303** |

A **3:1 bullish skew**. On a long-only book that is a systematic bias, not noise — the
model is largely reporting its own prior rather than reading the headline. The `Unknown`
escape absorbs 39% of the mass, which is the design working; the residual skew is not.

**This is the single result most likely to change with a better model**, and it is a
one-flag swap (`--backend anthropic`).

---

### The multi-agent ablation

Each configuration adds exactly one component, so the contribution of each is
separately visible. All run the identical execution path and cost model, and the
`Full+Debate` row includes the bull/bear researcher debate over ~18,610 escalated
decisions (≈37,000 constrained-scoring calls) on top of every specialist.

![Sharpe forest plot, full ablation](results/figures/sharpe_forest.png)

| Configuration | Sharpe (net) | CAGR | MaxDD | Exposure | Turnover/yr | vs Buy&Hold (Holm p) |
|---|---|---|---|---|---|---|
| Buy&Hold *(reference)* | **+0.578** | 16.2% | −38.6% | 1.00 | 0.1× | — |
| RSI(14) | +0.424 | 12.3% | −39.8% | 0.60 | 4.3× | 0.575 |
| Tech+Regime | +0.344 | 10.1% | −20.7% | 0.87 | 20.0× | 0.575 |
| Tech+Sent+Regime | +0.177 | 7.6% | −24.2% | 0.90 | 43.0× | 0.205 |
| **Full+Debate** | +0.131 | 6.9% | −25.9% | 0.91 | 41.2× | 0.066 |
| MACD | +0.020 | 5.1% | −29.4% | 0.78 | 42.8× | 0.205 |
| KDJ+RSI | −0.344 | −0.1% | −39.4% | 0.34 | 50.9× | **0.006** |
| Tech-only | −0.412 | 3.1% | −25.4% | 0.16 | 5.5× | **0.046** |
| MeanReversion | −1.408 | −18.1% | −79.6% | 0.76 | 223.5× | **<0.001** |
| **Tech+Sentiment** | **−2.090** | −14.6% | **−70.8%** | 0.46 | **123.9×** | **<0.001** |

**Adding the sentiment agent makes every configuration worse, and it is the only
change in the whole ablation with a statistically unambiguous effect.**
`Tech+Sentiment`'s own Sharpe interval is **[−2.77, −1.40]** — the single result in this
entire study whose 95% CI does not touch zero in the *losing* direction. The bullish-prior
signal flips constantly as the day's headline mix changes, producing 123.9× annual
turnover for a signal with no predictive content (see the sentiment event study above).

**Adding the debate layer does not help either.** `Full+Debate` (Sharpe +0.131, CAGR
6.9%) sits *below* `Tech+Sent+Regime` (+0.177), which sits below `Tech+Regime` alone
(+0.344, 10.1% CAGR). The incremental-contribution test (`Full+Debate` vs `Tech-only`)
gives +0.544 with a CI of [−0.32, +1.28] — directionally positive but nowhere near
significant (p_holm = 0.456). The honest reading across every configuration tried:
**the only component with a positive, if non-significant, contribution is the regime
filter**, and even `Tech+Regime` does not beat Buy&Hold.

---

## Does the null hold across market regimes?

Every result above averages over 7.6 years. That average can hide a strategy that only
works in one condition -- or, the more interesting possibility, a risk overlay that
looks unremarkable on average while doing real work specifically when markets fall.

The regime classifier is **market-derived, not strategy-derived**: a mechanical, causal
function of the Nifty's own price history (trailing-60-day return and distance from its
running peak), fixed and validated against known market history *before any strategy's
returns were loaded*. Defining "crash" by which days a strategy happened to lose money
would be circular; defining it from the benchmark's own drawdown is not.

| Regime | Trading days | What it is |
|---|---|---|
| Crash | 82 | Nifty down >10% over the trailing 60 sessions -- almost entirely the COVID crash, Mar-Jun 2020 |
| Choppy | 973 | Neither falling sharply nor near a high |
| Bull | 851 | Within 3% of the Nifty's running all-time high |

![Crash-regime drawdown](results/figures/crash_drawdown.png)

**During the 82-day COVID crash, `Tech+Regime` drew down −1.1% while Buy&Hold drew down
−23.6%.** Every configuration carrying the risk overlay (regime filter, volatility
scaling, the drawdown brake) clusters near zero; every configuration without it clusters
near Buy&Hold's loss, including the classical baselines. This is not a fluke of the
crash regime's small sample — it is the risk overlay's designed behaviour: the regime
agent's trend, relative-strength and range-position votes all turn sharply negative
together when a market is actually falling, and the book goes largely flat.

The trade-off shows up just as clearly in the calmer regimes, in the full table
([`results/improvements/regime_breakdown.csv`](results/improvements/regime_breakdown.csv)):
in the "bull" regime, `Tech+Regime` returns +47.1% cumulative against Buy&Hold's +63.8% —
real upside given up in exchange for the crash protection above. Averaged over the whole
window, the give-up in calm markets outweighs the protection in the crash, which is
exactly why `Tech+Regime`'s full-period Sharpe (+0.34) still trails Buy&Hold's (+0.58).
**The null result is not "this system does nothing" — it is "this system trades some
upside for downside protection, and over this particular 7.6-year window (one real
crash, two long calm stretches) that trade did not pay off on net."** Whether it would
pay off over a longer window with more crashes is precisely the kind of question the
power analysis below says this study cannot answer from 7.6 years of data.

---

## Trying to make it work: nine further attempts, pre-registered

The result above invites an obvious question: was that the ceiling, or just what this
particular setup happened to find? Nine additional, independently-motivated attempts
were run to answer it — **each registered in [`PREREGISTRATION.md`](PREREGISTRATION.md)
before its result was seen**, specifically so a favourable result could not be selected
after the fact. The same pass bar applied to all of them: the net Sharpe's 95% CI must
exclude zero **and** the strategy must beat Buy&Hold after Holm correction.

| # | Attempt | Best result | Verdict |
|---|---|---|---|
| A1–A3 | Sentiment debiasing (cross-sectional de-mean, trailing de-mean, ≥3-headline filter) | p = 0.25 (min-headlines filter flips the spread to the *correct* sign, but not significantly) | Fails |
| B1/B2 | Longer forward horizons (5-day, 20-day) | Cost drag falls 2.8%→1.1%/yr as predicted, but CI still crosses zero and still loses to Buy&Hold | Fails |
| C1 | Cross-sectional label (beat the day's median, removing market beta) | Same pattern at every horizon tested (1/5/20-day) | Fails |
| E1 | Gradient-boosted trees in place of the LSTM | Worse than the LSTM at every setting tried | Fails |
| E2/E2b | GBM + cross-sectional label + **4× wider universe** (40 liquid NSE names) | **See below — this is where a real bug was caught** | Fails |

**The GBM + wide-universe attempt is the most important line in this table, not
because it worked, but because of what happened while testing it.** Before a fix,
E2/E2b's own Sharpe intervals appeared to exclude zero for the first time in the whole
study — a real, if modest, positive result. Investigating *why* the interval's lower
bound sat above its own point estimate (which should never happen) surfaced a bug: the
bootstrap CI functions had never been updated to accept the same `periods_per_year` used
everywhere else, so a 20-day-horizon backtest's headline Sharpe and its "95% CI" were
silently computed on two different annualisation scales.

| | Sharpe | CI (buggy) | CI (fixed) |
|---|---|---|---|
| E2 (GBM, cross-sectional, h=5, 40 names) | +0.420 | [+0.02, +3.32] — *looked significant* | **[−0.25, +1.19] — noise** |
| E2b (GBM, cross-sectional, h=20, 40 names) | +0.589 | [+0.74, +8.50] — *looked significant* | **[−0.10, +1.52] — noise** |

Fixed in `nse_agents/backtest/stats.py`, with a regression test asserting a CI's point
estimate must match an independently-computed Sharpe at the same annualisation. Full
account in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md#7). This is the single clearest artefact
in this repository of the discipline the pre-registration exists to enforce: a result
that looked like a genuine finding was caught and reversed before it was reported, not
after.

**After the fix: none of the nine attempts clear the bar.**

---

## How large an edge would this study actually have found?

"No configuration beat Buy&Hold" is a weak claim on its own — it invites *did you even
look hard enough?* A **Monte Carlo power analysis** answers that precisely, using the
exact `block_bootstrap_sharpe` test that decided every pass/fail verdict above, run
against synthetic return series built from this study's own real Buy&Hold data (so the
volatility, autocorrelation and fat tails are realistic, not assumed Gaussian) with a
known, injected true Sharpe.

![Power curve](results/figures/power_curve.png)

| Horizon | Periods (years) | Minimum detectable Sharpe @ 80% power |
|---|---|---|
| 1-day | 1,911 (7.6y) | **1.047** |
| 5-day | 382 (7.6y) | 1.087 |
| 20-day | 95 (7.5y) | 0.978 |

All three converge tightly on **≈1.0**, because it is total elapsed time (7.5–7.6 years
throughout), not how it is sliced into rebalancing periods, that determines this study's
statistical power — a clean, testable prediction the simulation confirms rather than
assumes.

**This sharpens the claim.** A Sharpe of 1.0 is already an excellent result by any
industry standard; most real systematic strategies run at 0.3–0.7. The honest summary of
this entire study is therefore not *"there is no edge in NSE technical or sentiment
signals"* — it is: **no edge above an unusually large threshold (~1.0 Sharpe) was found
across ten architectures, three horizons, two label constructions, two model classes,
and a 4× wider universe, and this study was not powered to reliably detect anything
smaller.** Extending the data window or the universe further is what would lower that
threshold; nothing here manufactured a smaller one improperly.

---

## How to reproduce

```bash
cd nse_agents
uv venv --python 3.11 && uv pip install -e ".[dev]"

.venv/bin/python -m pytest tests/ -q          # 68 tests, ~2.5min (power-analysis tests are Monte Carlo)

# 1. Price data is fetched and cached on first use (Yahoo, split/bonus adjusted).
# 2. Headline corpus: 36,630 Indian headlines, 2016-2026. Resumable, and shardable
#    with --tag so several workers can run in parallel without racing (a single
#    worker takes ~4h; five take ~50min).
.venv/bin/python scripts/fetch_news.py --start 2016-01-01 --symbols RELIANCE,TCS --tag w1 &
.venv/bin/python scripts/fetch_news.py --start 2016-01-01 --symbols INFY,SBIN   --tag w2 &
wait && .venv/bin/python scripts/merge_news.py

# 3. Walk-forward training: 4 architectures x 3 seeds x 16 folds (~12 min, CPU).
.venv/bin/python scripts/train_technical.py --epochs 25 --seeds 3
.venv/bin/python scripts/evaluate_technical.py

# 4. LLM sentiment scoring (local Qwen2.5-1.5B on MPS; no API key needed, ~60 min
#    for 35k headlines). Aggregation is split out so the trading-day alignment and
#    the event study can be revised without re-scoring the corpus.
.venv/bin/python scripts/score_sentiment.py --backend local
.venv/bin/python scripts/aggregate_sentiment.py --tag local

# 5. The multi-agent ablation. The debate pass is the expensive part (~37k
#    constrained-scoring calls, ~2.5h on CPU); every call is cached, so re-runs are free.
.venv/bin/python scripts/run_agents.py --backend local --debate-mode disagreement

# 6. The improvement campaign (nine pre-registered attempts), the power
#    analysis, and the regime-stability check. See PREREGISTRATION.md before
#    reading the improvement-attempt results.
.venv/bin/python scripts/improve_sentiment.py     # A1-A3: sentiment debiasing
.venv/bin/python scripts/improve_horizon.py       # B1/B2/C1: horizons, cross-sectional label
.venv/bin/python scripts/improve_gbm.py           # E1/E2: gradient-boosted trees, wider universe
.venv/bin/python scripts/power_analysis.py        # ~13 min: Monte Carlo detection power
.venv/bin/python scripts/regime_analysis.py       # crash/choppy/bull breakdown, ~1 min

# 7. Figures.
.venv/bin/python scripts/make_figures.py
```

To run the LLM arms on a materially more capable model instead:

```bash
uv pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
.venv/bin/python scripts/score_sentiment.py --backend anthropic --tag anthropic
```

Everything is cached — prices to CSV, headlines to JSONL, LLM responses to SQLite — so
re-runs are deterministic and cost nothing.

---

## Live pipeline: the same agents, running on real data today

Everything above is backtesting. This section adds a genuinely different capability:
the full agent pipeline -- technical model, regime agent (with India VIX and Nifty
momentum), sentiment agent, bull/bear debate, and a cost-aware risk manager -- running
end to end against **today's real prices and today's real headlines**, producing the
same structured, auditable decision the backtest produces.

**This is a demonstration that the system runs live, not a new evaluation and not a
trading recommendation.** The production model has no held-out accuracy of its own --
there is no future to hold out for a prediction about today, since today's outcome
doesn't exist yet. Its expected performance is exactly what the walk-forward study
already measured for its architecture (plain LSTM, 50.45% OOS accuracy, indistinguishable
from chance). Every output from this pipeline carries that disclaimer.

```bash
.venv/bin/python scripts/train_production_model.py   # once, or whenever retraining
.venv/bin/python scripts/run_live_signal.py --backend local
```

Sample output, run against real NSE data on 2026-09-08:

```
symbol       action   score    size
----------------------------------------
TCS          BUY     +0.354   0.200
LT           BUY     +0.301   0.200
RELIANCE     BUY     +0.193   0.200
ICICIBANK    BUY     +0.129   0.200
...
```

Each decision carries the full rationale, exactly as in the backtest -- e.g. TCS's
actual output that day: *"[technical +0.01] LSTM assigns P(up)=0.505... [regime -0.41]
TCS is below its 200-day average... while the Nifty itself is down 2.8% over 20
sessions; India VIX sits in the 24% percentile. [sentiment +1.00] 17 headline(s) for
TCS; mean tone +0.554 (positive)... [debate] bull case Strong (0.71) vs bear case None
(0.11); net +0.60."*

**Two design decisions this pipeline enforces, both opt-in so they cannot silently
change any number already reported above:**

- **Cost-aware sizing** (`RiskLimits.cost_aware`, default off): refuses a position
  unless the estimated edge -- score x confidence x the stock's own typical daily
  move -- clears the real round-trip transaction cost. A heuristic, stated as one; this
  project's whole finding is that these scores carry ~0 real predictive edge, so a
  calibrated bps forecast from them would be invented precision.
- **Macro regime features** (`RegimeAgent(use_macro=True)`, default off): India VIX
  percentile and Nifty 20-day momentum, added as a confidence penalty and a fourth vote
  respectively. Free via Yahoo; FII/DII institutional flow data was investigated and has
  no reliable free source (NSE's endpoint returns 403 without more session engineering).

**Hosted-LLM backends beyond the local model** (`--backend anthropic` / `--backend openai`,
both use structured tool-calling / JSON-schema output so a stance's confidence cannot be
generated independently of its stated reasoning) are implemented and unit-tested against
the documented API contracts, but unverified end-to-end -- this environment has no API
key for either. They activate the moment one is supplied.

An **intraday cost-sensitivity check** (`scripts/cost_sensitivity.py`) reruns the key
strategies under the intraday cost schedule (~14 bps round-trip vs delivery's 32) as a
labelled sensitivity check: MeanReversion's catastrophic -1.41 Sharpe improves to -0.20,
but every strategy, including that one, still fails to beat Buy&Hold.

---

## Architecture

Dependency order, which is also the order to read the code in:

**1. `data/`** — `prices.py` pulls split/bonus-adjusted OHLCV straight from Yahoo's chart
endpoint (RELIANCE and INFY both had 1:1 bonuses in-sample; unadjusted series would
manufacture fake −50% days). `news.py` fetches Indian headlines in month-windowed
Google News queries and — critically — attributes each to the first trading day it
could be acted on. `features.py` builds 15 causal technical features. `dataset.py`
pools all symbols into date-sorted sequences.

**2. `models/`** — `attn_lstm.py` implements the peephole LSTM cell by hand (PyTorch
has none) plus Bahdanau temporal attention, with `peephole` and `attention` as
**runtime flags** so the ablation exercises one code path rather than comparing two
implementations. `train.py` does purged, embargoed, rolling walk-forward with per-fold
scaling and a chronological validation tail.

**3. `llm/`** — a backend protocol with three implementations (local Qwen, hosted
Claude, and a deterministic echo stub for tests), all behind a SQLite response cache.
The local backend's `classify_batch` reads label-token logits from one forward pass
instead of generating text: ~50× faster, exactly deterministic, and never unparseable —
which matters enormously at 1.5B.

**4. `agents/`** — each specialist returns the same `Opinion` (signed stance,
confidence, rationale), or **abstains**. Abstention is load-bearing: "no news today" and
"the news is genuinely mixed" imply different position sizes, and collapsing both to 0.0
is how a sentiment model ends up trading its own prior. `orchestrator.py` runs three
passes — gather, batch-debate, then a strictly sequential decision pass, because the
risk manager's drawdown brake depends on the equity curve realised so far and cannot be
vectorised without letting tomorrow size today.

**5. `backtest/`** — `costs` in `config.py`, `engine.py` for portfolio construction,
`baselines.py` for the classical comparators, `metrics.py`, and `stats.py` for the
inference.

### Two design choices worth defending

**The risk manager is deterministic, not an LLM.** Every other agent is a forecaster and
is allowed to be wrong. The risk layer is a constraint system, and a constraint that can
hallucinate is not a constraint. Position limits, volatility targeting and the drawdown
brake are rules an auditor can re-derive by hand.

**The trader is a fixed confidence-weighted mean, not a learned stacker.** With ~1,900
out-of-sample days, fitting a meta-model over four agents would fit the test period, and
its Sharpe would measure that overfitting. A fixed weighting can be wrong; it cannot be
tuned to the answer.

---

## How correctness is verified

**Correctness is established before performance is measured.** The 28 tests are mostly
not shape assertions — the important ones are active attacks on the pipeline:

| Test | The attack |
|---|---|
| `test_features_are_causal` | Multiply the final bar by 1.5×, 20× the volume. **Assert no earlier feature moves.** Catches centred windows, backfills, whole-series scaling. |
| `test_features_do_not_use_same_bar_close_for_future_bars` | Perturb bar 200; assert bars 0–199 are bit-identical. |
| `test_forward_return_is_open_to_open_and_shifted` | Assert `fwd_ret[t] == open[t+2]/open[t+1] − 1` for every t, and that the last row is NaN, not 0. |
| `test_news_after_close_moves_to_next_day` | A 10:00 GMT (15:30 IST) headline must be attributed to the *next* session. |
| `test_walk_forward_folds_are_ordered_and_embargoed` | Assert the embargo gap is really present and train/test indices are disjoint, on every fold. |
| `test_a_high_win_rate_can_still_be_a_bad_strategy` | 99 small wins, one −50% day: 99% hit rate, negative return. Asserts the metrics expose it. |
| `test_pure_noise_does_not_produce_a_significant_sharpe` | Zero-mean series must not be declared a winner. |
| `test_deflated_sharpe_penalises_a_wider_search` | The same track record must be worth less after more configurations were tried. |
| `test_gross_exposure_cap_prevents_unfunded_leverage` | Ten names at a 20% per-name cap is a 200% book unless the portfolio cap fires. |

### Real bugs these caught

- **Unfunded leverage.** `max_gross_exposure` was declared in `RiskLimits` but never
  applied: per-name sizing cannot see the rest of the book, so `Tech+Regime` ran at
  **117.8% gross exposure** with no funding cost, inflating both return and Sharpe. Fixed
  by a portfolio-level pass in the orchestrator (`apply_gross_cap`), which scales the
  day's whole book proportionally and records it in the audit trail.
- **Idle cash scored as zero.** The risk-state update credited flat capital with 0%
  rather than the risk-free rate, penalising selectivity and biasing every comparison
  toward always-invested baselines.
- **After-close headlines.** Google News timestamps are GMT. Attributing a 23:30 IST
  earnings-reaction headline to that afternoon's close would have leaked the market's
  own reaction into the signal that predicts it.

---

## Threats to validity

Stated plainly, because a study whose headline result is negative has to be at least as
careful about what could make it *wrongly* negative.

1. **News corpus survivorship.** The corpus is **36,630 unique headlines across
   2016-01-01 to 2026-09-10**, 913–2,594 trading days covered per symbol (ITC thinnest,
   SBIN richest); 34,992 survive the SEO/filler noise filter. But headlines for 2016 are
   fetched in 2026, so what remains is what is still indexed and linked — plausibly
   biased toward stories that turned out to matter. `coverage_report` publishes the true
   per-symbol span and count rather than describing the corpus in prose, and the
   per-symbol imbalance (ITC has a third of SBIN's coverage) means the sentiment agent
   abstains far more often on some names than others.
2. **No point-in-time fundamentals.** This is why there is a `RegimeAgent` and not a
   fundamental agent. Only *current* ratios are freely available for NSE names, and
   substituting today's P/E into a 2019 decision embeds every subsequent earnings
   surprise — the signal would appear to predict the future because it literally
   contains it. The agent is named for what it actually does.
3. **Long-only.** Delivery-segment shorting is unavailable to Indian retail at this
   horizon, which caps achievable Sharpe. A long/short version of this study would be a
   different (and untradeable) experiment.
4. **7.7 years is short.** As the forest plot shows, this window cannot resolve a Sharpe
   of 0.5 from zero. That cuts both ways: it is also why this study cannot *rule out* a
   small real edge.
5. **A 1.5B sentiment model is a weak test of the sentiment hypothesis.** Lopez-Lira &
   Tang's central claim is precisely that this capability is emergent. A null result
   from Qwen2.5-1.5B is evidence about small models, not about LLM sentiment in general
   — which is why the `--backend anthropic` path exists and is a one-flag swap.
6. **Debate stances are scored, not generated.** Conviction comes from constrained label
   logits so that ~19k decision points are affordable and measurable. Free-form
   transcripts are generated for a *sample* as the explainability artefact. The decision
   pipeline runs on every day; the narration does not.
7. **10 symbols, chosen for liquidity.** They are survivors of the 2016–2026 period. A
   fully survivorship-free study needs point-in-time index constituents.

---

## Relation to the literature

| Paper | What this project takes | Where it diverges |
|---|---|---|
| **PLSTM-TAL** (Heliyon 2024) | Peephole LSTM + temporal attention; the 7-metric evaluation vocabulary (accuracy, precision, recall, F1, MCC, AUC) | Evaluated with purged walk-forward across 16 folds and 3 seeds, on individual NSE equities rather than indices, and with P&L reported net of Indian costs. The architecture's advantage does not survive this. |
| **Lopez-Lira & Tang** (2023) | The headline-scoring prompt design: ask about the *share price*, not the news, and permit "unknown" | Applied to Indian headlines and NSE names; run as a standalone event study *before* any strategy is built on it |
| **TradingAgents** (2024) | Specialist roles, adversarial bull/bear framing, a risk team in the loop, the baseline set (Buy&Hold, MACD, KDJ+RSI, mean reversion) | The risk layer is deterministic rather than LLM-driven; the contribution of each agent is isolated by ablation rather than the system being reported as a whole |
| **An, Sun & Wang** (IEEE 2022) | Framing of non-stationarity and the sim-to-real gap | Addressed through rolling (not expanding) windows and an explicit cost model, rather than through RL |
| **Explainable DL for trend prediction** (Heliyon 2024) | The explainability requirement | Explanation is the decision's audit trail — which agent said what, and what the risk layer overrode — not a post-hoc attribution over a black box |

---

## What I would do next

Updated after the improvement campaign and power analysis. In priority order:

1. **Rerun the sentiment and debate arms on a frontier model.** One flag
   (`--backend anthropic`). The paper this replicates says the capability is emergent;
   a 1.5B null is not a test of that, and it is the single untried lever most likely to
   change a result rather than confirm the existing null.
2. **Extend the data window.** The power analysis is precise about why: this study's
   detection floor (~Sharpe 1.0) is set by *years of data*, not by anything about the
   models or horizons tried. More years — not a wider search over the same window — is
   the only lever that lowers it.
3. **Point-in-time fundamentals** would allow a real fundamental agent instead of a
   regime proxy. This is a data-acquisition problem, not a modelling one.
4. **Point-in-time index constituents**, to remove the survivorship in the (now 40-name)
   universe.
5. **A formal earnings/event-window study for the sentiment agent** — the one setting
   academic literature suggests news predictability is most likely to survive, and not
   yet isolated here (the event study above pools all headlines, not just
   earnings-adjacent ones).
