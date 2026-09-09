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
- [How to reproduce](#how-to-reproduce)
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

The **Deflated Sharpe Ratio** makes it worse, correctly. Given the number of
configurations searched here, the expected maximum Sharpe from pure noise is **0.86** —
higher than any Sharpe actually observed. Under that null, no strategy in this study
clears the bar.

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
separately visible. All run the identical execution path and cost model.

| Configuration | Sharpe (net) | CAGR | MaxDD | Exposure | Turnover/yr |
|---|---|---|---|---|---|
| Buy&Hold *(reference)* | **+0.578** | 16.2% | −38.6% | 1.00 | 0.1× |
| Tech+Regime | +0.344 | 10.1% | −20.7% | 0.87 | 20.0× |
| Tech+Sent+Regime | +0.177 | 7.6% | −24.2% | 0.90 | 43.0× |
| Tech-only | −0.412 | 3.1% | −25.4% | 0.16 | 5.5× |
| Tech+Sentiment | **−2.090** | −14.6% | −70.8% | 0.46 | **123.9×** |

**Adding the sentiment agent makes every configuration worse.** `Tech+Sentiment`
is the worst strategy in the entire study: the bullish-prior signal flips constantly as
the day's headline mix changes, producing 123.9× annual turnover — roughly 20% of
capital per year in costs alone — for a signal with no predictive content. Adding the
regime agent on top (`Tech+Sent+Regime`) partly rescues it by damping the churn, but
still lands below `Tech+Regime` without sentiment.

The honest reading: **the only component that helps is the regime filter**, and even
`Tech+Regime` does not beat Buy&Hold.

---

## How to reproduce

```bash
cd nse_agents
uv venv --python 3.11 && uv pip install -e ".[dev]"

.venv/bin/python -m pytest tests/ -q          # 28 tests, ~1s

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
#    constrained-scoring calls); every call is cached, so re-runs are free.
.venv/bin/python scripts/run_agents.py --backend local --debate-mode disagreement

# 6. Figures.
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

In priority order, and honestly — the first item is worth more than the rest combined:

1. **Rerun the sentiment and debate arms on a frontier model.** One flag
   (`--backend anthropic`). The paper this replicates says the capability is emergent;
   a 1.5B null is not a test of that.
2. **Point-in-time fundamentals** would allow a real fundamental agent instead of a
   regime proxy. This is a data-acquisition problem, not a modelling one.
3. **A weekly or monthly horizon.** At 32 bps round-trip, a daily signal must clear a
   very high bar. The same pipeline at a 20-day horizon faces ~1/20th the cost drag,
   and that is where a real edge would most plausibly survive.
4. **Point-in-time index constituents**, to remove the survivorship in the 10-name
   universe.
