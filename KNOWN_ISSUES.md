# Known issues and open questions

Live findings that are not yet resolved, kept separate from the README so that the
write-up states conclusions and this file states doubts.

---

## 1. The technical model is at chance — is that the market, or the setup?

**Status: the central open question. Currently believed to be real.**

All four architectures score 49.96%–50.45% out-of-sample directional accuracy over
19,110 predictions. Before concluding "daily NSE direction is unpredictable from
technical features", these were ruled out:

- **Not a leakage-in-reverse bug.** The same pipeline reaches high in-sample accuracy;
  it is out-of-sample accuracy that collapses. If features were broken, both would fail.
- **Not under-training.** Early stopping fires on a chronological validation tail;
  `best_epoch` varies from 0 to 10 across folds, so folds that want more epochs get
  them.
- **Not a label bug.** `test_forward_return_is_open_to_open_and_shifted` pins the label
  to `open[t+2]/open[t+1] − 1` for every row, and the pooled up-rate is 50.6%, which is
  what an equity panel should look like.
- **Not one bad fold.** Per-fold accuracy ranges 0.47–0.55 with no systematic drift.

**Not yet ruled out:** that 15 standard technical features on a 30-day window are simply
the wrong hypothesis space, and that intraday microstructure or cross-sectional
(rather than per-name time-series) features would behave differently. A cross-sectional
ranking formulation is the most promising untried variant.

---

## 2. Sentiment scored by a 1.5B model shows a strong bullish prior — now quantified

**Measured over the full 34,992-headline corpus:** P(Good) = 0.458, P(Bad) = 0.154,
P(Unknown) = 0.388, mean signed score **+0.303**. That is a 3:1 bullish skew.

The consequence is visible in the ablation: `Tech+Sentiment` is the worst strategy in
the study (Sharpe −2.09) at **123.9× annual turnover**, because the signal flips with
the day's headline mix while carrying no predictive content (event study p = 0.986).

Partially mitigated — the `Unknown` label absorbs 39% of the mass and its share is
subtracted from the agent's confidence, and the noise filter drops SEO and
"prediction for tomorrow" filler. The residual skew is not mitigated.

**A frontier-model comparison is the correct next step** and is a one-flag change
(`--backend anthropic`). Until it is run, the honest claim is "a 1.5B model cannot do
this on NSE headlines", not "LLM sentiment does not work".

---

## 3. News corpus survivorship is real and unquantified

Headlines for 2016 are retrieved in 2026. What Google still indexes is plausibly
biased toward stories that turned out to matter. This cannot be fixed with the free
sources available, only measured — and it is not currently measured. A comparison
against a point-in-time archive for a single year would bound it.

---

## 4. The disagreement gate did not discriminate — diagnosed wrongly at first, now fixed

`debate_mode="disagreement"` was meant to escalate only contested days. It escalated
**97.4%** of them (18,610 of 19,110).

**The first diagnosis in this file was wrong.** It blamed a scale mismatch — the
technical agent's stances span ±0.05 while the regime agent's span ±1 — and proposed
z-scoring each agent against its own trailing distribution. That was implemented and
measured: it moved escalation only from 97.4% to **91.3%**.

Decomposing the gate showed why:

| Trigger | Share of all decisions |
|---|---|
| Stances straddle zero | **69.5%** |
| Wide spread only | 27.9% |
| Not escalated | 2.6% |

The dominant trigger is **sign disagreement**, which z-scoring cannot touch — it rescales
stances but does not move them across zero. And sign disagreement is exactly what near-
zero, roughly symmetric stances produce by chance: two such agents land on opposite sides
of zero about half the time, and with a mean of 2.59 active agents that reaches ~70%. The
gate was faithfully detecting noise disagreeing with noise.

**The fix that worked** is to gate on *conviction* rather than sign: escalate only when
one agent holds a confidence-weighted stance above `+floor` and another below `-floor`.

| Gate | Escalation rate |
|---|---|
| raw (original) | 97.4% |
| zscore (first attempt) | 91.3% |
| **conviction, floor 0.10** | **10.6%** |
| conviction, floor 0.20 | 2.2% |
| conviction, floor 0.30 | 0.7% |

At floor 0.10 the debate pass costs ~9x less and fires only where two agents genuinely
hold opposing views. Available as `disagreement_metric="conviction"`. **The headline
results were produced with the original `raw` gate**, so they are unaffected; the
conviction gate changes cost, not any reported number.

---

## 5. Debate stance is scored, not generated

Bull/bear conviction comes from constrained label logits over {Strong, Weak, None}
rather than free generation, for cost and measurability. Free-form transcripts exist
only for a sample. It is **not verified** that the scored conviction and the generated
argument agree — a model could log "Strong" while writing a hedged paragraph. Checking
agreement on the sampled subset would close this.

---

## 6. Cost model uses the delivery schedule throughout

STT at 0.1% per side and 0.015% stamp duty on the buy are the **delivery** rates. A
daily-rebalanced strategy holding overnight is genuinely delivery, so this is correct
as specified — but it is the conservative end. An intraday variant (0.025% STT,
sell-side only) would roughly halve the cost drag and is a one-line change in
`CostModel`. The conclusions are stated against the delivery schedule.

---

## 7. A real bug was caught mid-analysis: bootstrap CIs used the wrong annualisation for horizon > 1

While running the pre-registered improvement attempts (B1/B2/C1, E1/E2 -- see
`PREREGISTRATION.md`), `block_bootstrap_sharpe` and `paired_sharpe_difference` were found
to default to `periods_per_year=252` regardless of what the point estimate (via
`compute_performance`) had been annualised with. For a 20-day-horizon backtest this means
the CI was computed at daily annualisation while the headline Sharpe was computed
correctly at `252/20` -- two different scales combined into one printed line.

**This was caught, not shipped, because of an internal consistency check**: the CI's own
lower bound sat *above* its point estimate, which is not supposed to happen. That
shouldn't-happen observation is what triggered the investigation.

**Before the fix**, the buggy numbers looked like the first real positive result in the
whole study:

| Variant | Sharpe (own CI, buggy) | Verdict looked like |
|---|---|---|
| E2 GBM cross-sectional h=5, wide universe | 0.420, CI **[+0.02, +3.32]** | CI excludes zero |
| E2b GBM cross-sectional h=20, wide universe | 0.589, CI **[+0.74, +8.50]** | CI excludes zero |

**After the fix** (`periods_per_year` threaded through correctly):

| Variant | Sharpe (own CI, fixed) | Verdict |
|---|---|---|
| E2 GBM cross-sectional h=5, wide universe | 0.420, CI **[-0.25, +1.19]** | Crosses zero -- noise |
| E2b GBM cross-sectional h=20, wide universe | 0.589, CI **[-0.10, +1.52]** | Crosses zero -- noise |

The point estimates did not change (they were already correctly annualised via
`compute_performance`); only the interval did. What looked like "the first strategy in
the study with a statistically real positive Sharpe" was a units mismatch. Both variants
also still lose to Buy&Hold on the paired test (p=0.007 and p=0.143 respectively).

**Fixed in `nse_agents/backtest/stats.py`**, with a regression test
(`test_bootstrap_ci_uses_the_same_annualisation_as_its_own_point_estimate`) that asserts
the CI's point equals an independently-computed Sharpe at the same `periods_per_year`,
and that the wrong default measurably shifts it. See git history for the fix.

**Why this belongs in the write-up, not just the commit log:** it is the single clearest
demonstration in this project that internal consistency checks catch real errors before
they become false claims -- exactly the discipline the pre-registration exists to protect.

---

## 8. Regime-stability check: the null hides a real risk/return trade-off, not "no effect"

Added after the improvement campaign found no configuration that beats Buy&Hold on
average. Breaking the same walk-forward results down by a market-derived (not
strategy-derived) regime classifier shows the risk-managed configurations
(`Tech+Regime`, `Tech+Sent+Regime`, `Full+Debate`) drew down **−1.1% to −7.7%** during
the 82-day COVID crash while every configuration without the risk overlay, including
Buy&Hold, drew down **−21% to −24%**. The same configurations give up real upside in
calm/bull regimes, which is why the full-period average still favours Buy&Hold.

**Not yet resolved:** whether this trade-off is net-positive over a longer window with
more crash regimes than the one available here (82 days is not enough to say anything
statistically about the crash regime specifically -- see `scripts/regime_analysis.py`,
which deliberately reports cumulative return and drawdown rather than a bootstrapped
Sharpe for that regime, to avoid manufacturing false precision on ~80 data points).
