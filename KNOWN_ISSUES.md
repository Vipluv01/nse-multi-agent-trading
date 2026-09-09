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

## 4. The disagreement gate does not discriminate

`debate_mode="disagreement"` was meant to escalate only contested days, mirroring a desk
that does not send consensus calls to committee. **In practice it escalates 97.4% of
days (18,610 of 19,110)**, so it is behaving as `always` at ~2× the intended cost.

The cause is a scale mismatch, not a threshold that is merely too low: the technical
agent's stance is `2·P(up) − 1` with P(up) ≈ 0.50, so its stances cluster within ±0.05,
while the regime agent's votes span the full ±1. Any pair drawn from those two
distributions almost always spans more than the 0.30 threshold.

The fix is to compare agents on a common scale — z-scoring each agent's stance against
its own historical distribution before measuring spread — rather than to raise the
threshold, which would just move the arbitrary cut. Not yet implemented.

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
