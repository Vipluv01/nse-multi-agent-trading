# Pre-registration of improvement attempts

Written **before** any of the results below were computed, and committed before the runs.
Its purpose is to make the search cost visible: the Deflated Sharpe Ratio in the main
results already charges the study for the number of configurations tried, and every
attempt listed here is added to that count whether it succeeds or fails.

Reporting only the variants that worked, after trying this many, would manufacture
exactly the overfitting the project set out to measure.

## Attempts, each with a stated prior reason

| # | Attempt | Defect it addresses | Prediction |
|---|---|---|---|
| A1 | Cross-sectionally de-mean sentiment per day | The 1.5B model's measured 3:1 bullish prior is a near-constant offset; subtracting the day's cross-sectional mean removes a constant and leaves *relative* sentiment | Should remove the churn; may or may not reveal signal |
| A2 | De-mean sentiment against a trailing per-symbol baseline (causal, 60d) | Some symbols are systematically covered more positively than others | Similar to A1, additionally removes per-name bias |
| A3 | Require >= 3 headlines before the sentiment agent forms a view | Single-headline days are the noisiest and the agent currently acts on them at low confidence | Fewer, better-supported signals |
| B1 | 5-day forward horizon | At 32 bps round-trip, a 1-day signal must clear a very high bar; a 5-day horizon faces ~1/5th the cost drag | Best single-change candidate |
| B2 | 20-day forward horizon | Same argument, stronger | Lower turnover still |
| C1 | Cross-sectional ranking label (beat the day's median stock) rather than absolute up/down | KNOWN_ISSUES #1: absolute direction is dominated by market beta, which no per-name feature can predict; relative rank strips the common factor | The most promising untried variant |
| D1 | Z-score each agent's stance against its own trailing distribution before measuring disagreement | The gate fires on 97.4% of days because agent stances live on different scales | Should make the gate discriminate |

## Rules fixed in advance

1. **Every attempt is reported**, including failures, in `results/improvements/`.
2. **No attempt is evaluated on anything but the same purged walk-forward protocol**,
   the same cost model, and the same out-of-sample window as the main study.
3. **The threshold sweep is reported for any variant that is claimed to work**, so a
   result that survives only one decision threshold is visible as such.
4. **Success criterion, fixed now:** a variant "works" only if its net Sharpe's 95%
   block-bootstrap interval excludes zero *and* it beats Buy&Hold on a paired test after
   Holm correction. Beating the previous variant is not sufficient.
5. The trial count fed to the Deflated Sharpe is updated to include every attempt here.

---

## Addendum, registered after A1-A3 and before running E1/E2

A1-A3 (sentiment debiasing) failed against the pre-registered bar. B1/B2/C1 (horizon,
cross-sectional label) are still running as this is written. Two more attempts are
registered now, before their results are seen, for the same reason as the first set:
so success is not selected after the fact.

| # | Attempt | Reason | Prediction |
|---|---|---|---|
| E1 | Gradient-boosted trees (`HistGradientBoostingClassifier`) on the same causal features, in place of the LSTM | LSTMs are not the strongest tabular learner on financial panels with limited history (~2,600 rows/symbol); GBMs are the standard strong baseline in quant research and its absence from the study so far is a real methodology gap, not a new avenue for search | May match or modestly beat the LSTM; unlikely to manufacture an edge that was not in the features to begin with |
| E2 | Expand the price-only universe from 10 to 40 liquid NSE large/mid caps, for the cross-sectional label only | A cross-sectional "beat the median" label needs breadth to have statistical power -- with 10 names the median is a noisy statistic computed from 9 other draws. This does not touch sentiment (news is not fetched for the extra 30 names) so it changes nothing about the study's LLM findings | More power to detect a real cross-sectional effect if one exists; if none does, the extra names will not manufacture one |

**E1 and E2 are run together** (GBM trained on the 40-name cross-sectional panel) since
E2 without E1 would need ~5x the LSTM compute this machine does not have spare right now,
and the combination is the more powerful test of the "cross-sectional edge" hypothesis
than either alone.

Same success criterion as before: net Sharpe's 95% CI excludes zero AND beats Buy&Hold
after Holm correction. No new criterion is introduced after seeing a result.

---

## Final verdict, and the power analysis

None of A1-A3, B1/B2, C1, or E1/E2 cleared the pre-registered bar (net Sharpe CI excludes
zero AND beats Buy&Hold after Holm correction). Two attempts (E2, E2b) briefly appeared
to clear the first half of that bar before a periods_per_year annualisation bug in the
bootstrap CI was caught and fixed -- see `KNOWN_ISSUES.md` #7. After the fix, both CIs
cross zero like everything else.

**A power analysis was run after every attempt above failed**, to answer the question a
list of failures alone cannot: how large would a real edge have had to be for this
study's own test to find it? Using Monte Carlo simulation with the exact
`block_bootstrap_sharpe` test used to pass/fail every result in this document, against
synthetic data built from this study's own real Buy&Hold return series (so the
volatility, autocorrelation and fat tails are realistic, not assumed Gaussian):

| Horizon | Periods | Years | MDE @ 80% power | MDE @ 50% power |
|---|---|---|---|---|
| 1-day | 1,911 | 7.6 | **1.047** | 0.632 |
| 5-day | 382 | 7.6 | 1.087 | 0.684 |
| 20-day | 95 | 7.5 | 0.978 | 0.639 |

All three converge tightly on **MDE ≈ 1.0**, because total elapsed time (7.5-7.6 years
throughout), not how it is sliced into periods, is what determines this test's power.

**This changes the claim.** "No configuration beat Buy&Hold" is a weak statement --
it invites "did you look hard enough?" The precise version is: **this study, with the
data available to it, could not have reliably found a true annualised Sharpe below
roughly 1.0.** A Sharpe of 1.0 is already an excellent result by any standard (many
successful systematic strategies run at 0.3-0.7). The honest reading of nine failed
attempts plus this power analysis is not "there is no edge in NSE technical/sentiment
signals" -- it is "no edge above an unusually large threshold was found, and this study
was not powered to see anything smaller." Extending the data window or the universe
further is what would lower that threshold; nothing tried here manufactured a smaller
one improperly.
