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

## 6. Cost model uses the delivery schedule throughout — now checked, doesn't change the answer

STT at 0.1% per side and 0.015% stamp duty on the buy are the **delivery** rates. A
daily-rebalanced strategy holding overnight is genuinely delivery, so this is correct
as specified — but it is the conservative end.

**Checked with `scripts/cost_sensitivity.py`**: under the intraday schedule (0.025% STT,
sell-side only; ~14 bps round-trip vs delivery's 32), cost drag falls roughly in half
across every strategy — MeanReversion's Sharpe improves from **−1.41 to −0.20**, the
largest move in the study. But every strategy, including that one, still fails to beat
Buy&Hold after Holm correction. The delivery schedule was conservative but not
load-bearing for the conclusion: it wasn't hiding an edge.

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

---

## 9. The live pipeline caught two more real bugs before its first real run

Building `scripts/run_live_signal.py` (technical, regime, sentiment, debate and a
cost-aware risk gate against real, current data) surfaced two defects the walk-forward
study's own test suite never could, because both only manifest for inputs the study
never produced:

- **`TechnicalAgent` crashed on a `None` attention value.** The walk-forward study's
  default `TrainConfig` always has `attention=True`, so `attn_recent5` was always a real
  float or NaN in every OOS CSV the study ever produced. The live pipeline's production
  model deliberately uses the *empirically best* architecture (plain LSTM, no attention
  module), so `attn_recent5` is `None` — a case the code's `x == x` NaN check silently
  never handled, since `None == None` is `True` in Python and the check then tried
  `float(None)`. Two occurrences (the rationale string and the evidence dict) both
  crashed identically. Fixed with a real `has_attention` check; regression-tested for
  both `None` and `NaN`.
- **The rationale hardcoded "PLSTM-TAL" regardless of which architecture actually ran.**
  A cosmetic-looking bug with a real-accuracy consequence: the live pipeline's plain-LSTM
  predictions were being narrated as coming from PLSTM-TAL, which is false. `TechnicalAgent`
  now takes an `architecture` label and the live script passes the checkpoint's real one.
- **`ckpt.save()` hardcoded the module-level `FEATURE_COLUMNS` constant** rather than
  recording whatever feature set the model was actually built with. Caught by a test
  training a small toy model with a different input width — a real defect (a checkpoint
  trained on a custom feature subset would have silently claimed the wrong input shape at
  load time), not a test-only edge case, and now enforced with a dimension check at save
  time.

**Not yet resolved:** the hosted Anthropic/OpenAI `classify_batch` implementations are
implemented against the documented structured-output contracts and unit-tested with
mocked responses, but have never been run against a live API (no key in this
environment) — a genuine end-to-end run could still surface something the mocks don't
capture, the same way the two bugs above only surfaced once real, non-walk-forward data
was pushed through the pipeline.

---

## 10. Persistent paper trading caught two more real bugs — one of them self-inflicted mid-fix

Building the SQLite-backed portfolio tracker, the mock broker, and the CLI/report
surfaces around them (`nse_agents/live/state_store.py`, `mock_broker.py`, `engine.py`,
`cli.py`, `scripts/generate_paper_report.py`) surfaced two more real defects:

- **An annualised Sharpe from a handful of tracked days is not a conservative estimate,
  it is a meaningless one.** The first version of `paper-status` computed one anyway —
  a 2-day sample produced **"+62.33"**, plus a numpy `RuntimeWarning: Degrees of freedom
  <= 0`. Fixed by withholding Sharpe (and the Buy&Hold comparison) until 60+ days are
  tracked, reporting cumulative return and max drawdown in the meantime instead — the
  same discipline `scripts/regime_analysis.py` already applies to the 82-day crash
  regime, applied here to a live account for the same reason.
- **Cumulative return and max drawdown, describing the same account, disagreed.** The
  headline "Total return" figure marks positions at *today's live price*; the first
  drawdown computation used only the *recorded* equity-history snapshots (written at
  each rebalance), so an account that fell in value since its last rebalance showed
  `-2.03%` cumulative return next to `0.00%` max drawdown — the live drop was invisible
  to the metric computing "worst point so far." Fixed by including today's live
  mark-to-market as the final point in the drawdown path, in both `cli.py` and
  `scripts/generate_paper_report.py`; regression-tested via the invariant that
  |cumulative return| can never exceed |max drawdown| from the same starting capital.

**A self-inflicted third issue, worth recording precisely because it was self-inflicted:**
the first attempt at fixing the drawdown computation introduced a genuine
`IndentationError` — a comment block and three assignment lines left outside the `if`
they were meant to be inside, in both files identically, since one was copy-pasted from
the other. `ast.parse()` on each file catches this class of error immediately and cheaply;
it is now a explicit step before treating any manual edit here as done, not just before
committing.

---

## 11. Ensemble scoring, health checks, notifications, and a dashboard — all built, none of the network-facing parts verified live

This round added a multi-provider ensemble sentiment scorer, a pre-market health check,
Telegram/webhook notifications, and a Streamlit dashboard. What's genuinely tested vs.
what remains unverified, stated plainly rather than left implicit:

- **`EnsembleBackend`** (`nse_agents/agents/ensemble.py`): the combination math
  (agreement via total-variation distance, per-batch fallback, the all-providers-fail
  error path) is fully tested against stub backends. **Never run with two real,
  simultaneous provider calls** — no Anthropic or OpenAI key in this environment, so
  whether real-world disagreement patterns look like the synthetic test cases is
  untested.
- **`healthcheck`**: every check runs a real operation (a real cached-price read, a
  real RSS HTTP request, a real `PRAGMA integrity_check`) except the LLM checks, which
  by default only check *configuration presence* — `--ping-llm` exists for a real call
  but has, for the same reason as above, never been exercised against a live key.
- **`TelegramNotifier` / `WebhookNotifier`**: both raise a clear, specific
  "not configured" error rather than silently no-op'ing, and the HTTP-call logic is
  tested against a mocked `urllib.request.urlopen` matching each API's documented
  response shape. **Neither has sent a single real message** — no bot token, no
  webhook URL. Markdown escaping in particular (Telegram's `parse_mode: Markdown` is
  fussy about unescaped `_`, `*`, `` ` ``, `[` in message text) is the most likely
  thing to break on a first real send and should be checked then, not assumed fixed now.
- **The Streamlit dashboard**: verified to actually start and serve (HTTP 200, zero
  tracebacks in the server log) against both an empty and a populated paper-trading
  account — this is a real runtime check, not just `ast.parse()`. Its business logic
  (`nse_agents/live/dashboard_data.py`) is unit-tested directly and reuses the exact
  same equity/drawdown computation as `cli.py` and `generate_paper_report.py`, so it
  cannot drift into a fourth, independently-wrong version of the KNOWN_ISSUES #10 bug.
  What is **not** tested is anything about the interactive session itself (widget
  state, tab-switching, the symbol selector) — Streamlit's own script-rerun model
  makes that a browser-driven test, not a pytest one.
- **Scheduling was deliberately not built.** "Daily at 9:00 AM / 3:30 PM IST" needs a
  cron or launchd entry (see README.md), not a background daemon silently started by
  this project. Building and installing an actual persistent scheduled job on the
  user's machine is a different, more consequential action than adding a Python module,
  and wasn't done without being asked to.

## 12. Risk attribution, hyperparameter sensitivity, and DB maintenance — one benchmark-construction bug caught before it produced a wrong number

This round added benchmark-relative risk attribution (Treynor, Information Ratio,
upside/downside capture), two extended benchmarks, a hyperparameter sensitivity sweep,
and SQLite backup/vacuum maintenance commands. Two real bugs were caught building it,
both by the project's own pre-existing testing discipline rather than by inspection:

- **A close-vs-open timing mismatch nearly shipped a wrong beta.** The first draft of
  the Nifty Next 50 / equal-weight-universe benchmark series (`nse_agents/backtest/benchmarks.py`)
  was built with a naive `pct_change()` on closing prices. Checked against a real
  Buy&Hold portfolio of this study's own NSE stocks — which should show a beta near
  1.0, since Buy&Hold holds most of the same names — the naive benchmark produced a
  correlation of **0.002** and a beta of **0.002**. The cause: every return series
  elsewhere in this codebase is built open-to-open (`nse_agents.data.prices.forward_return`,
  documented in `CLAUDE.md`'s non-obvious-rules list), and a close-to-close series is
  one trading session out of phase with it. Rebuilding the benchmark with
  `forward_return` fixed it — correlation 0.95, beta 1.02 for the same pair. Regression
  test: `tests/test_benchmarks.py::test_equal_weight_universe_uses_open_to_open_not_close_to_close`.
- **`information_ratio` silently returned "no edge" for an undefined case.** A
  portfolio that beats its benchmark by an identical amount every single day has
  ~zero tracking error *and* a nonzero active return — dividing one by the other is
  undefined, not zero. The first cut special-cased near-zero tracking error to always
  return `0.0`, which would report a genuine (if unrealistic) constant edge as "tracks
  the benchmark exactly." Fixed to return `0.0` only when the active-return mean is
  *also* near zero (real tracking), and `NaN` otherwise (an ill-defined ratio). Caught
  by this module's own test suite, not by review — see
  `tests/test_risk_attribution.py::test_information_ratio_is_nan_for_a_constant_riskless_edge`.
- **`db-vacuum`'s first cut reported a misleading number.** A single
  before/after file-size delta went *negative* on a real test database, because
  WAL-checkpointing (absorbing the write-ahead log into the main file) grows the file
  at the same time `VACUUM` shrinks it — two opposite effects, reported as one
  confusing number. Fixed to report three separate sizes (before checkpoint, after
  checkpoint, after vacuum) plus the two effects broken out
  (`checkpoint_grew_file_by_bytes`, `vacuum_reclaimed_bytes`).
- **The Nifty Next 50 ticker was verified, not assumed.** Yahoo Finance's `^NSMIDCP`
  symbol text reads "midcap," which would be the wrong index entirely if taken at face
  value — resolved by fetching the ticker's own quote metadata (`longName`/`shortName`),
  which confirms "NIFTY NEXT 50." Not inferred from price level or the symbol string.
- **Two new metrics were placed in `metrics.py`, not `stats.py`,** despite the request
  naming `stats.py`: that module's own docstring scopes it to statistical machinery
  (bootstrap CIs, hypothesis tests, multiple-testing correction), while Treynor/IR/beta/
  capture ratios are point-in-time, benchmark-relative performance figures — the same
  category as the `Performance` dataclass and `sharpe_ratio`/`max_drawdown` already in
  `metrics.py`. Sortino and Calmar ratios, also requested, already existed in
  `compute_performance` from an earlier round and were not rebuilt.
- **The hyperparameter sweep is a "+"-shaped design, not a full 4×3 factorial** —
  cost-threshold sensitivity is measured crossed at horizon=1 only, reusing the main
  study's own cached PLSTM-TAL out-of-sample predictions; horizon sensitivity is
  measured crossed at the reference 32bps cost only, reusing the existing
  `horizon_label_variants.csv` cache for h=1/5/20 and training only h=10 fresh (the one
  cell genuinely missing from the cache). A full crossed grid would have needed three
  more multi-seed walk-forward trainings for a search space already this exhaustively
  covered elsewhere in the study. Scope stated in `PREREGISTRATION.md` before any of
  these numbers were seen — see the README's hyperparameter-sensitivity section for
  the result.
