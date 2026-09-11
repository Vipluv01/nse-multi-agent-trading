# Changelog

## [0.1.0]

Every entry below is a real commit in this repository's history (`git log`), not a hand-written summary -- run `git log --oneline` yourself to cross-check.

### Multi-agent LLM trading framework for NSE equities (`7ff01116`)

Reimplements three papers on Indian equities under one evaluation protocol:
PLSTM-TAL (peephole LSTM + temporal attention), Lopez-Lira & Tang (LLM news
sentiment), and TradingAgents (bull/bear debate with a risk team).

The headline result is negative, and that is the contribution:

- No architecture beats chance out-of-sample (49.96-50.45% over 19,110
  walk-forward predictions; plain LSTM scores highest, best p = 0.21).
- LLM headline sentiment shows no return predictability at 1.5B (slope
  p = 0.986 over 15,200 symbol-days, tercile ordering inverted) - a direct
  replication of the paper's own small-model finding.
- Adding the sentiment agent is actively harmful: Tech+Sentiment is the worst
  configuration in the study (Sharpe -2.09 at 123.9x annual turnover).
- Nothing beats buy-and-hold, whose own Sharpe (0.58) is not distinguishable
  from zero over 7.7 years.

The machinery exists to make that trustworthy: purged/embargoed walk-forward
validation, a full Indian transaction-cost model (32 bps round-trip),
open-to-open execution, block-bootstrap confidence intervals, Holm correction
and a Deflated Sharpe Ratio.

53 tests, written as attacks on the pipeline rather than shape assertions -
perturb the future and assert the past does not move. They caught six real
bugs, including unfunded leverage at 117.8% gross exposure, after-close
headline leakage, and 16% of the news corpus being silently discarded.

### Complete multi-agent ablation, nine pre-registered improvement attempts, and a power analysis (`b6565541`)

Full debate run (18,610 escalated decisions, ~37k constrained-scoring calls) now
included in the agent ablation. Full+Debate (Sharpe +0.13) sits below Tech+Regime
(+0.34) -- adding the sentiment agent and the debate layer does not help. Tech+Sentiment
remains the study's one unambiguous result: Sharpe -2.09, 95% CI [-2.77,-1.40], the only
interval in the whole study that excludes zero on the losing side.

Pushed hard on "does anything actually work" before accepting the null:

- A1-A3: sentiment debiasing (cross-sectional/trailing de-mean, min-headline filter)
- B1/B2/C1: longer horizons (5d/20d) and a cross-sectional ranking label, removing
  the market-beta contamination in the absolute-direction label
- E1/E2: gradient-boosted trees in place of the LSTM, on a 4x wider (40-name)
  cross-sectional universe

All nine pre-registered in PREREGISTRATION.md before their results were seen. None
clear the bar (net Sharpe CI excludes zero AND beats Buy&Hold after Holm correction).

Caught and fixed a real bug in the process: block_bootstrap_sharpe and
paired_sharpe_difference never received the periods_per_year threading added
elsewhere, so every horizon>1 result had a CI computed on the wrong annualisation.
Before the fix, two GBM configurations briefly looked like the study's first real
positive Sharpe. After the fix, both CIs cross zero like everything else. Caught by
an internal consistency check (a CI's lower bound sitting above its own point
estimate), not by external review -- documented in full in KNOWN_ISSUES.md #7 with a
regression test.

Added a Monte Carlo power analysis (nse_agents/backtest/power.py) using the exact
block-bootstrap test that decided every verdict above, run against synthetic returns
built from this study's own real Buy&Hold data. Result: minimum detectable Sharpe at
80% power is ~1.0 across all three horizons tested, converging tightly because total
elapsed years -- not rebalancing frequency -- sets this study's statistical power.
This sharpens the whole study's conclusion from "no edge was found" to "no edge above
an unusually large threshold was found, and this design could not have reliably
detected anything smaller."

68 tests (was 53): dedicated coverage for the conviction-based debate gate fix
(escalation 97.4% -> 10.6%, after a first z-score attempt only reached 91.3% and the
real cause -- sign disagreement between near-zero, low-conviction stances -- was
diagnosed), the periods_per_year regression, and the power analysis mechanics
(monotonicity, null calibration, sample-size sensitivity).

### Add a regime-stability check: the risk overlay protects capital in a real crash (`f932995d`)

Every result so far averages over the full 7.6-year OOS window, which can hide a
strategy that only works in one market condition. Broke the same walk-forward
results down by a market-derived (not strategy-derived) regime classifier -- a
mechanical, causal function of the Nifty's own trailing-60-day return and distance
from its running peak, fixed and validated against known market history (the COVID
crash, the 2022 correction) before any strategy's returns were loaded.

Finding: during the 82-day COVID crash, every configuration carrying the risk
overlay (regime filter, vol-scaling, drawdown brake) drew down -1.1% to -7.7%.
Every configuration without it, including Buy&Hold, drew down -21% to -24%. The
same configurations give up real upside in calm/bull regimes, which is why the
full-period average still favours Buy&Hold -- so this doesn't reverse the null
result, it explains the mechanism behind it: a real risk/return trade-off that
didn't pay off net over this particular window (one crash, two long calm
stretches), not "no effect at all."

Reports cumulative return and max drawdown for the crash regime rather than a
bootstrapped Sharpe (82 days is not enough for that to mean anything), and every
regime x strategy combination is reported, not a favourable slice.

4 new tests, including a direct lookahead attack on the regime classifier
(perturb the last 30 days, assert no earlier label changes) -- 72 total.

### Add a live inference pipeline, cost-aware risk gating, macro features, and hosted-LLM scaffolding (`f24e65d3`)

Everything published so far is backtesting. This adds a genuinely new capability: the
full agent pipeline running end-to-end against real, current NSE prices and headlines,
producing the same structured, auditable decision the backtest produces -- proof the
system runs live, not only as a historical replay. Explicitly labelled throughout as a
demonstration, not a new evaluation: the production model has no held-out accuracy of
its own, since there is no future to hold out for a prediction about today.

New capabilities, each opt-in so none of them can silently change a number already
published in README.md on a plain re-run:

- Model checkpointing (nse_agents/models/checkpoint.py) -- nothing in the walk-forward
  study needed this before, since every fold's model was trained and discarded.
- Cost-aware risk gating (RiskLimits.cost_aware): refuses a position unless an estimated
  edge (score x confidence x the stock's own typical daily move, stated as a heuristic,
  not a calibrated forecast) clears the real round-trip transaction cost.
- India VIX and Nifty-momentum macro features in RegimeAgent (use_macro=True): a fourth
  vote and an additional confidence penalty. FII/DII institutional flow data was
  investigated and has no reliable free source (NSE's endpoint: 403 without more session
  engineering).
- Anthropic and OpenAI backends now implement classify_batch via native structured
  output (tool-calling / JSON schema): the schema forces reasoning before label and
  confidence in one call, so a stance's confidence cannot be generated independently of
  its stated reasoning. Unit-tested against the documented API contracts with mocked
  responses; unverified end-to-end (no API key in this environment).
- Intraday cost-sensitivity check (scripts/cost_sensitivity.py): resolves the open item
  in KNOWN_ISSUES #6. Cost drag roughly halves under intraday rates and MeanReversion's
  Sharpe improves from -1.41 to -0.20, but every strategy still fails to beat Buy&Hold.

Building this against real (non-walk-forward) data surfaced three more real bugs the
study's own tests never could, because none of them can occur on data the walk-forward
study itself produces: TechnicalAgent crashed on a None attention value (the production
model deliberately uses the empirically-best architecture, plain LSTM, which has no
attention module -- every walk-forward CSV always had attention enabled); the rationale
hardcoded "PLSTM-TAL" regardless of which architecture actually ran; and ckpt.save()
hardcoded the feature-column constant rather than recording what the model was actually
built with, caught by a test training a differently-shaped toy model.

19 new tests (91 total): checkpoint round-trip and validation, the None-vs-NaN
regression, hosted-backend parsing/fallback against mocked API contracts, and the
cost-aware/macro-feature opt-in defaults.

### Add persistent paper-trading state, a mock broker, a second RSS news source, and a CLI (`b5029227`)

Extends the live pipeline (previous commit) with real persistence: a SQLite-backed
portfolio tracker (nse_agents/live/state_store.py) recording cash, positions
(weighted-average cost basis), realised/unrealised P&L, transaction costs, and a full
trade log across runs -- executed through a mock broker (nse_agents/live/mock_broker.py,
nse_agents/live/broker.py) that reuses the project's own CostModel rather than
re-encoding the STT/stamp-duty schedule a second time, plus an additive bid-ask-spread
slippage model on top. `scripts/run_live_signal.py --persist` now rebalances a real
account toward each day's target weights; `python -m nse_agents.cli paper-status` and
`scripts/generate_paper_report.py` read it back.

A second, independent headline source (LocalRSSAggregator in nse_agents/data/news.py)
pulls from Economic Times' RSS feeds via feedparser, filtered client-side by company
name. Moneycontrol's RSS was checked directly and returns HTTP 403 to a scripted client
even with full browser headers -- registered in FEEDS and marked broken rather than
silently dropped, the same pattern already used for FII/DII flow data and Zerodha Kite
Connect.

Two more real bugs, caught the way every other one in this project has been -- by
testing against a case the walk-forward study's own inputs never produce:

- An annualised Sharpe from a handful of tracked days is not a conservative estimate,
  it's a meaningless one. The first cut of `paper-status` computed one anyway -- a 2-day
  sample produced "+62.33" alongside a numpy degrees-of-freedom warning. Fixed by
  withholding Sharpe (and the Buy&Hold comparison) below 60 tracked days, reporting
  cumulative return and drawdown in the meantime, matching the same discipline
  scripts/regime_analysis.py already applies to the 82-day crash regime.
- Cumulative return and max drawdown, describing the same account, disagreed: the
  headline return figure marks positions at today's live price, but the first drawdown
  computation used only the recorded equity-history snapshots (written at each
  rebalance) -- an account that fell since its last rebalance showed -2.03% return next
  to 0.00% drawdown. Fixed by including today's live mark-to-market as the final point
  in the drawdown path; regression-tested against the invariant that |return| can never
  exceed |drawdown| from the same starting capital.

A third, self-inflicted issue is recorded in KNOWN_ISSUES.md #10 precisely because it
was self-inflicted: the first attempt at the drawdown fix left a comment block and three
statements outside the `if` they belonged in, in both files identically (one was
copy-pasted from the other) -- a plain `ast.parse()` catches this immediately, and is
now a checked step before treating a manual edit here as done.

33 new tests (124 total): state-store arithmetic (buy/sell cost-basis, P&L realisation,
atomicity under a refused sell), mock-broker slippage direction and CostModel reuse, a
5-day end-to-end paper-trading integration test, RSS filtering/dedup against a synthetic
feed response, and the drawdown-invariant regression.

### Add ensemble sentiment scoring, a health check, notifications, and a dashboard (`a1405c9f`)

Four additions, each honestly scoped against what this environment can actually
verify:

- Multi-provider ensemble scoring (nse_agents/agents/ensemble.py): combines
  classify_batch-capable backends with an inter-provider agreement metric (1 -
  total variation distance, mean over every provider pair) and per-batch fallback
  -- one provider's rate limit or outage narrows the ensemble to whoever
  answered rather than crashing the run. classify_batch-compatible, so it's a
  drop-in for the existing sentiment/debate code paths with zero other changes.
  Unverified against real multi-provider traffic (no Anthropic/OpenAI key here);
  tested at the same mocked-backend boundary as each individual hosted backend.

- A pre-market health check (python -m nse_agents.cli healthcheck): price feed
  freshness against the real trading calendar (a 4-day threshold, not a naive
  24h one that would false-fail every Monday), LLM key configuration (presence
  by default, --ping-llm opts into a real call), Economic Times RSS
  reachability, and the paper-trading SQLite file's integrity (PRAGMA
  integrity_check plus required-table presence). Exit code 0 iff everything
  passes; every check is a real operation against real data, not a stub.

- Daily notifications (nse_agents/live/notifier.py): Telegram Bot API and
  generic Slack/Discord webhooks via urllib (matching the project's existing
  HTTP convention rather than adding a requests dependency), with
  send_with_fallback trying each configured channel in turn. Both notifiers
  raise a clear "not configured" error rather than silently no-op'ing, and
  neither has sent a real message in this environment. Content-building
  (build_premarket_briefing, build_eod_summary) is pure and fully tested;
  scripts/send_premarket_briefing.py and scripts/send_eod_summary.py are the
  cron/launchd entry points this module deliberately does not schedule itself.

- An interactive Streamlit dashboard (scripts/dashboard.py, business logic
  factored into nse_agents/live/dashboard_data.py with no streamlit import so
  it's unit-testable): Overview (equity curve, cumulative return, drawdown,
  Sharpe withheld below the same 60-day floor as cli.py/generate_paper_report.py),
  Positions & Trades, and Sentiment & Macro tabs. Verified to actually start
  and serve without error (HTTP 200, zero tracebacks) against both an empty
  and a populated paper-trading account -- a real runtime check, not just
  ast.parse().

54 new tests (178 total): ensemble combination/agreement/fallback against
stub backends, healthcheck against real price data, a real RSS request, a
real corrupted SQLite file, notifier send logic against mocked API responses
per each platform's documented contract, dashboard_data against a real
MockBroker-backed account, and the two cron scripts via subprocess with
--dry-run.

KNOWN_ISSUES.md #11 states plainly what's tested vs. what's genuinely
unverified: the combination/health-check/content-building logic is real and
tested; the live multi-provider traffic and the actual Telegram/webhook sends
are not, for lack of credentials in this environment, and that gap is
recorded rather than implied away.

### Add risk attribution, extended benchmarks, hyperparameter sensitivity, and DB maintenance (`7e8c731b`)

Benchmark-relative risk attribution (beta, Treynor, Information Ratio, upside/downside
capture) against Nifty 50, an equal-weight universe index, and Nifty Next 50, plus a
pre-registered hyperparameter sensitivity sweep (horizon x cost threshold x conviction
floor) confirming the study's null result isn't an artefact of any single hand-set
parameter -- no cell in 16 configurations clears its own Sharpe CI above zero.

Two real bugs caught building this: a naive close-to-close benchmark construction gave
a beta of 0.002 against a real Buy&Hold portfolio that should be ~1.0 (fixed by using
the codebase's open-to-open return convention throughout); and information_ratio
silently returned 0.0 for an undefined case (near-zero tracking error with a nonzero
active return) instead of NaN. Also adds db-backup/db-vacuum CLI subcommands for the
paper-trading SQLite store (now WAL-mode), with vacuum() reporting three separate sizes
rather than one before/after delta that a checkpoint-growth/VACUUM-shrinkage conflation
made misleading.

### Add synthetic stress tests, an institutional factsheet, and a price-data audit (`6d821b2e`)

Synthetic market-stress tester (scripts/stress_test_scenarios.py) lays four hand-built
price/volatility shocks on top of the study's own real recent price history and runs
the real RegimeAgent -> Trader -> RiskManager pipeline against each, isolated from the
technical agent's own near-chance signal. Finding: the regime overlay fully de-risks
ahead of a sustained decline but not a single flash-crash day or a pure volatility
spike -- it behaves like a slow-turning trend signal, not a shock absorber.

Institutional factsheet generator (scripts/generate_factsheet.py -> results/FACTSHEET.md)
reports the flagship Full+Debate configuration's key metrics, a monthly return heatmap,
a multi-benchmark comparison, and LLM usage stats -- explicitly stated as the same
result already reported in README.md, not a more favourable one.

Price-data anomaly audit (nse_agents/data/audit.py, `cli.py audit-data`) scans cached
OHLCV series for possible unadjusted corporate actions, volume/trading-day gaps, stale
repeated prices, and outlier return spikes, and found real, previously undocumented
issues in this study's own cached data (77 zero-volume rows clustered on 5 shared
holiday dates; 2 large single-name moves, one of which is real news rather than a data
artefact -- flagged as "possible," not asserted, since the tool cannot tell the two
apart from price data alone).

Two real bugs caught building this: the stress tester's first draft compared every
scenario against one fixed "today" baseline, producing a false de-risking verdict from
a pure rolling-window calendar-shift artefact (fixed with a matched, same-length,
zero-shock control on the same synthetic date); and the audit tool's outlier z-score
divided by a near-zero rolling standard deviation on a smooth synthetic fixture, the
same failure mode already fixed once for information_ratio.

### Add an instant circuit breaker, DB schema migrations, and a LaTeX paper generator (`b175aa2c`)

Circuit breaker (nse_agents/agents/circuit_breaker.py): two deterministic, instant
de-risking triggers -- an overnight gap-down (>=3%) and an ATR expansion (>2.5x the
causal 14-day average true range) -- wired into RiskManager.size() as an unconditional
override that cannot be outvoted by a still-bullish combined opinion. This complements
rather than fixes RegimeAgent: last round's stress test found the regime overlay is a
slow-turning trend signal that does not react to a single flash-crash day (by design,
since averaging four medium-term votes is what makes it robust to single-day noise);
the circuit breaker is the instant-reaction mechanism that has to live outside that
average. Fed the exact -10% return the Flash Crash stress scenario defines, it fires
immediately where the regime-only pipeline did not.

SQLite schema migrations (nse_agents/live/migrations.py, `cli.py db-migrate`): schema
changes tracked via PRAGMA user_version, each migration a single self-contained DDL
statement applied atomically and rolled back whole on failure, run automatically on
every PaperTradingStore connect without touching existing trade logs or account state.
First migration adds a risk_events table for circuit-breaker/regime de-risking triggers
that produced no trade to log otherwise.

LaTeX paper generator (scripts/generate_paper.py -> results/paper.tex): an IEEE-
conference-shaped writeup assembled entirely from already-cached result tables
(architecture ablation, gross-vs-net Sharpe, multi-agent ablation with Deflated Sharpe
Ratio, hyperparameter sensitivity) -- no new computation or number. Its five references
match README.md's citation table at the same level of detail, with no invented DOIs or
page numbers. Its actual LaTeX compilation is unverified, since this environment has no
LaTeX toolchain -- only structural sanity (balanced braces/environments, table cell
counts) is tested.

### Add a two-factor regression, model cards, and a structured metrics exporter (`7d926675`)

Factor decomposition (nse_agents/backtest/factor_model.py, scripts/factor_regression_report.py):
market and momentum factors, regressed against every strategy in the ablation. This is a
two-factor model, not the four-factor Fama-French model the name usually implies -- SMB and
HML need point-in-time market-cap/book-value history this project has never fetched and will
not fake, the same reasoning RegimeAgent's own docstring gives for having no fundamental
agent. Real finding: only 4/10 strategies show significant (p<0.05) two-factor alpha, and
Buy&Hold is one of them -- an equal-weight-vs-cap-weight artefact, not stock-picking skill.
Full+Debate's own alpha (+3.0%, t=0.92) is not significant: the multi-agent system adds no
detectable return beyond market beta and this universe's own momentum tilt.

Model cards (scripts/generate_model_cards.py -> results/MODEL_CARDS.md): one Mitchell et al.
(2019)-style card per core agent (TechnicalAgent, SentimentAgent, RegimeAgent, the debate
engine), every number pulled from an already-cached result file. Caught before shipping: the
first draft's Debate Engine card reported the flagship's real 97.4% escalation rate (the
original "raw" gate) next to text describing a lower-escalation fix, worded in a way that
read as if the fix had been adopted -- when in fact every published number in this study was
produced with the raw gate on purpose. Fixed, with a regression test asserting the card states
the gate was deliberately never replaced.

Structured metrics exporter (`cli.py export-metrics`): exports one strategy's gross/net
performance, per-symbol decision/signal log, equity curve, and factor-regression metrics as
JSON or CSV. Its own _json_default exists because the stdlib json module cannot serialise
numpy.float64/int64/bool_ without one -- caught before it silently stringified real numbers
into JSON strings.

### Verify LaTeX PDF compilation, add dashboard AppTest coverage, .env support, and GitHub docs (`ce7ba010`)

LaTeX (KNOWN_ISSUES.md #14): results/paper.tex now genuinely compiles, not just structurally
verified. Installed TinyTeX (~/Library/TinyTeX, no sudo) plus `tlmgr install ieeetran`,
confirmed a real 2-page PDF with zero fatal LaTeX errors via pdflatex run twice. paper.pdf is
committed; the build is reproducible (scripts/generate_paper.py + pdflatex). A new test
(tests/test_generate_paper.py) runs the real pdflatex binary end-to-end and skips cleanly on
any machine without a LaTeX toolchain -- it does not assume TinyTeX exists elsewhere.

Dashboard (scripts/dashboard.py, nse_agents/live/dashboard_data.py): added a 4th "Backtest
Ablation" tab showing the main study's own cached summary.csv and Sharpe forest plot.
tests/test_dashboard_render.py uses Streamlit's AppTest framework to run the real app and
assert zero exceptions across all four tabs, both empty and populated paper-trading states,
and the symbol-selector widget -- closing the "interactive session untested" gap KNOWN_ISSUES
#11 previously flagged. Caught and fixed a real deprecation warning in the process
(use_container_width -> width="stretch").

Notifications (nse_agents/live/notifier.py): added a minimal, dependency-free .env loader
(not python-dotenv, matching this module's own urllib-over-requests precedent), applied
automatically to every caller. Added a real Telegram getMe schema-verification test
(tests/test_notifier.py) that will run the moment real credentials exist -- it is currently
skipped, and KNOWN_ISSUES.md #11 says so explicitly rather than claiming a verification that
did not happen (this project has no way to create a Telegram bot on its own; that needs a
human's phone/account).

Also: results/GITHUB_PRESENTATION.md documents recommended repo description/topics and
records what was actually checked about README rendering on GitHub (all 9 figure links
resolve; no badges or LaTeX math blocks currently exist, the latter by design since GitHub
doesn't reliably render LaTeX math in READMEs). Applying repo metadata itself needs `gh`
CLI/auth this environment doesn't have -- manual steps included instead.

===END===
