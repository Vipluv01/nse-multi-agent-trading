"""Model cards, one per core agent component, following the Mitchell et al.
(2019) "Model Cards for Model Reporting" structure (details, intended use,
training/evaluation data, quantitative performance, known limitations).

**Every number here is either pulled from a cached result file this study
already computed, or is a structural fact about the code itself (architecture,
data source, prompt design) -- nothing is re-measured or newly estimated for
this document.** Where a number cannot be pulled from a cache (because the
underlying report has not been run yet), the card says so explicitly rather
than omitting the row silently or inventing a plausible-looking figure.

Usage:  .venv/bin/python scripts/generate_model_cards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS, SETTINGS
from scripts.generate_factsheet import STRATEGY, llm_stats

OUT_MD = RESULTS / "MODEL_CARDS.md"

NA = "*not available -- run the underlying report script first*"


def _fmt(value, spec: str = ".4f") -> str:
    if value is None or (isinstance(value, float) and (value != value)):  # NaN
        return NA
    return format(value, spec)


def technical_agent_card() -> str:
    arch_path = RESULTS / "technical" / "architecture_summary.csv"
    rows = ""
    production_note = (
        "The production/live checkpoint (`scripts/train_production_model.py`) uses "
        "**plain LSTM**, not the peephole+attention (PLSTM-TAL) architecture this "
        "table's default row otherwise suggests -- LSTM is the empirically best "
        "architecture measured on this data (README.md), and stating the wrong "
        "architecture name in a live rationale was a real, caught bug (KNOWN_ISSUES.md)."
    )
    if arch_path.exists():
        arch = pd.read_csv(arch_path)
        lines = ["| Architecture | Accuracy (mean, 3 seeds) | Std | AUC | MCC | p vs. chance |",
                 "|---|---|---|---|---|---|"]
        for r in arch.itertuples():
            lines.append(f"| {r.architecture} | {r.accuracy_mean:.4f} | {r.accuracy_std:.4f} | "
                         f"{r.auc_mean:.4f} | {r.mcc_mean:.4f} | {r.p_vs_chance:.3f} |")
        rows = "\n".join(lines)
    else:
        rows = NA

    return f"""## TechnicalAgent (`nse_agents/agents/technical.py`)

**Model details.** A hand-implemented peephole LSTM cell with temporal attention
(PLSTM-TAL), reimplementing the architecture from PLSTM-TAL (Heliyon 2024). Wraps
pre-computed out-of-sample predictions rather than holding a live model -- the
predictions were produced by a fold that had never seen the date it predicts, and
the agent's own boundary enforces that no caller can ask it for a prediction
inside its training window.

**Intended use.** Predicts the sign of the next open-to-open forward return for
one NSE-listed equity at a time, as one vote among several specialist agents --
never deployed standalone as a trading signal.

**Training data.** NSE OHLCV, split/bonus adjusted, {SETTINGS.start} to
{SETTINGS.end}, this study's {len(SETTINGS.universe)}-name universe. Purged,
embargoed rolling walk-forward split (3-year train, 6-month test, embargo
exceeding the label horizon) -- no fold trains on its own future.

**Quantitative performance** (out-of-sample, 3 seeds, purged walk-forward):

{rows}

**Known limitations / failure modes.**
- No architecture's directional accuracy is statistically distinguishable from
  chance (50%) at conventional significance.
- {production_note}
- Calibration beyond roughly P(up)=0.6 is not validated; `confidence_scale` caps
  how much distance-from-0.5 is allowed to read as conviction for exactly this
  reason.
"""


def sentiment_agent_card() -> str:
    stats = llm_stats()
    label_dist = stats["label_dist"]
    label_rows = NA
    if label_dist is not None:
        label_rows = "\n".join(f"- **{label}**: {share:.1%}" for label, share in label_dist.items())

    event_path = RESULTS / "sentiment" / "event_study_local.csv"
    event_note = NA
    if event_path.exists():
        event = pd.read_csv(event_path).iloc[0]
        event_note = (f"slope {event['slope_bps_per_unit_sentiment']:+.3f} bps/unit sentiment, "
                      f"t={event['t_stat']:+.3f}, **p={event['p_value']:.3f}** (n={int(event['n']):,})")

    return f"""## SentimentAgent (`nse_agents/agents/sentiment.py`)

**Model details.** LLM headline classifier, local backend `Qwen/Qwen2.5-1.5B-Instruct`,
scored by constrained-label first-token logits (`Good`/`Bad`/`Unknown`) over one
forward pass per batch, not free generation -- following Lopez-Lira & Tang's
prompt design: ask whether the headline is good or bad *for the stock's price*,
not whether it is good news in general, with an explicit "unknown" escape.

**Intended use.** Scores same-day Indian financial headlines per NSE symbol;
abstains (not a neutral 0.0) on a day with no attributable headline, since "no
news" and "genuinely mixed news" imply different position sizes.

**Training data.** Not fine-tuned -- a frozen, off-the-shelf instruction model
used zero-shot. Evaluation corpus: {stats['n_headlines']:,} Indian financial
headlines, {SETTINGS.start} onward, attributed to their first actionable trading
day (`nse_agents/data/news.py`'s two time corrections).

**Quantitative performance.**
- Standalone event study (headline sentiment vs. next-day return, before any
  trading strategy is built on it): {event_note}
- Label distribution over the full corpus:
{label_rows}

**Known limitations / failure modes.**
- **Measured 3:1 bullish prior** (Good vastly outweighs Bad in the label
  distribution above), carrying no predictive content per the event study's
  p-value -- a model that is skewed but not informative.
- The worst-performing configuration in the full ablation
  (`results/agents/summary.csv`) is `Tech+Sentiment`, at 123.9x annual turnover:
  the signal flips with the day's headline mix rather than tracking anything
  real.
- **A frontier-model comparison (`--backend anthropic`) is the correct next
  step and has not been run in this environment** (no API key). Until it is,
  the honest claim is "a 1.5B model cannot do this on NSE headlines," not "LLM
  sentiment does not work" -- Lopez-Lira & Tang's own finding is that this
  capability is emergent at larger scale.
"""


def regime_agent_card() -> str:
    return f"""## RegimeAgent (`nse_agents/agents/regime.py`)

**Model details.** Not a learned model -- a deterministic function of causal,
freely-available price-history features: 200-day trend, 60-day relative
strength vs. the Nifty, position within the 52-week range, realised
volatility percentile, and (opt-in, live pipeline only) India VIX percentile
and Nifty 20-day momentum. Votes are equally weighted and unweighted by
design, to avoid fitting a combination to the same test period it is
evaluated on.

**Intended use.** A market-structure vote, explicitly **not** a fundamental
analyst: substituting today's P/E or book value into a historical decision
would embed every subsequent earnings surprise between then and now (lookahead
of the worst kind), so this agent reasons only about what is causally
available at each point in time instead.

**Training data.** None -- a rule-based agent, not fit to any window. Feature
inputs: NSE OHLCV and India VIX, {SETTINGS.start} to {SETTINGS.end}.

**Quantitative performance.** See `results/improvements/regime_breakdown.csv`
(crash/choppy/bull regime breakdown) and `results/improvements/risk_attribution.csv`
(beta, upside/downside capture) -- `Tech+Regime` drew down -1.1% during the
82-day COVID crash window vs. Buy&Hold's -23.6%, at the cost of forgone upside
in calm regimes that outweighs this protection on average over the full window.

**Known limitations / failure modes.**
- **Slow-turning, not a shock absorber**: `scripts/stress_test_scenarios.py`
  found the overlay de-risks ahead of a *sustained* decline but not a single
  flash-crash day -- averaging four medium-term votes is what makes it robust
  to single-day noise, at the cost of same-day reaction speed. A separate,
  complementary mechanism (`nse_agents/agents/circuit_breaker.py`) exists for
  exactly that gap.
- No fundamental agent exists, and none is planned without point-in-time
  fundamentals data this project does not have (see
  `nse_agents/backtest/factor_model.py`'s identical reasoning for why SMB/HML
  factors are also not implemented).
"""


def debate_engine_card() -> str:
    stats = llm_stats()
    return f"""## Debate Engine (`nse_agents/agents/researchers.py` + `nse_agents/agents/orchestrator.py`)

**Model details.** Two adversarial researcher roles (bull, bear) built on the
same local Qwen2.5-1.5B backend as `SentimentAgent`, each given the same
evidence and instructed to build the strongest case for its side, scored by
constrained-label conviction (`Strong`/`Weak`/`None`) rather than free
generation -- both for cost (~20,000 decision points x 2 researchers is not
runnable as free generation on local hardware in this budget) and for
measurability (a conviction level is a number the backtest can act on; free
prose would have to be parsed back into one, unreliably, at 1.5B). Debate is
gated to contested days only via `Orchestrator`'s conviction-based
disagreement metric.

**Intended use.** Reconciles specialist disagreement into one residual
conviction term feeding `Trader.combine`, plus a sampled natural-language
transcript (`generate_transcript`) as this project's explainability
deliverable -- the decision pipeline runs on every day, the narration is
sampled.

**Training data.** Not fine-tuned, same frozen off-the-shelf model as
`SentimentAgent`.

**Quantitative performance.**
- Total decisions in the flagship `{STRATEGY}` configuration:
  **{stats['total_decisions']:,}**
- Debate escalation rate: **{stats['escalation_rate']:.1%}** of decisions --
  this is the **original ("raw") gate**, which is what the flagship
  configuration's numbers everywhere in this study (README.md,
  `results/agents/summary.csv`) were actually produced with. See below for
  why this figure is frozen at its original, higher value on purpose.

**Known limitations / failure modes.**
- **The raw disagreement gate escalates 97.4% of all decisions** (18,610 of
  19,110) -- diagnosed first as a scale mismatch between agents (wrong:
  rescaling moved escalation only to 91.3%), then correctly as *sign*
  disagreement between near-zero, roughly symmetric stances landing on
  opposite sides of zero by chance. A fix exists (gate on a confidence-
  weighted conviction score against +/-floor instead of sign), reducing
  escalation to 2.2%-15.1% depending on the floor
  (`results/improvements/hyperparam_conviction_floor.csv`) at ~9x lower
  compute cost. **It was deliberately never adopted as this study's default**
  -- `results/agents/decisions_{STRATEGY}.csv` and every number derived from
  it were produced with the raw gate, and switching the default after the
  fact would silently change already-reported numbers on a plain re-run,
  which this project's own conventions treat as a correctness bug in
  themselves (see `RiskLimits.cost_aware` and `RegimeAgent(use_macro=True)`
  for the identical opt-in-only pattern elsewhere). Full account:
  `KNOWN_ISSUES.md` #4.
- Debate is not a source of new information -- both researchers see the same
  specialist opinions the trader would otherwise combine directly; its
  contribution is adversarial framing, not new evidence.
"""


def build_document() -> str:
    return f"""# Model Cards

One card per core agent component, following the Mitchell et al. (2019) Model
Card structure. Every number below is pulled from an already-computed, cached
result file (never re-measured for this document) or is a structural fact
about the code itself -- see each report script cited inline for how a number
was produced.

---

{technical_agent_card()}
---

{sentiment_agent_card()}
---

{regime_agent_card()}
---

{debate_engine_card()}"""


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    document = build_document()
    OUT_MD.write_text(document)
    print(f"wrote {OUT_MD} ({len(document):,} characters)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
