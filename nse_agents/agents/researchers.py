"""Bull and Bear researchers: the structured debate layer.

TradingAgents has bull and bear researchers argue before a trader decides. The
mechanism that matters is *adversarial framing*: the same evidence is handed to
two agents instructed to build opposite cases, and the residual between their
convictions is the debate's output. An agent asked neutrally tends to echo the
specialists it was given; asked to prosecute a side, it surfaces the strongest
counter-evidence, which is what a lone forecaster never does.

Conviction is read by constrained scoring over {Strong, Weak, None} rather than
free generation. Two reasons, and the second is the important one:

1. Cost. Nearly 20,000 decision points times two researchers is not runnable as
   free generation on local hardware inside this project's budget.
2. Measurability. A conviction *level* is a number the backtest can act on and
   the ablation can switch off. Free prose would have to be parsed back into a
   number anyway, and unreliably at 1.5B.

Full natural-language transcripts are still generated -- for a sample of
decisions, via ``generate_transcript`` -- because the readable argument is the
project's explainability deliverable. What is sampled is the *narration*; the
decision pipeline itself runs on every day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Opinion

CONVICTION_LABELS = ["Strong", "Weak", "None"]
CONVICTION_VALUES = {"Strong": 1.0, "Weak": 0.45, "None": 0.0}

BULL_SYSTEM = (
    "You are a bullish equity researcher on an Indian trading desk. You are "
    "given analyst findings on an NSE-listed stock. Build the strongest "
    "responsible case to BUY it for the next trading day. State how strong that "
    "case is in one word: Strong, Weak, or None. Answer None if the evidence "
    "does not support buying."
)

BEAR_SYSTEM = (
    "You are a bearish equity researcher on an Indian trading desk. You are "
    "given analyst findings on an NSE-listed stock. Build the strongest "
    "responsible case to AVOID or SELL it for the next trading day. State how "
    "strong that case is in one word: Strong, Weak, or None. Answer None if the "
    "evidence does not support avoiding it."
)

BRIEF_TEMPLATE = (
    "Stock: {symbol} (NSE)\n"
    "Date: {date}\n"
    "Analyst findings:\n{findings}\n\n"
    "How strong is your case? Answer Strong, Weak, or None."
)

TRANSCRIPT_SUFFIX = (
    "\n\nNow write your argument in two sentences, citing the findings above."
)


def format_findings(opinions: list[Opinion]) -> str:
    lines = []
    for opinion in opinions:
        if opinion.abstained:
            lines.append(f"- {opinion.agent}: no view ({opinion.rationale})")
        else:
            lines.append(
                f"- {opinion.agent}: stance {opinion.stance:+.2f} "
                f"(confidence {opinion.confidence:.2f}) -- {opinion.rationale}"
            )
    return "\n".join(lines)


@dataclass
class DebateOutcome:
    bull_conviction: float
    bear_conviction: float
    bull_label: str
    bear_label: str

    @property
    def residual(self) -> float:
        """Net debate signal in [-1, 1]."""
        return float(np.clip(self.bull_conviction - self.bear_conviction, -1.0, 1.0))

    @property
    def contested(self) -> bool:
        """Both sides hold a real case -- genuine disagreement, not consensus."""
        return self.bull_conviction > 0.3 and self.bear_conviction > 0.3


def run_debates(
    backend,
    briefs: list[tuple[str, str, str]],
    batch_size: int = 16,
    progress: bool = True,
) -> list[DebateOutcome]:
    """Score bull and bear conviction for every (symbol, date, findings) brief."""
    if not briefs:
        return []
    prompts = [
        BRIEF_TEMPLATE.format(symbol=s, date=d, findings=f) for s, d, f in briefs
    ]
    outcomes: list[DebateOutcome] = []
    step = max(batch_size * 20, 200)
    for start in range(0, len(prompts), step):
        chunk = prompts[start : start + step]
        bull = backend.classify_batch(BULL_SYSTEM, chunk, CONVICTION_LABELS, batch_size=batch_size)
        bear = backend.classify_batch(BEAR_SYSTEM, chunk, CONVICTION_LABELS, batch_size=batch_size)
        for bu, be in zip(bull, bear):
            # Expected conviction under the label distribution, not the argmax:
            # a 51/49 Strong/Weak split should not read the same as 99/1.
            bull_value = sum(CONVICTION_VALUES[k] * v for k, v in bu.as_dict().items())
            bear_value = sum(CONVICTION_VALUES[k] * v for k, v in be.as_dict().items())
            outcomes.append(DebateOutcome(bull_value, bear_value, bu.top(), be.top()))
        if progress:
            print(f"  debated {min(start + step, len(prompts)):,}/{len(prompts):,}", flush=True)
    return outcomes


def generate_transcript(backend, symbol: str, date: str, findings: str) -> dict[str, str]:
    """Free-form bull and bear arguments, for the explainability artefact."""
    brief = BRIEF_TEMPLATE.format(symbol=symbol, date=date, findings=findings) + TRANSCRIPT_SUFFIX
    return {
        "bull": backend.complete(BULL_SYSTEM, brief, max_tokens=180).text,
        "bear": backend.complete(BEAR_SYSTEM, brief, max_tokens=180).text,
    }
