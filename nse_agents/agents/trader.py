"""Trader: combines the specialists and the debate into one auditable call.

The combination is a **confidence-weighted mean of stances**, not a learned
meta-model. That is a deliberate restraint: with roughly 1,900 out-of-sample
days available, fitting a stacker on top of four agents would be fitting the
test period, and the resulting Sharpe would measure the stacker's overfitting
rather than the agents' information. A fixed, stated weighting can be wrong,
but it cannot be tuned to the answer.

Abstentions are excluded from the mean rather than counted as zero, so a day
with no news is decided by the agents that do have a view, at the confidence
those agents actually hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Decision, Opinion
from .researchers import DebateOutcome
from .risk import RiskManager, RiskState


@dataclass
class TraderConfig:
    # Weights across specialists. Equal by default; the ablation study varies
    # which agents are present rather than tuning these.
    agent_weights: dict[str, float] | None = None
    debate_weight: float = 0.35
    buy_threshold: float = 0.05
    use_debate: bool = True


class Trader:
    name = "trader"

    def __init__(self, config: TraderConfig | None = None, risk: RiskManager | None = None):
        self.config = config or TraderConfig()
        self.risk = risk or RiskManager()

    def combine(
        self, opinions: list[Opinion], debate: DebateOutcome | None
    ) -> tuple[float, float]:
        """Return ``(score, aggregate_confidence)``."""
        weights = self.config.agent_weights or {}
        active = [o for o in opinions if not o.abstained]
        if not active:
            return 0.0, 0.0

        numerator, denominator = 0.0, 0.0
        for opinion in active:
            w = weights.get(opinion.agent, 1.0) * opinion.confidence
            numerator += w * opinion.stance
            denominator += w
        score = numerator / denominator if denominator > 1e-9 else 0.0
        confidence = float(np.mean([o.confidence for o in active]))

        if debate is not None and self.config.use_debate:
            score = (1.0 - self.config.debate_weight) * score + self.config.debate_weight * debate.residual
            if debate.contested:
                # Both cases stand up: act smaller, not differently.
                confidence *= 0.7
        return float(np.clip(score, -1.0, 1.0)), float(np.clip(confidence, 0.0, 1.0))

    def decide(
        self,
        date: pd.Timestamp,
        symbol: str,
        opinions: list[Opinion],
        debate: DebateOutcome | None,
        realized_vol: float | None,
        state: RiskState,
    ) -> Decision:
        score, confidence = self.combine(opinions, debate)
        size, notes = self.risk.size(score, confidence, realized_vol, state)

        if score > self.config.buy_threshold and size > 0:
            action = "BUY"
        elif score > self.config.buy_threshold:
            action = "HOLD"   # wanted to buy, risk layer refused
        else:
            action = "FLAT"

        return Decision(
            date=pd.Timestamp(date),
            symbol=symbol,
            action=action,
            score=score,
            size=size,
            opinions=opinions,
            rationale=self.explain(symbol, action, score, confidence, opinions, debate, notes),
            risk_notes=notes,
            debated=debate is not None,
        )

    def explain(
        self,
        symbol: str,
        action: str,
        score: float,
        confidence: float,
        opinions: list[Opinion],
        debate: DebateOutcome | None,
        risk_notes: list[str],
    ) -> str:
        """The auditable rationale: what was said, by whom, and what overrode it."""
        parts = [f"{action} {symbol} (score {score:+.3f}, confidence {confidence:.2f})."]
        for opinion in opinions:
            if opinion.abstained:
                parts.append(f"[{opinion.agent}] abstained: {opinion.rationale}")
            else:
                parts.append(f"[{opinion.agent} {opinion.stance:+.2f}] {opinion.rationale}")
        if debate is not None:
            parts.append(
                f"[debate] bull case {debate.bull_label} ({debate.bull_conviction:.2f}) vs "
                f"bear case {debate.bear_label} ({debate.bear_conviction:.2f}); "
                f"net {debate.residual:+.2f}"
                + (" -- contested, size reduced." if debate.contested else ".")
            )
        if risk_notes:
            parts.append("[risk] " + "; ".join(risk_notes) + ".")
        return " ".join(parts)
