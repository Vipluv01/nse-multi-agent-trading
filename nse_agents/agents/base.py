"""The common agent contract.

Every specialist returns the same shape -- a signed stance, a confidence, and a
rationale in words -- so the trader can combine them and, more importantly, so
the final recommendation can be traced back to which agent said what. That
traceability is the "explainable" half of the project's claim; an ensemble that
averaged raw scores would score identically and explain nothing.

An agent that lacks the evidence to form a view returns ``abstain()`` rather
than a neutral stance. The distinction is load-bearing: "no news today" and
"the news is genuinely mixed" imply different position sizes, and collapsing
them to 0.0 is how a sentiment model ends up silently trading on its own prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Opinion:
    agent: str
    stance: float          # -1 (strong sell) .. +1 (strong buy)
    confidence: float      # 0 .. 1
    rationale: str
    abstained: bool = False
    evidence: dict = field(default_factory=dict)

    @staticmethod
    def abstain(agent: str, why: str) -> "Opinion":
        return Opinion(agent, 0.0, 0.0, why, abstained=True)

    def weighted(self) -> float:
        return 0.0 if self.abstained else self.stance * self.confidence


@dataclass
class Decision:
    date: pd.Timestamp
    symbol: str
    action: str            # BUY | HOLD | FLAT
    score: float           # the pre-risk combined score, -1 .. 1
    size: float            # post-risk target weight, 0 .. max_weight
    opinions: list[Opinion]
    rationale: str
    risk_notes: list[str] = field(default_factory=list)
    debated: bool = False

    def to_row(self) -> dict:
        row = {
            "date": self.date,
            "symbol": self.symbol,
            "action": self.action,
            "score": self.score,
            "size": self.size,
            "debated": self.debated,
            "rationale": self.rationale,
            "risk_notes": "; ".join(self.risk_notes),
        }
        for opinion in self.opinions:
            row[f"{opinion.agent}_stance"] = opinion.stance
            row[f"{opinion.agent}_conf"] = opinion.confidence
            row[f"{opinion.agent}_abstained"] = opinion.abstained
        return row


class Agent(Protocol):
    name: str

    def opine(self, symbol: str, date: pd.Timestamp) -> Opinion: ...
