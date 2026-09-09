"""Risk manager.

**Deliberately deterministic, not an LLM.** Every other agent here is a
forecaster and is allowed to be wrong; the risk layer is a constraint system,
and a constraint that can hallucinate is not a constraint. TradingAgents places
a risk team in the loop, and this implements that role -- but position limits,
volatility targeting and a drawdown brake are rules an auditor must be able to
read, re-derive and trust to fire identically every time. Asking a 1.5B model
to decide whether a limit has been breached would be the least defensible
design choice in the project.

The brake is causal: it acts on the equity curve realised *up to yesterday*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import SETTINGS, CostModel

TRADING_DAYS = 252


@dataclass
class RiskLimits:
    max_weight_per_name: float = SETTINGS.max_weight_per_name
    max_gross_exposure: float = 1.0
    target_annual_vol: float = 0.20
    vol_scaling: bool = True
    drawdown_brake_at: float = -0.15   # start cutting once the book is 15% down
    drawdown_flat_at: float = -0.30    # fully flat at 30%
    min_confidence: float = 0.10       # below this, the desk does not act
    # Cost-threshold filter. Defaults OFF: every number already published in
    # README.md was computed without it, and turning it on by default would
    # silently change those results on a plain re-run. The live pipeline
    # (scripts/run_live_signal.py) is the one caller that enables it.
    cost_aware: bool = False
    costs: CostModel = field(default_factory=CostModel)
    min_edge_over_cost_bps: float = 0.0  # require edge to clear cost by this much, not just tie


@dataclass
class RiskState:
    """Rolling state the manager needs. Updated only with realised history."""

    equity_peak: float = 1.0
    equity: float = 1.0

    @property
    def drawdown(self) -> float:
        return self.equity / self.equity_peak - 1.0

    def update(self, daily_return: float) -> None:
        self.equity *= 1.0 + daily_return
        self.equity_peak = max(self.equity_peak, self.equity)


class RiskManager:
    name = "risk"

    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()

    def size(
        self,
        score: float,
        confidence: float,
        realized_vol: float | None,
        state: RiskState,
    ) -> tuple[float, list[str]]:
        """Turn a combined score into a target weight, with an audit trail."""
        notes: list[str] = []

        if score <= 0:
            return 0.0, ["score not positive; long-only book stays flat"]
        if confidence < self.limits.min_confidence:
            return 0.0, [f"aggregate confidence {confidence:.2f} below floor "
                         f"{self.limits.min_confidence:.2f}; no position"]

        if self.limits.cost_aware and realized_vol and realized_vol > 1e-6:
            # A single-name expected-return forecast from a score alone would
            # be invented precision -- this study's whole finding is that
            # these scores carry ~0 real predictive edge. Instead, the edge is
            # estimated the way a desk actually would from a conviction score:
            # scale the stock's own typical daily move by how convinced the
            # combined signal is (score x confidence), and refuse to trade
            # unless that estimate clears the real round-trip cost. This is a
            # heuristic, stated as one, not a calibrated forecast.
            daily_move_bps = (realized_vol / np.sqrt(TRADING_DAYS)) * 1e4
            expected_edge_bps = score * confidence * daily_move_bps
            round_trip_bps = self.limits.costs.cost_bps("buy") + self.limits.costs.cost_bps("sell")
            required_bps = round_trip_bps + self.limits.min_edge_over_cost_bps
            if expected_edge_bps < required_bps:
                return 0.0, [
                    f"estimated edge {expected_edge_bps:.1f}bps does not clear "
                    f"round-trip cost {round_trip_bps:.1f}bps "
                    f"(required {required_bps:.1f}bps); no position"
                ]

        weight = self.limits.max_weight_per_name
        notes.append(f"base weight {weight:.3f}")

        if self.limits.vol_scaling and realized_vol and realized_vol > 1e-6:
            scale = float(np.clip(self.limits.target_annual_vol / realized_vol, 0.25, 1.5))
            weight *= scale
            notes.append(
                f"vol scaling x{scale:.2f} (realised {realized_vol:.1%} vs target "
                f"{self.limits.target_annual_vol:.1%})"
            )

        drawdown = state.drawdown
        if drawdown <= self.limits.drawdown_flat_at:
            return 0.0, notes + [f"drawdown {drawdown:.1%} at or beyond flat limit; book halted"]
        if drawdown < self.limits.drawdown_brake_at:
            span = self.limits.drawdown_brake_at - self.limits.drawdown_flat_at
            taper = float(np.clip((drawdown - self.limits.drawdown_flat_at) / span, 0.0, 1.0))
            weight *= taper
            notes.append(f"drawdown brake: {drawdown:.1%} drawdown, size x{taper:.2f}")

        weight = min(weight, self.limits.max_weight_per_name)
        return float(max(weight, 0.0)), notes

    def apply_gross_cap(
        self, sized: list[tuple[float, list[str]]]
    ) -> list[tuple[float, list[str]]]:
        """Scale a day's whole book down to the gross-exposure limit.

        Per-name sizing cannot enforce this on its own: ten names each inside a
        20% per-name cap is a 200% gross book. Without this pass the backtest
        quietly runs leverage it never pays for, which inflates return and
        Sharpe alike. Scaling is proportional, so relative conviction across
        names is preserved.
        """
        total = sum(weight for weight, _ in sized)
        if total <= self.limits.max_gross_exposure or total <= 0:
            return sized
        scale = self.limits.max_gross_exposure / total
        note = (f"gross exposure {total:.2f} over limit "
                f"{self.limits.max_gross_exposure:.2f}; book scaled x{scale:.2f}")
        return [(weight * scale, notes + [note]) for weight, notes in sized]
