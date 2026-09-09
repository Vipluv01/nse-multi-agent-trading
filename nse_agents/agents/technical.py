"""Technical analyst: the walk-forward PLSTM-TAL, wrapped as an agent.

Reads pre-computed out-of-sample predictions rather than holding the model, for
one reason that is not convenience: the predictions in that file were produced
by a model that had never seen the date it is predicting. If the agent held a
live model, nothing in the orchestration layer would stop it being asked for a
prediction on a date inside its own training window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Opinion


class TechnicalAgent:
    name = "technical"

    def __init__(self, oos: pd.DataFrame, confidence_scale: float = 4.0):
        frame = oos.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self._lookup = frame.set_index(["symbol", "date"]).sort_index()
        # A near-0.5 probability is a coin flip and must not read as conviction.
        # Confidence rises with distance from 0.5, saturating well before the
        # extremes, because this classifier's calibration beyond ~0.6 is thin.
        self.confidence_scale = confidence_scale

    def opine(self, symbol: str, date: pd.Timestamp) -> Opinion:
        try:
            row = self._lookup.loc[(symbol, pd.Timestamp(date))]
        except KeyError:
            return Opinion.abstain(self.name, "no out-of-sample prediction for this date")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        prob = float(row["prob_up"])
        stance = float(np.clip((prob - 0.5) * 2.0, -1.0, 1.0))
        confidence = float(np.clip(abs(prob - 0.5) * self.confidence_scale, 0.0, 1.0))

        attention = row.get("attn_recent5", np.nan)
        focus = (
            f" The attention layer put {float(attention):.0%} of its weight on the last "
            f"5 sessions."
            if attention == attention  # NaN check
            else ""
        )
        direction = "upward" if stance > 0 else "downward"
        return Opinion(
            agent=self.name,
            stance=stance,
            confidence=confidence,
            rationale=(
                f"PLSTM-TAL assigns P(up)={prob:.3f} for {symbol} at the next open, "
                f"a mild {direction} tilt.{focus}"
            ),
            evidence={"prob_up": prob, "attn_recent5": float(attention) if attention == attention else None},
        )
