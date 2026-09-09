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

    def __init__(self, oos: pd.DataFrame, confidence_scale: float = 4.0, architecture: str = "PLSTM-TAL"):
        """``architecture`` names whichever model actually produced ``oos`` in the
        agent's own rationale text. Defaults to PLSTM-TAL, matching the main
        walk-forward study's default TrainConfig -- but the live pipeline passes
        the production checkpoint's real architecture (plain LSTM, per README's
        own empirical result), since stating the wrong model name in an
        explainability rationale is a real accuracy bug, not a cosmetic one.
        """
        frame = oos.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self._lookup = frame.set_index(["symbol", "date"]).sort_index()
        self.architecture = architecture
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

        attention = row.get("attn_recent5", None)
        # Two cases read as "no attention weight to report", not one: NaN (a
        # walk-forward run where the CSV column exists but this row's value is
        # missing) and None (a model with no attention module at all, e.g. the
        # production checkpoint deliberately uses the empirically best
        # architecture, plain LSTM, per README -- which has none). `x != x` is
        # only true for NaN, so it must be checked before `is None` would ever
        # apply to a real float.
        has_attention = attention is not None and not (isinstance(attention, float) and attention != attention)
        focus = (
            f" The attention layer put {float(attention):.0%} of its weight on the last "
            f"5 sessions."
            if has_attention
            else ""
        )
        direction = "upward" if stance > 0 else "downward"
        return Opinion(
            agent=self.name,
            stance=stance,
            confidence=confidence,
            rationale=(
                f"{self.architecture} assigns P(up)={prob:.3f} for {symbol} at the next open, "
                f"a mild {direction} tilt.{focus}"
            ),
            evidence={"prob_up": prob, "attn_recent5": float(attention) if has_attention else None},
        )
