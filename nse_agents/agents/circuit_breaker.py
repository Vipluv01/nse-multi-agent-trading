"""Instant de-risking triggers, complementing `RegimeAgent`'s slow-turning trend
signal rather than duplicating it.

`scripts/stress_test_scenarios.py` found, honestly, that the regime overlay is a
slow-turning trend signal: it fully de-risks ahead of a *sustained* decline but a
single flash-crash day is not enough on its own to flip a name past
`buy_threshold` (see KNOWN_ISSUES.md #13). That is not a bug in the regime overlay
-- averaging four medium-term votes is what makes it robust to noise on any single
day, and a mechanism built to react instantly to one bad day is a different,
complementary tool, not a fix to the first one. This module is that tool: two
purely mechanical, symbol-level triggers that bypass the combined-opinion vote
entirely and force a position flat the same day the trigger fires.

Both triggers are deliberately **deterministic, not learned** -- the same
reasoning `risk.py`'s own module docstring gives for why the risk manager is
never an LLM applies here without modification: a circuit breaker that can be
argued out of firing is not a circuit breaker.

1. **Overnight gap-down**: today's open at least `gap_down_threshold` (3%, as a
   negative fraction) below yesterday's close. A gap is information the close-to-
   close medium-term votes cannot see by construction -- it happens *between*
   sessions.
2. **ATR expansion**: today's high-low range beyond `atr_multiple` (2.5x) times
   the **prior** 14-session average true range. Computed causally (`shift(1)`,
   the same discipline `nse_agents/data/audit.py`'s outlier check uses and for
   the same reason): today's own huge range must never inflate the reference it
   is being measured against, or a circuit breaker day could never trip its own
   ATR check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GAP_DOWN_THRESHOLD = -0.03
ATR_MULTIPLE = 2.5
ATR_PERIOD = 14


@dataclass(frozen=True)
class CircuitBreakerLimits:
    gap_down_threshold: float = GAP_DOWN_THRESHOLD
    atr_multiple: float = ATR_MULTIPLE
    atr_period: int = ATR_PERIOD


@dataclass(frozen=True)
class CircuitBreakerTrigger:
    triggered: bool
    reason: str | None = None


def average_true_range(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """Causal rolling ATR: today's row is the mean true range over the *prior*
    ``period`` sessions, never including today's own high/low/close. True range
    is the standard three-way max (Wilder 1978): today's own range, and the gap
    from yesterday's close to today's high or low, whichever is larger --
    a pure high/low range alone misses a gap that itself constitutes a big move.
    """
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean().shift(1)


class CircuitBreaker:
    name = "circuit_breaker"

    def __init__(self, limits: CircuitBreakerLimits | None = None):
        self.limits = limits or CircuitBreakerLimits()

    def check(
        self,
        prev_close: float,
        today_open: float,
        today_high: float,
        today_low: float,
        atr: float | None,
    ) -> CircuitBreakerTrigger:
        """Evaluate both triggers for one symbol on one day. Gap-down is
        checked first -- it needs only yesterday's close and today's open, so
        it is available even on a name with too little history for a 14-day
        ATR yet, while the ATR check is skipped (never blocks a genuine gap
        finding) when ``atr`` is ``None``/NaN/non-positive."""
        if prev_close and prev_close > 0:
            gap = today_open / prev_close - 1.0
            if gap <= self.limits.gap_down_threshold:
                return CircuitBreakerTrigger(
                    True,
                    f"gap-down {gap:+.1%} at the open (prev close {prev_close:.2f} -> "
                    f"open {today_open:.2f}), past the {self.limits.gap_down_threshold:+.0%} trigger",
                )

        if atr is not None and not (isinstance(atr, float) and np.isnan(atr)) and atr > 0:
            daily_range = today_high - today_low
            if daily_range > self.limits.atr_multiple * atr:
                return CircuitBreakerTrigger(
                    True,
                    f"daily range {daily_range:.2f} is {daily_range / atr:.1f}x the prior "
                    f"{self.limits.atr_period}-day ATR ({atr:.2f}), past the "
                    f"{self.limits.atr_multiple}x trigger",
                )

        return CircuitBreakerTrigger(False)

    def check_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Vectorised convenience: given an OHLC frame (sorted by date, with
        ``open``/``high``/``low``/``close`` columns), return one
        ``CircuitBreakerTrigger`` per row. Row 0 never triggers on the gap
        check (no prior close) and needs ``atr_period`` prior rows before the
        ATR check can fire -- both degrade to "no trigger", never a crash on
        insufficient history, matching every other agent's own abstain-on-
        insufficient-data convention in this project."""
        atr = average_true_range(frame, self.limits.atr_period)
        prev_close = frame["close"].shift(1)
        results = []
        for i in range(len(frame)):
            results.append(self.check(
                prev_close=float(prev_close.iloc[i]) if pd.notna(prev_close.iloc[i]) else 0.0,
                today_open=float(frame["open"].iloc[i]),
                today_high=float(frame["high"].iloc[i]),
                today_low=float(frame["low"].iloc[i]),
                atr=float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else None,
            ))
        return pd.Series(results, index=frame.index)
