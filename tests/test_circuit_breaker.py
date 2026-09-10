"""Tests for nse_agents/agents/circuit_breaker.py and its wiring into
RiskManager.size() / Trader.decide()."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.circuit_breaker import (
    GAP_DOWN_THRESHOLD,
    CircuitBreaker,
    CircuitBreakerLimits,
    CircuitBreakerTrigger,
    average_true_range,
)
from nse_agents.agents.risk import RiskLimits, RiskManager, RiskState
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.agents.base import Opinion


def _ohlc_frame(n=40, price=100.0, daily_range=1.0, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = price + np.cumsum(rng.normal(0, 0.1, n))
    high = close + daily_range / 2
    low = close - daily_range / 2
    open_ = close  # not used by ATR, only by the gap check elsewhere
    return pd.DataFrame({"date": dates, "open": open_, "high": high, "low": low, "close": close})


def test_average_true_range_is_causal_shift_by_one():
    """Perturbing only the LAST row's high/low must not change any earlier
    row's ATR value -- the same no-lookahead discipline this project applies
    to every other rolling feature (see test_no_lookahead.py)."""
    frame = _ohlc_frame(n=30)
    before = average_true_range(frame).copy()
    tampered = frame.copy()
    tampered.loc[tampered.index[-1], ["high", "low"]] = [500.0, 1.0]
    after = average_true_range(tampered)
    pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])


def test_average_true_range_excludes_todays_own_range_from_its_own_reference():
    """Row t's ATR must be computed from the prior period only -- a huge
    range on day t itself must not inflate the value ATR[t] reports."""
    frame = _ohlc_frame(n=30, daily_range=1.0)
    frame.loc[frame.index[-1], ["high", "low"]] = [200.0, 1.0]  # today's own huge range
    atr = average_true_range(frame, period=14)
    # ATR at the huge-range row reflects the prior 14 (normal) days, not today.
    assert atr.iloc[-1] < 5.0


def test_gap_down_triggers_at_or_beyond_the_threshold():
    breaker = CircuitBreaker()
    trigger = breaker.check(prev_close=100.0, today_open=97.0, today_high=98.0, today_low=95.0, atr=None)
    assert trigger.triggered
    assert "gap-down" in trigger.reason


def test_gap_down_does_not_trigger_just_inside_the_threshold():
    breaker = CircuitBreaker()
    trigger = breaker.check(prev_close=100.0, today_open=97.5, today_high=98.0, today_low=97.0, atr=None)
    assert not trigger.triggered


def test_gap_down_threshold_is_the_documented_default():
    assert GAP_DOWN_THRESHOLD == -0.03


def test_atr_expansion_triggers_beyond_the_multiple():
    breaker = CircuitBreaker()
    # ATR of 2.0, range of 6.0 -> 3.0x, beyond the default 2.5x.
    trigger = breaker.check(prev_close=100.0, today_open=100.0, today_high=103.0, today_low=97.0, atr=2.0)
    assert trigger.triggered
    assert "ATR" in trigger.reason


def test_atr_expansion_does_not_trigger_within_the_multiple():
    breaker = CircuitBreaker()
    trigger = breaker.check(prev_close=100.0, today_open=100.0, today_high=101.0, today_low=99.0, atr=2.0)
    assert not trigger.triggered


def test_atr_check_is_skipped_when_atr_is_unavailable():
    """Too little history for a 14-day ATR yet must never crash or block a
    real gap-down finding -- it degrades to 'no ATR signal', not an error."""
    breaker = CircuitBreaker()
    trigger = breaker.check(prev_close=100.0, today_open=99.5, today_high=100.0, today_low=95.0, atr=None)
    assert not trigger.triggered  # neither gap (only -0.5%) nor ATR (skipped) fires


def test_gap_down_checked_before_atr_and_wins_independently():
    breaker = CircuitBreaker()
    # A severe gap-down with a perfectly normal range/ATR still triggers.
    trigger = breaker.check(prev_close=100.0, today_open=90.0, today_high=90.5, today_low=89.5, atr=50.0)
    assert trigger.triggered
    assert "gap-down" in trigger.reason


def test_custom_limits_change_the_threshold():
    strict = CircuitBreaker(CircuitBreakerLimits(gap_down_threshold=-0.01))
    trigger = strict.check(prev_close=100.0, today_open=98.5, today_high=99.0, today_low=98.0, atr=None)
    assert trigger.triggered  # -1.5% clears the tightened -1% threshold
    default = CircuitBreaker()
    not_triggered = default.check(prev_close=100.0, today_open=98.5, today_high=99.0, today_low=98.0, atr=None)
    assert not not_triggered.triggered  # -1.5% does not clear the default -3% threshold


def test_check_frame_returns_one_trigger_per_row_and_never_crashes_on_short_history():
    frame = _ohlc_frame(n=5)  # far short of the 14-day ATR window
    triggers = CircuitBreaker().check_frame(frame)
    assert len(triggers) == 5
    assert all(isinstance(t, CircuitBreakerTrigger) for t in triggers)


def test_risk_manager_size_is_forced_flat_when_circuit_breaker_fires():
    """The instant override: a strongly positive score and ample confidence
    must still come back as zero size when the circuit breaker has fired."""
    manager = RiskManager(RiskLimits())
    state = RiskState()
    trigger = CircuitBreakerTrigger(True, "gap-down -10.0% at the open")
    size, notes = manager.size(score=0.8, confidence=0.9, realized_vol=0.2, state=state, circuit_trigger=trigger)
    assert size == 0.0
    assert any("circuit breaker" in n for n in notes)


def test_risk_manager_size_is_unaffected_when_circuit_breaker_has_not_fired():
    manager = RiskManager(RiskLimits())
    state = RiskState()
    trigger = CircuitBreakerTrigger(False)
    size_with_trigger, _ = manager.size(0.8, 0.9, 0.2, state, circuit_trigger=trigger)
    size_without_trigger, _ = manager.size(0.8, 0.9, 0.2, state)
    assert size_with_trigger == pytest.approx(size_without_trigger)
    assert size_with_trigger > 0.0


def test_trader_decide_threads_the_circuit_trigger_through_to_zero_size():
    trader = Trader(TraderConfig(use_debate=False), RiskManager(RiskLimits()))
    bullish = Opinion(agent="regime", stance=0.9, confidence=0.8, rationale="strong buy")
    trigger = CircuitBreakerTrigger(True, "ATR expansion 3.2x")
    decision = trader.decide(
        pd.Timestamp("2024-01-01"), "TEST", [bullish], None,
        realized_vol=0.2, state=RiskState(), circuit_trigger=trigger,
    )
    assert decision.size == 0.0
    assert decision.action == "HOLD"  # wanted to buy (score > threshold), risk layer refused


def test_flash_crash_scenario_triggers_instant_de_risking_unlike_the_regime_only_pipeline():
    """The complementary finding to KNOWN_ISSUES.md #13: RegimeAgent alone did
    NOT de-risk on a single flash-crash day (a -10% single-day shock), but the
    circuit breaker's gap-down check -- using the exact same -10% return this
    project's own stress-test scenario defines -- fires immediately."""
    from scripts.stress_test_scenarios import SCENARIOS

    flash_crash_return = SCENARIOS["Flash Crash"]["daily_returns"][0]
    assert flash_crash_return == pytest.approx(-0.10)

    prev_close = 100.0
    shocked_open = prev_close * (1.0 + flash_crash_return)
    trigger = CircuitBreaker().check(
        prev_close=prev_close, today_open=shocked_open,
        today_high=shocked_open * 1.01, today_low=shocked_open * 0.99, atr=None,
    )
    assert trigger.triggered

    manager = RiskManager(RiskLimits())
    size, notes = manager.size(score=0.6, confidence=0.7, realized_vol=0.2, state=RiskState(), circuit_trigger=trigger)
    assert size == 0.0
