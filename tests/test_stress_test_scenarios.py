"""Tests for scripts/stress_test_scenarios.py.

The synthetic-panel construction is pure and cheap (no LLM calls, no training) and
is unit-tested directly, including a regression test for the real bug this script's
own build caught: comparing a shocked scenario against a fixed "today" baseline
confounds the shock with a rolling-window calendar-shift artefact. The full
regime/trader pipeline call is exercised once, end-to-end against real cached price
data, since it is cheap (pure lookups, no training or LLM calls).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import BENCHMARK, SETTINGS
from scripts.stress_test_scenarios import (
    SCENARIOS,
    _shocked_price_frame,
    _shocked_vix_frame,
    build_scenario_panels,
    run_regime_pipeline,
)


def _fake_base_frame(n_days: int = 60, start_close: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame({
        "date": dates, "open": start_close, "high": start_close * 1.001,
        "low": start_close * 0.999, "close": start_close, "volume": 1_000_000.0,
    })


def test_shocked_price_frame_compounds_the_requested_daily_returns():
    base = _fake_base_frame(n_days=10, start_close=100.0)
    shocked = _shocked_price_frame(base, [-0.10])
    assert len(shocked) == 11
    assert shocked["close"].iloc[-1] == pytest.approx(90.0, rel=1e-9)
    # The real base days must be untouched.
    pd.testing.assert_frame_equal(shocked.iloc[:10].reset_index(drop=True), base)


def test_shocked_price_frame_appends_business_days_after_the_last_real_date():
    base = _fake_base_frame(n_days=5)
    shocked = _shocked_price_frame(base, [0.05, 0.05])
    new_dates = shocked["date"].iloc[-2:]
    assert (new_dates > base["date"].iloc[-1]).all()
    assert new_dates.is_monotonic_increasing


def test_prolonged_bear_scenario_compounds_to_exactly_minus_30_percent():
    """The whole point of the geometric daily-return construction: 126 days of
    the same compounded rate must land on exactly -30%, not an approximation."""
    base = _fake_base_frame(n_days=5, start_close=100.0)
    shocked = _shocked_price_frame(base, SCENARIOS["Prolonged Bear Market"]["daily_returns"])
    final_close = shocked["close"].iloc[-1]
    assert final_close == pytest.approx(70.0, rel=1e-6)


def test_shocked_vix_frame_flat_carry_forward_when_no_level_given():
    base = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": [15, 16, 17, 18, 19.0], "high": [15, 16, 17, 18, 19.0],
        "low": [15, 16, 17, 18, 19.0], "close": [15, 16, 17, 18, 19.0], "volume": 0.0,
    })
    shocked = _shocked_vix_frame(base, n_days=3, level=None)
    assert (shocked["close"].iloc[-3:] == 19.0).all()


def test_shocked_vix_frame_pins_to_the_requested_level():
    base = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": 15.0, "high": 15.0, "low": 15.0, "close": 15.0, "volume": 0.0,
    })
    shocked = _shocked_vix_frame(base, n_days=4, level=45.0)
    assert (shocked["close"].iloc[-4:] == 45.0).all()


def test_matched_control_and_shock_land_on_the_same_calendar_date():
    """Regression for the real bug this script's own build caught: a shocked
    scenario and its zero-return control must be evaluated on the identical
    last date, or any comparison between them is confounded by which day fell
    out of the rolling relative-strength window, not by the shock itself."""
    base = _fake_base_frame(n_days=300)
    scenario_panels = build_scenario_panels(
        {BENCHMARK: base, "^INDIAVIX": base, **{sym: base for sym in SETTINGS.universe}},
        {"daily_returns": [-0.10], "vix_level": None},
    )
    control_panels = build_scenario_panels(
        {BENCHMARK: base, "^INDIAVIX": base, **{sym: base for sym in SETTINGS.universe}},
        {"daily_returns": [0.0], "vix_level": None},
    )
    assert scenario_panels[BENCHMARK]["date"].iloc[-1] == control_panels[BENCHMARK]["date"].iloc[-1]


@pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data_cache" / "prices").exists(),
    reason="requires cached price data",
)
def test_prolonged_bear_pipeline_de_risks_every_name_vs_its_matched_control():
    """End-to-end against real cached data: a sustained -30% market-wide decline
    must leave every universe name flat, unlike a single flash-crash day."""
    from scripts.stress_test_scenarios import _load_real_base

    base = _load_real_base()
    bear = SCENARIOS["Prolonged Bear Market"]
    shocked = run_regime_pipeline(build_scenario_panels(base, bear))
    control = run_regime_pipeline(build_scenario_panels(base, {"daily_returns": [0.0] * 126, "vix_level": None}))

    assert (shocked["action"] == "FLAT").all()
    assert shocked["size"].sum() == 0.0
    assert shocked["regime_stance"].mean() < control["regime_stance"].mean()
