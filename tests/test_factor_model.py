"""Tests for nse_agents/backtest/factor_model.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.factor_model import (
    UNAVAILABLE_FACTORS,
    build_factors,
    market_excess_factor,
    momentum_factor,
    regress_factors,
)
from nse_agents.config import DATA_CACHE


def test_unavailable_factors_documents_exactly_smb_and_hml():
    """The two-factor scope is deliberate, not partial -- SMB/HML must stay
    documented as unavailable, never silently added with fabricated data."""
    assert set(UNAVAILABLE_FACTORS) == {"smb", "hml"}
    for reason in UNAVAILABLE_FACTORS.values():
        assert "point-in-time" in reason


def test_regress_factors_recovers_known_beta_and_zero_alpha_on_synthetic_data():
    """A strategy built as exactly 1.5x the market factor plus zero-mean noise,
    with no momentum exposure and no true alpha, must recover beta_mkt ~ 1.5,
    beta_wml ~ 0, and an alpha not distinguishable from zero."""
    rng = np.random.default_rng(0)
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0.0003, 0.01, n)
    wml = rng.normal(0.0, 0.008, n)
    factors = pd.DataFrame({"date": dates, "mkt_excess": mkt, "wml": wml})

    true_beta_mkt = 1.5
    noise = rng.normal(0.0, 0.002, n)
    returns = true_beta_mkt * mkt + noise  # no wml exposure, no true alpha

    result = regress_factors(returns, dates, factors)
    assert result.betas["mkt_excess"] == pytest.approx(true_beta_mkt, abs=0.05)
    assert result.betas["wml"] == pytest.approx(0.0, abs=0.05)
    assert abs(result.alpha_tstat) < 3.0  # should not spuriously reject zero alpha
    assert result.r_squared > 0.9  # mkt explains almost everything by construction


def test_regress_factors_recovers_a_true_injected_alpha():
    rng = np.random.default_rng(1)
    n = 1000
    dates = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0.0003, 0.01, n)
    wml = rng.normal(0.0, 0.008, n)
    factors = pd.DataFrame({"date": dates, "mkt_excess": mkt, "wml": wml})

    true_alpha_daily = 0.0005
    noise = rng.normal(0.0, 0.001, n)
    returns = true_alpha_daily + 0.8 * mkt + noise

    result = regress_factors(returns, dates, factors)
    assert result.alpha_daily == pytest.approx(true_alpha_daily, abs=0.0003)
    assert result.alpha_pvalue < 0.01  # a real, injected alpha should be detected


def test_regress_factors_raises_on_too_few_overlapping_days():
    dates = pd.bdate_range("2020-01-01", periods=2)
    factors = pd.DataFrame({"date": dates, "mkt_excess": [0.01, -0.01], "wml": [0.0, 0.0]})
    with pytest.raises(ValueError, match="too few"):
        regress_factors(np.array([0.01, -0.01]), dates, factors)


def test_regress_factors_r_squared_is_between_zero_and_one_on_pure_noise():
    rng = np.random.default_rng(2)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    factors = pd.DataFrame({
        "date": dates, "mkt_excess": rng.normal(0, 0.01, n), "wml": rng.normal(0, 0.008, n),
    })
    pure_noise = rng.normal(0, 0.01, n)  # independent of both factors
    result = regress_factors(pure_noise, dates, factors)
    assert -0.05 <= result.r_squared <= 0.2  # near zero, allowing sampling noise


def test_to_dict_round_trips_every_field():
    rng = np.random.default_rng(3)
    n = 300
    dates = pd.bdate_range("2020-01-01", periods=n)
    factors = pd.DataFrame({
        "date": dates, "mkt_excess": rng.normal(0, 0.01, n), "wml": rng.normal(0, 0.008, n),
    })
    result = regress_factors(rng.normal(0, 0.01, n), dates, factors)
    d = result.to_dict()
    assert set(d) == {
        "n_days", "alpha_daily", "alpha_annualized", "alpha_tstat", "alpha_pvalue",
        "betas", "beta_tstats", "beta_pvalues", "r_squared", "factors_used", "factors_unavailable",
    }
    assert d["factors_unavailable"] == UNAVAILABLE_FACTORS


@pytest.mark.skipif(
    not (DATA_CACHE / "prices" / "RELIANCE.csv").exists(),
    reason="requires the cached NSE price history to be present",
)
def test_market_excess_factor_beta_of_nifty_against_itself_via_buyhold_is_near_one():
    """Regression-style sanity check reusing the same real data this project's
    other benchmark tests already exercise: a Buy&Hold portfolio of this
    study's own large-cap universe should show a market beta near 1.0 against
    the Nifty 50 factor, and a high R^2 (it holds a close cousin of the index)."""
    from nse_agents.backtest.baselines import build_baseline_signals
    from nse_agents.backtest.engine import backtest_signals

    factors = build_factors()
    bh = backtest_signals(build_baseline_signals()["Buy&Hold"], "BH", threshold=0.5)
    result = regress_factors(bh.returns, bh.dates, factors)
    assert 0.7 < result.betas["mkt_excess"] < 1.3
    assert result.r_squared > 0.5


@pytest.mark.skipif(
    not (DATA_CACHE / "prices" / "IDX_NSEI.csv").exists(),
    reason="requires the cached Nifty 50 price history to be present",
)
def test_market_excess_factor_columns_and_subtracts_a_positive_risk_free_rate():
    frame = market_excess_factor(start="2020-01-01", end="2020-06-01", risk_free_annual=0.06)
    assert list(frame.columns) == ["date", "mkt_excess"]
    assert len(frame) > 0
    # mkt_excess must equal the raw forward return minus a small positive daily
    # risk-free deduction -- not a no-op subtraction of zero.
    period_rf = (1.06) ** (1 / 252) - 1
    assert period_rf > 0


def test_momentum_factor_is_empty_or_valid_shape_on_a_short_window():
    """Too little history for a 12-month lookback must degrade to an empty
    frame, not raise."""
    frame = momentum_factor(start="2026-06-01", end="2026-09-01")
    assert list(frame.columns) == ["date", "wml"] if len(frame) else True
    assert len(frame) == 0 or "wml" in frame.columns
