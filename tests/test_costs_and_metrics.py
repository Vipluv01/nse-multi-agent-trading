import numpy as np
import pandas as pd
import pytest

from nse_agents.backtest.engine import backtest_signals, run_backtest, signals_to_weights
from nse_agents.backtest.metrics import (
    classification_metrics,
    compute_performance,
    max_drawdown,
    sharpe_ratio,
)
from nse_agents.config import CostModel


def test_costs_are_side_asymmetric():
    """Stamp duty is buy-side only in India, so the two legs must differ."""
    costs = CostModel()
    assert costs.cost(1_000_000, "buy") > costs.cost(1_000_000, "sell")


def test_costs_scale_linearly_above_the_brokerage_cap():
    costs = CostModel(brokerage_rate=0.0)
    assert costs.cost(2_000_000, "buy") == pytest.approx(2 * costs.cost(1_000_000, "buy"))


def test_zero_turnover_is_free():
    assert CostModel().cost(0.0, "buy") == 0.0


def test_max_drawdown_matches_a_hand_computed_case():
    # +10%, then -50% -> peak 1.10, trough 0.55, drawdown -50%.
    returns = np.array([0.10, -0.50])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_sharpe_is_excess_of_the_risk_free_rate():
    """Earning exactly the risk-free rate must score zero excess Sharpe."""
    rng = np.random.default_rng(2)
    daily_rf = (1.06 ** (1 / 252)) - 1.0
    noise = rng.normal(0.0, 0.008, 4000)
    # Centre the noise exactly, so the sample mean *is* the risk-free rate and
    # the expected excess Sharpe is 0 by construction rather than in
    # expectation -- otherwise the test is asserting against sampling error.
    returns = daily_rf + noise - noise.mean()
    excess = sharpe_ratio(returns, 0.06)
    gross = sharpe_ratio(returns, 0.0)
    assert excess == pytest.approx(0.0, abs=1e-9)
    assert gross > excess, "ignoring the risk-free rate must flatter the result"


def test_sharpe_of_a_zero_variance_series_is_guarded_not_infinite():
    """A constant series has no defined Sharpe; it must not return inf or NaN."""
    returns = np.full(252, (1.06 ** (1 / 252)) - 1.0)
    assert sharpe_ratio(returns, 0.0) == 0.0
    assert sharpe_ratio(returns, 0.06) == 0.0


def test_a_high_win_rate_can_still_be_a_bad_strategy():
    """The failure mode the metrics exist to expose.

    99 small wins and one catastrophic loss: a 99% hit rate and a negative
    return. Any report that quotes hit rate alone would call this a success.
    """
    returns = np.array([0.001] * 99 + [-0.5])
    perf = compute_performance(returns)
    assert perf.hit_rate == pytest.approx(0.99)
    assert perf.max_drawdown < -0.4
    assert np.prod(1 + returns) < 1.0


def test_classification_metrics_on_a_perfect_predictor():
    y = np.array([0, 1, 0, 1])
    prob = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = classification_metrics(y, prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["mcc"] == pytest.approx(1.0)


def _toy_signals(n_days=6, symbols=("A", "B")):
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for date in dates:
        for symbol in symbols:
            rows.append({"date": date, "symbol": symbol, "prob_up": 0.9, "fwd_ret": 0.01})
    return pd.DataFrame(rows)


def test_weights_respect_the_per_name_cap():
    signals = _toy_signals(symbols=tuple("ABCDEFGH"))
    weighted = signals_to_weights(signals, threshold=0.5, max_weight=0.20)
    assert weighted["weight"].max() <= 0.20 + 1e-12
    per_day = weighted.groupby("date")["weight"].sum()
    assert (per_day <= 1.0 + 1e-9).all()


def test_costs_reduce_returns_and_are_charged_only_on_turnover():
    """A position opened once and held must be charged once, not daily."""
    signals = _toy_signals()
    result = backtest_signals(signals, "held")
    # Turnover on day 1 (entering), then nothing while the weights are static.
    assert result.daily["turnover"].iloc[0] > 0
    assert result.daily["turnover"].iloc[1:].sum() == pytest.approx(0.0, abs=1e-12)
    assert result.daily["cost"].iloc[1:].sum() == pytest.approx(0.0, abs=1e-12)
    assert (result.daily["net_return"] <= result.daily["gross_return"] + 1e-12).all()


def test_idle_cash_earns_the_risk_free_rate():
    """A fully flat book must not be scored as a zero return."""
    signals = _toy_signals()
    signals["prob_up"] = 0.1  # nothing clears the threshold
    result = backtest_signals(signals, "flat")
    assert result.daily["exposure"].sum() == pytest.approx(0.0)
    assert (result.daily["gross_return"] > 0).all()


def test_gross_exposure_cap_prevents_unfunded_leverage():
    """Ten names inside a 20% per-name cap is a 200% book unless capped."""
    from nse_agents.agents.risk import RiskLimits, RiskManager

    manager = RiskManager(RiskLimits(max_gross_exposure=1.0, max_weight_per_name=0.20))
    sized = [(0.20, []) for _ in range(10)]
    capped = manager.apply_gross_cap(sized)
    assert sum(w for w, _ in capped) == pytest.approx(1.0)
    # Relative sizing must be preserved, not truncated.
    assert len({round(w, 9) for w, _ in capped}) == 1
    # And the scaling must be recorded, not silent.
    assert any("gross exposure" in n for _, notes in capped for n in notes)


def test_gross_cap_is_a_no_op_below_the_limit():
    from nse_agents.agents.risk import RiskLimits, RiskManager

    manager = RiskManager(RiskLimits(max_gross_exposure=1.0))
    sized = [(0.2, []), (0.3, [])]
    assert manager.apply_gross_cap(sized) == sized


def test_annualisation_respects_the_rebalancing_frequency():
    """Annualising 20-day returns with sqrt(252) inflates Sharpe by ~4.5x."""
    rng = np.random.default_rng(4)
    periodic = rng.normal(0.004, 0.03, 400)          # 400 non-overlapping 20-day periods
    wrong = sharpe_ratio(periodic, 0.0, periods_per_year=252)
    right = sharpe_ratio(periodic, 0.0, periods_per_year=252 / 20)
    assert wrong > right
    assert wrong / right == pytest.approx(np.sqrt(20), rel=1e-6)


def test_cagr_uses_the_right_number_of_years():
    """252 weekly bars is ~5 years, not 1 -- CAGR must not be inflated 5x."""
    returns = np.full(252, 0.002)
    fast = compute_performance(returns, periods_per_year=252)
    slow = compute_performance(returns, periods_per_year=252 / 5)
    assert fast.cagr > slow.cagr
    assert slow.n_days == fast.n_days == 252
