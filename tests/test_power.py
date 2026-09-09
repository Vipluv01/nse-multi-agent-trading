"""Tests for the power analysis -- the properties a correct implementation must have,
not just that it runs.
"""

import numpy as np
import pytest

from nse_agents.backtest.power import PowerCurve, analytic_se_approximation, simulate_detection_power


@pytest.fixture(scope="module")
def reference():
    rng = np.random.default_rng(1)
    return rng.normal(0.0004, 0.011, 400)


def test_power_at_zero_is_near_the_significance_level(reference):
    """Under the null (no real edge), the test should false-positive at roughly alpha."""
    curve = simulate_detection_power(
        reference, target_sharpes=np.array([0.0]), n_periods=400,
        n_sims=300, n_boot=300, block=15, alpha=0.05,
    )
    # Block bootstrap CIs run conservative at small samples; this should be at or
    # below alpha, and nowhere near it being large (which would mean the test
    # over-detects noise as a real edge -- exactly the failure mode this whole
    # project exists to avoid).
    assert curve.power[0] <= 0.10


def test_power_increases_monotonically_with_true_sharpe(reference):
    targets = np.array([0.0, 0.5, 1.0, 1.5, 2.5])
    curve = simulate_detection_power(
        reference, target_sharpes=targets, n_periods=400,
        n_sims=150, n_boot=300, block=15,
    )
    assert np.all(np.diff(curve.power) >= -0.05), "power must not meaningfully decrease with a larger true effect"
    assert curve.power[-1] > curve.power[0]


def test_more_periods_lowers_the_minimum_detectable_effect(reference):
    """More data must make small edges easier, not harder, to detect."""
    targets = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    small_n = simulate_detection_power(reference, targets, n_periods=100, n_sims=150, n_boot=300, block=10)
    large_n = simulate_detection_power(reference, targets, n_periods=2000, n_sims=150, n_boot=300, block=10)
    mde_small = small_n.minimum_detectable_effect(0.80)
    mde_large = large_n.minimum_detectable_effect(0.80)
    assert mde_large < mde_small


def test_minimum_detectable_effect_returns_inf_when_grid_is_too_narrow():
    curve = PowerCurve(
        target_sharpes=np.array([0.0, 0.2]), power=np.array([0.02, 0.10]),
        n_periods=100, periods_per_year=252, n_sims=100,
    )
    assert curve.minimum_detectable_effect(0.80) == float("inf")


def test_minimum_detectable_effect_interpolates_correctly():
    curve = PowerCurve(
        target_sharpes=np.array([0.0, 1.0, 2.0]), power=np.array([0.05, 0.80, 1.0]),
        n_periods=100, periods_per_year=252, n_sims=100,
    )
    assert curve.minimum_detectable_effect(0.80) == pytest.approx(1.0)


def test_analytic_approximation_shrinks_with_more_years():
    short = analytic_se_approximation(252, 252)     # 1 year
    long = analytic_se_approximation(252 * 10, 252)  # 10 years
    assert long < short
    assert long == pytest.approx(short / np.sqrt(10), rel=1e-6)


def test_injected_mean_actually_controls_the_realised_sharpe(reference):
    """The mechanism check: a huge target Sharpe must produce huge realised Sharpes,
    not silently be ignored (a bug that would make every power number meaningless)."""
    from nse_agents.backtest.metrics import sharpe_ratio
    from nse_agents.backtest.stats import _block_indices

    residual = reference - reference.mean()
    period_vol = residual.std(ddof=1)
    period_rf = 0.0
    target = 3.0
    period_mean = period_rf + target * period_vol / np.sqrt(252)

    rng = np.random.default_rng(2)
    idx = _block_indices(2000, 15, rng)
    synthetic = period_mean + residual[idx % len(residual)]
    realised = sharpe_ratio(synthetic, 0.0, periods_per_year=252)
    assert realised == pytest.approx(target, abs=0.5)
