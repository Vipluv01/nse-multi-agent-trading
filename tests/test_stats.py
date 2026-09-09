import numpy as np
import pytest

from nse_agents.backtest.stats import (
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    holm_bonferroni,
    paired_sharpe_difference,
    probabilistic_sharpe_ratio,
)


def test_holm_is_more_conservative_than_raw_p_values():
    corrected = holm_bonferroni({"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04})
    for name, info in corrected.items():
        assert info["p_holm"] >= info["p_raw"]


def test_holm_is_monotone_in_rank():
    corrected = holm_bonferroni({"a": 0.001, "b": 0.01, "c": 0.2, "d": 0.5})
    ordered = sorted(corrected.values(), key=lambda info: info["rank"])
    adjusted = [info["p_holm"] for info in ordered]
    assert adjusted == sorted(adjusted), "adjusted p-values must not decrease with rank"


def test_holm_stops_rejecting_after_the_first_failure():
    """Step-down: once one hypothesis survives, every weaker one must too."""
    corrected = holm_bonferroni({"a": 0.001, "b": 0.30, "c": 0.31})
    assert corrected["a"]["significant"]
    assert not corrected["b"]["significant"]
    assert not corrected["c"]["significant"]


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0008, 0.01, 1500)
    interval = block_bootstrap_sharpe(returns, n_boot=500)
    assert interval.low <= interval.point <= interval.high


def test_pure_noise_does_not_produce_a_significant_sharpe():
    """A zero-mean series must not be declared a winner."""
    rng = np.random.default_rng(11)
    returns = rng.normal(0.0, 0.012, 1500)
    interval = block_bootstrap_sharpe(returns, n_boot=500)
    assert not interval.excludes_zero()


def test_paired_difference_of_a_series_with_itself_is_zero():
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0005, 0.01, 800)
    interval = paired_sharpe_difference(returns, returns, n_boot=200)
    assert interval.point == pytest.approx(0.0, abs=1e-9)


def test_deflated_sharpe_penalises_a_wider_search():
    """The same track record is worth less if more configurations were tried."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.01, 1200)
    few = np.array([0.4, 0.5, 0.6])
    many = rng.normal(0.3, 0.6, 200)
    dsr_few, exp_few = deflated_sharpe_ratio(returns, few)
    dsr_many, exp_many = deflated_sharpe_ratio(returns, many)
    assert exp_many > exp_few
    assert dsr_many < dsr_few


def test_psr_falls_when_returns_are_negatively_skewed():
    """Same mean and volatility, worse tail -> less trustworthy Sharpe."""
    rng = np.random.default_rng(9)
    symmetric = rng.normal(0.0008, 0.01, 2000)
    skewed = symmetric.copy()
    skewed[:20] -= 0.06          # a fat left tail
    skewed += 0.06 * 20 / 2000   # restore the mean
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(symmetric)
