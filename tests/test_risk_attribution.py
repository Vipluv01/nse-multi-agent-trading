"""Tests for the benchmark-relative risk-attribution metrics (Treynor, Information
Ratio, upside/downside capture, beta)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.metrics import (
    beta,
    downside_capture_ratio,
    information_ratio,
    treynor_ratio,
    upside_capture_ratio,
)


def test_beta_of_a_series_against_itself_is_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.012, 500)
    assert beta(returns, returns) == pytest.approx(1.0, abs=1e-9)


def test_beta_of_a_zero_variance_benchmark_is_nan_not_infinite():
    returns = np.array([0.01, 0.02, -0.01, 0.005])
    flat_benchmark = np.zeros(4)
    result = beta(returns, flat_benchmark)
    assert np.isnan(result)


def test_beta_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        beta(np.array([0.01, 0.02]), np.array([0.01, 0.02, 0.03]))


def test_upside_and_downside_capture_of_self_are_both_one():
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0005, 0.012, 500)
    assert upside_capture_ratio(returns, returns) == pytest.approx(1.0, abs=1e-9)
    assert downside_capture_ratio(returns, returns) == pytest.approx(1.0, abs=1e-9)


def test_upside_capture_below_one_means_giving_up_market_upside():
    """A strategy that only captures half the benchmark's up-day moves."""
    rng = np.random.default_rng(2)
    benchmark = rng.normal(0.0, 0.012, 1000)
    portfolio = np.where(benchmark > 0, benchmark * 0.5, benchmark)
    result = upside_capture_ratio(portfolio, benchmark)
    assert result == pytest.approx(0.5, abs=0.05)


def test_downside_capture_below_one_means_real_crash_protection():
    """Mirrors the actual mechanism found in scripts/regime_analysis.py: losing
    less than the benchmark on down days is real downside protection."""
    rng = np.random.default_rng(3)
    benchmark = rng.normal(0.0, 0.012, 1000)
    portfolio = np.where(benchmark < 0, benchmark * 0.2, benchmark)  # protected on down days
    result = downside_capture_ratio(portfolio, benchmark)
    assert result == pytest.approx(0.2, abs=0.05)
    assert result < 1.0


def test_capture_ratios_are_nan_when_benchmark_never_moves_that_direction():
    portfolio = np.array([0.01, 0.02, 0.005])
    all_up_benchmark = np.array([0.01, 0.02, 0.005])  # never negative
    assert np.isnan(downside_capture_ratio(portfolio, all_up_benchmark))


def test_information_ratio_is_zero_when_tracking_the_benchmark_exactly():
    rng = np.random.default_rng(4)
    benchmark = rng.normal(0.0005, 0.012, 500)
    assert information_ratio(benchmark, benchmark) == pytest.approx(0.0, abs=1e-9)


def test_information_ratio_is_positive_when_consistently_beating_the_benchmark():
    """Uses a portfolio that outperforms the benchmark on 70% of days (real
    variation in the active return, not a constant), so tracking error is
    genuinely nonzero and the sign of the mean active return is what's being
    tested -- a purely constant offset is covered by its own dedicated test
    below, since it hits a different, near-zero-tracking-error code path."""
    rng = np.random.default_rng(5)
    benchmark = rng.normal(0.0002, 0.012, 2000)
    noise = rng.normal(0.0003, 0.0005, 2000)  # a real edge with real day-to-day variation
    portfolio = benchmark + noise
    assert information_ratio(portfolio, benchmark) > 0


def test_information_ratio_is_nan_for_a_constant_riskless_edge():
    """A portfolio that beats the benchmark by an identical amount every
    single day has ~zero tracking error but a nonzero active return -- an
    ill-defined ratio (dividing a real number by ~0), not a genuine zero.
    Regression for a real bug this module's own tests caught: the first cut
    of information_ratio special-cased near-zero tracking error to always
    return 0.0, silently reporting 'no edge' for what is actually an
    undefined, not-really-zero case.
    """
    rng = np.random.default_rng(5)
    benchmark = rng.normal(0.0002, 0.012, 1000)
    portfolio = benchmark + 0.0003  # identical constant offset every day
    assert np.isnan(information_ratio(portfolio, benchmark))


def test_treynor_ratio_is_nan_when_beta_is_near_zero():
    """Dividing by a near-zero beta would produce an arbitrarily large,
    meaningless ratio -- it must come back as NaN, not a huge number.

    Two independent random draws do not reliably have near-zero *empirical*
    covariance at a finite sample size (sampling noise alone can easily
    produce an empirical beta well away from the true population value of 0
    -- this test originally relied on that and was flaky). Residualising one
    series against the other guarantees exactly zero sample covariance by
    construction, regardless of sample size or seed.
    """
    rng = np.random.default_rng(6)
    benchmark = rng.normal(0.0, 0.012, 500)
    raw = rng.normal(0.0003, 0.008, 500)
    slope = np.cov(raw, benchmark, ddof=1)[0, 1] / np.var(benchmark, ddof=1)
    market_neutral = raw - slope * (benchmark - benchmark.mean())  # exactly zero beta now
    result = treynor_ratio(market_neutral, benchmark)
    assert np.isnan(result)


def test_treynor_ratio_positive_for_a_genuinely_outperforming_beta_one_portfolio():
    """A constant offset added to the benchmark guarantees beta exactly 1.0
    by construction (Cov(b+k, b) = Var(b)), but the *sign* of the annualised
    excess return still depends on the random draw's own sample mean, which
    a single finite sample is not guaranteed to land on the right side of
    (this failed under the original seed for exactly that reason -- a
    real, if unlucky, sampling-noise failure, not a function bug). Forcing
    the benchmark's sample mean to a known, small positive value removes
    that remaining source of flakiness.
    """
    rng = np.random.default_rng(7)
    raw_benchmark = rng.normal(0.0, 0.012, 1000)
    benchmark = raw_benchmark - raw_benchmark.mean() + 0.0002  # exact sample mean by construction
    portfolio = benchmark + 0.0004  # beta exactly 1.0, consistent excess
    result = treynor_ratio(portfolio, benchmark, risk_free_annual=0.0)
    assert result > 0


def test_nan_rows_are_dropped_from_both_series_together():
    """A NaN in either series on a given day must drop that day from both,
    not just one -- otherwise the two series silently desynchronise."""
    returns = np.array([0.01, np.nan, 0.02, 0.015])
    benchmark = np.array([0.008, 0.01, np.nan, 0.012])
    # Only index 0 and 3 are NaN-free in both.
    result = beta(returns, benchmark)
    assert np.isfinite(result)  # must not raise or silently include the NaN rows


def test_real_open_to_open_alignment_gives_a_plausible_beta_for_an_nse_portfolio():
    """Regression for the exact bug this module's build caught: a Buy&Hold
    portfolio of large-cap NSE stocks benchmarked against the Nifty, using the
    correct open-to-open convention, must show a beta near 1.0 (they share
    most of their constituents) -- not near zero, which is what a
    close-to-close-vs-open-to-open timing mismatch silently produced.
    """
    from nse_agents.backtest.baselines import build_baseline_signals
    from nse_agents.backtest.engine import backtest_signals
    from nse_agents.config import BENCHMARK, SETTINGS
    from nse_agents.data.prices import forward_return, load_prices

    bh = backtest_signals(build_baseline_signals()["Buy&Hold"], "BH", threshold=0.5)
    nifty = load_prices(BENCHMARK, SETTINGS.start, SETTINGS.end)
    nifty["fwd"] = forward_return(nifty, horizon=1)
    bench = nifty.set_index("date").reindex(bh.dates)["fwd"].fillna(0.0).to_numpy()

    result = beta(bh.returns, bench)
    assert 0.7 < result < 1.3, f"expected beta near 1.0 for an NSE large-cap portfolio, got {result}"
