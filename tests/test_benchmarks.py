"""Tests for the extended benchmark series (Nifty 50, Equal-Weight Universe,
Nifty Next 50) and their alignment onto a strategy's own dates."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.benchmarks import (
    NIFTY_NEXT_50_START,
    NIFTY_NEXT_50_TICKER,
    align_benchmark_to_dates,
    build_extended_benchmarks,
    equal_weight_universe_returns,
    nifty_50_returns,
    nifty_next_50_returns,
)


def test_all_three_benchmarks_are_present():
    benches = build_extended_benchmarks()
    assert set(benches) == {"Nifty50", "EqualWeightUniverse", "NiftyNext50"}
    for name, frame in benches.items():
        assert len(frame) > 0, f"{name} returned an empty frame"
        assert {"date", "fwd_ret"} <= set(frame.columns)


def test_nifty_next_50_never_starts_before_its_known_data_floor():
    """Yahoo has no history for this ticker before 2020-01-01 -- requesting
    an earlier start must not silently return an empty or garbage frame."""
    frame = nifty_next_50_returns(start="2015-01-01", end="2021-01-01")
    assert frame["date"].min() >= pd.Timestamp(NIFTY_NEXT_50_START)


def test_nifty_next_50_ticker_is_the_documented_yahoo_quirk():
    """Regression against silently 'fixing' what looks like a wrong ticker --
    ^NSMIDCP really is Nifty Next 50 on Yahoo despite the misleading symbol."""
    assert NIFTY_NEXT_50_TICKER == "^NSMIDCP"


def test_equal_weight_universe_is_the_mean_of_the_universe_forward_returns():
    from nse_agents.config import SETTINGS
    from nse_agents.data.prices import forward_return, load_prices

    two_symbols = SETTINGS.universe[:2]
    result = equal_weight_universe_returns(symbols=two_symbols, start="2024-01-01", end="2024-03-01")

    manual = []
    for symbol in two_symbols:
        frame = load_prices(symbol, "2024-01-01", "2024-03-01")
        frame["fwd_ret"] = forward_return(frame, horizon=1)
        manual.append(frame[["date", "fwd_ret"]].dropna())
    expected = pd.concat(manual).groupby("date")["fwd_ret"].mean()

    merged = result.set_index("date")["fwd_ret"].reindex(expected.index)
    np.testing.assert_allclose(merged.to_numpy(), expected.to_numpy(), rtol=1e-9)


def test_equal_weight_universe_uses_open_to_open_not_close_to_close():
    """The exact convention bug this module's build caught: verify the
    equal-weight series is NOT simply close-to-close pct_change."""
    from nse_agents.config import SETTINGS
    from nse_agents.data.prices import load_prices

    frame = load_prices(SETTINGS.universe[0], "2024-01-01", "2024-06-01")
    naive_close_to_close = frame["close"].pct_change().dropna()

    result = equal_weight_universe_returns(symbols=SETTINGS.universe[:1], start="2024-01-01", end="2024-06-01")
    # The open-to-open series must not be (numerically) identical to the naive
    # close-to-close series -- if it were, the convention bug would have crept
    # back in.
    common_len = min(len(result), len(naive_close_to_close))
    assert not np.allclose(
        result["fwd_ret"].to_numpy()[:common_len],
        naive_close_to_close.to_numpy()[:common_len],
        atol=1e-9,
    )


def test_align_benchmark_to_dates_fills_missing_days_with_zero():
    benchmark = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "fwd_ret": [0.01, 0.02],
    })
    dates = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04"])  # last day missing
    result = align_benchmark_to_dates(benchmark, dates)
    np.testing.assert_allclose(result, [0.01, 0.02, 0.0])


def test_align_benchmark_to_dates_preserves_order():
    benchmark = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "fwd_ret": [0.01, 0.02, 0.03],
    })
    dates = pd.DatetimeIndex(["2024-01-04", "2024-01-02", "2024-01-03"])  # deliberately out of order
    result = align_benchmark_to_dates(benchmark, dates)
    np.testing.assert_allclose(result, [0.03, 0.01, 0.02])


def test_nifty_50_returns_matches_the_configured_benchmark_ticker():
    from nse_agents.config import BENCHMARK

    frame = nifty_50_returns(start="2024-01-01", end="2024-03-01")
    assert len(frame) > 0
    assert BENCHMARK == "^NSEI"  # the ticker this function actually reads
