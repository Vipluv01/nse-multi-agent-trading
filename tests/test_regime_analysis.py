"""Tests for the regime-stability check.

The one property that matters most: the regime label for day t must depend only on
prices at or before t. A regime classifier with any lookahead would let "was this a
crash" be informed by what happened afterward -- silently answering a different,
easier question than the one this diagnostic claims to answer.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.regime_analysis import classify_regimes


def test_regime_label_is_unaffected_by_future_prices(monkeypatch):
    """Perturbing the last 30 days must not change any earlier day's regime label."""
    import scripts.regime_analysis as mod

    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-01", periods=500)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500)))
    base = pd.DataFrame({"date": dates, "close": close})

    def fake_loader(symbol, start, end):
        return base.copy()

    monkeypatch.setattr(mod, "load_prices", fake_loader)
    before = classify_regimes(start=str(dates[100].date()))

    tampered = base.copy()
    tampered.loc[tampered.index[-30:], "close"] *= 1.6  # crash the last month upward
    monkeypatch.setattr(mod, "load_prices", lambda *a, **kw: tampered.copy())
    after = classify_regimes(start=str(dates[100].date()))

    # Every label before the tampered window must be identical.
    cutoff = dates[-31]
    early_before = before.loc[before["date"] < cutoff, "regime"]
    early_after = after.loc[after["date"] < cutoff, "regime"]
    pd.testing.assert_series_equal(early_before.reset_index(drop=True),
                                    early_after.reset_index(drop=True))


def test_regime_labels_are_exhaustive_and_mutually_exclusive():
    """Every day gets exactly one of the three labels."""
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2019-01-01", periods=400)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, 400)))

    import scripts.regime_analysis as mod
    frame = pd.DataFrame({"date": dates, "close": close})
    orig = mod.load_prices
    mod.load_prices = lambda *a, **kw: frame.copy()
    try:
        result = classify_regimes(start=str(dates[0].date()))
    finally:
        mod.load_prices = orig

    assert set(result["regime"].unique()) <= {"crash", "bull", "choppy"}
    assert result["regime"].notna().all()


def test_a_real_crash_is_labelled_crash():
    """A sharp, sustained drop must be classified as a crash, not missed."""
    import scripts.regime_analysis as mod

    dates = pd.bdate_range("2019-01-01", periods=200)
    close = np.concatenate([
        np.full(100, 100.0),
        np.linspace(100.0, 60.0, 100),  # a real -40% drop over 100 sessions
    ])
    frame = pd.DataFrame({"date": dates, "close": close})
    orig = mod.load_prices
    mod.load_prices = lambda *a, **kw: frame.copy()
    try:
        result = classify_regimes(start=str(dates[0].date()))
    finally:
        mod.load_prices = orig

    tail = result.iloc[-20:]
    assert (tail["regime"] == "crash").any(), "a sustained -40% drop must trigger the crash label"


def test_a_calm_rising_market_is_not_labelled_crash():
    import scripts.regime_analysis as mod

    dates = pd.bdate_range("2019-01-01", periods=200)
    close = 100 * (1.0003 ** np.arange(200))  # smooth, steady rise
    frame = pd.DataFrame({"date": dates, "close": close})
    orig = mod.load_prices
    mod.load_prices = lambda *a, **kw: frame.copy()
    try:
        result = classify_regimes(start=str(dates[0].date()))
    finally:
        mod.load_prices = orig

    assert not (result["regime"] == "crash").any()
    assert (result["regime"] == "bull").mean() > 0.5
