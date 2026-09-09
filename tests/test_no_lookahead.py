"""The tests that matter most: nothing may see the future.

Every other result in this project is void if one of these fails, so they are
written as active attacks -- perturb the future, and assert the past does not
move -- rather than as assertions about shapes.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from nse_agents.config import WalkForward
from nse_agents.data.features import FEATURE_COLUMNS, build_features
from nse_agents.data.news import actionable_date
from nse_agents.data.prices import forward_return
from nse_agents.models.train import walk_forward_folds


def _synthetic_prices(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=n),
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - abs(rng.normal(0, 0.006, n))),
            "close": close,
            "volume": rng.integers(1e5, 1e7, n).astype(float),
        }
    )


def test_features_are_causal():
    """Perturbing the final bar must not change any earlier feature value.

    This catches centred rolling windows, backfills and whole-series scaling --
    the three ways a feature silently acquires future information.
    """
    base = _synthetic_prices()
    tampered = base.copy()
    tampered.loc[tampered.index[-1], ["open", "high", "low", "close"]] *= 1.5
    tampered.loc[tampered.index[-1], "volume"] *= 20

    left = build_features(base)[list(FEATURE_COLUMNS)].to_numpy()
    right = build_features(tampered)[list(FEATURE_COLUMNS)].to_numpy()

    # Compare everything except the perturbed final row.
    np.testing.assert_allclose(left[:-1], right[:-1], rtol=1e-9, atol=1e-12, equal_nan=True)


def test_features_do_not_use_same_bar_close_for_future_bars():
    """A mid-series perturbation may only affect that bar onward, never before."""
    base = _synthetic_prices()
    idx = 200
    tampered = base.copy()
    tampered.loc[tampered.index[idx], "close"] *= 1.3

    left = build_features(base)[list(FEATURE_COLUMNS)].to_numpy()
    right = build_features(tampered)[list(FEATURE_COLUMNS)].to_numpy()
    np.testing.assert_allclose(left[:idx], right[:idx], rtol=1e-9, atol=1e-12, equal_nan=True)


def test_forward_return_is_open_to_open_and_shifted():
    """fwd_ret at row t must equal open[t+2]/open[t+1]-1 for horizon 1."""
    frame = _synthetic_prices(50)
    fwd = forward_return(frame, horizon=1).to_numpy()
    opens = frame["open"].to_numpy()
    for t in range(len(frame) - 2):
        assert fwd[t] == pytest.approx(opens[t + 2] / opens[t + 1] - 1.0)
    # The final rows cannot be known and must be NaN, not zero.
    assert np.isnan(fwd[-1])


def test_forward_return_never_uses_the_decision_bar():
    """Changing the close of the decision bar must not move its own label."""
    frame = _synthetic_prices(50)
    before = forward_return(frame, 1).to_numpy()
    tampered = frame.copy()
    tampered.loc[tampered.index[10], "close"] *= 2.0
    after = forward_return(tampered, 1).to_numpy()
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_news_after_close_moves_to_next_day():
    """A headline published after 15:30 IST is not knowable at that close."""
    assert actionable_date(datetime(2024, 5, 6, 9, 59)) == "2024-05-06"
    assert actionable_date(datetime(2024, 5, 6, 10, 0)) == "2024-05-07"
    assert actionable_date(datetime(2024, 5, 6, 23, 30)) == "2024-05-07"


def test_walk_forward_folds_are_ordered_and_embargoed():
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2026-01-01"))
    config = WalkForward(train_years=3.0, test_months=6, embargo_days=10)
    folds = walk_forward_folds(dates, config)
    assert folds, "expected at least one fold"
    for fold in folds:
        # No training row may fall on or after the test window's start.
        assert dates[fold.train_idx].max() < fold.test_start
        # The embargo gap must actually be present.
        gap = (fold.test_start - dates[fold.train_idx].max()).days
        assert gap >= config.embargo_days, f"embargo gap only {gap} days"
        # Train and test indices must be disjoint.
        assert not set(fold.train_idx) & set(fold.test_idx)


def test_walk_forward_test_windows_do_not_overlap():
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", "2026-01-01"))
    folds = walk_forward_folds(dates)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_end <= later.test_start


def test_weekend_headlines_roll_forward_not_backward():
    """A Saturday headline must inform Monday, and must not be discarded."""
    from nse_agents.data.news import align_to_trading_days

    sessions = pd.DatetimeIndex(["2024-05-03", "2024-05-06", "2024-05-07"])  # Fri, Mon, Tue
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-05-03", "2024-05-04", "2024-05-05", "2024-05-06"]),
         "title": ["fri", "sat", "sun", "mon"]}
    )
    rolled = align_to_trading_days(frame, sessions)
    mapping = dict(zip(rolled["title"], rolled["date"]))
    assert mapping["fri"] == pd.Timestamp("2024-05-03")
    assert mapping["sat"] == pd.Timestamp("2024-05-06")   # forward, not back to Friday
    assert mapping["sun"] == pd.Timestamp("2024-05-06")
    assert mapping["mon"] == pd.Timestamp("2024-05-06")
    assert len(rolled) == 4, "no headline may be silently dropped"


def test_headlines_after_the_last_session_are_dropped_not_backfilled():
    """There is no future session to act on; they must not fall back onto the last one."""
    from nse_agents.data.news import align_to_trading_days

    sessions = pd.DatetimeIndex(["2024-05-03", "2024-05-06"])
    frame = pd.DataFrame({"date": pd.to_datetime(["2024-05-06", "2024-05-09"]), "title": ["a", "b"]})
    rolled = align_to_trading_days(frame, sessions)
    assert list(rolled["title"]) == ["a"]


def test_alignment_returns_a_clean_index_so_callers_cannot_compare_row_wise():
    """Regression: the returned frame is shorter and re-indexed.

    A caller that diffs it element-wise against the input raises
    "Can only compare identically-labeled Series objects" -- which is exactly
    how the sentiment stage failed. Counting must be set-wise.
    """
    from nse_agents.data.news import align_to_trading_days

    sessions = pd.DatetimeIndex(["2024-05-03", "2024-05-06"])
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2024-05-04", "2024-05-06", "2024-05-09"]),
         "title": ["sat", "mon", "past-the-end"]}
    )
    aligned = align_to_trading_days(frame, sessions)
    assert len(aligned) == 2 < len(frame)
    assert list(aligned.index) == [0, 1], "index must be a clean RangeIndex"

    # The set-wise count the scripts use must be correct.
    session_set = set(sessions)
    off_session = int((~frame["date"].isin(session_set)).sum())   # sat + past-the-end
    dropped = len(frame) - len(aligned)                            # past-the-end
    assert off_session - dropped == 1                              # only sat moved
