"""Tests for the dashboard's pure data functions -- no streamlit import here or in
the module under test, so these run as plain pytest with no dashboard session needed.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.dashboard_data import (
    backtest_ablation_table,
    compute_overview_stats,
    load_account_state,
    macro_regime_indicator,
    positions_table,
    sentiment_distribution,
    trades_table,
)
from nse_agents.live.mock_broker import MockBroker
from nse_agents.live.state_store import PaperTradingStore


@pytest.fixture
def store_with_trades(tmp_path):
    store = PaperTradingStore(tmp_path / "state.sqlite", initial_capital=1_000_000.0)
    broker = MockBroker()
    from nse_agents.live.engine import rebalance

    rebalance(store, broker, {"RELIANCE": 0.2, "TCS": 0.2}, pd.Timestamp("2026-08-28"))
    rebalance(store, broker, {"RELIANCE": 0.1, "TCS": 0.25}, pd.Timestamp("2026-08-31"))
    return store


def test_load_account_state_returns_a_working_store(tmp_path):
    path = tmp_path / "state.sqlite"
    PaperTradingStore(path, initial_capital=500_000.0)
    store = load_account_state(path)
    assert store.snapshot().initial_capital == pytest.approx(500_000.0)


def test_overview_stats_reflect_live_prices_not_just_recorded_snapshots(store_with_trades):
    broker = MockBroker()
    prices = {"RELIANCE": broker.get_quote("RELIANCE", pd.Timestamp("2026-08-31")),
              "TCS": broker.get_quote("TCS", pd.Timestamp("2026-08-31"))}
    stats = compute_overview_stats(store_with_trades, prices)
    assert stats.total_equity > 0
    assert stats.days_tracked == 2


def test_overview_stats_withhold_sharpe_below_the_floor(store_with_trades):
    prices = {"RELIANCE": 1300.0, "TCS": 2400.0}
    stats = compute_overview_stats(store_with_trades, prices)
    assert stats.sharpe is None  # only 2 days tracked, far below the 60-day floor
    assert stats.buyhold_sharpe is None


def test_overview_stats_drawdown_at_least_as_large_as_return(store_with_trades):
    """Same invariant KNOWN_ISSUES.md #10's regression test checks: |return|
    can never exceed |drawdown| from the same starting capital."""
    prices = {"RELIANCE": 1100.0, "TCS": 2000.0}  # a real drop from cost basis
    stats = compute_overview_stats(store_with_trades, prices)
    assert abs(stats.max_drawdown) >= abs(stats.total_return) - 1e-9


def test_positions_table_has_expected_columns_and_rows(store_with_trades):
    prices = {"RELIANCE": 1300.0, "TCS": 2400.0}
    table = positions_table(store_with_trades, prices)
    assert list(table.columns) == ["symbol", "quantity", "avg_price", "market_value", "return"]
    assert set(table["symbol"]) == {"RELIANCE", "TCS"}


def test_positions_table_handles_a_missing_price_gracefully(store_with_trades):
    table = positions_table(store_with_trades, {})  # no prices available at all
    assert len(table) == 2
    assert table["market_value"].isna().all()


def test_positions_table_empty_account_returns_empty_frame_with_right_columns(tmp_path):
    store = PaperTradingStore(tmp_path / "state.sqlite", initial_capital=1_000_000.0)
    table = positions_table(store, {})
    assert len(table) == 0
    assert list(table.columns) == ["symbol", "quantity", "avg_price", "market_value", "return"]


def test_trades_table_returns_every_recorded_fill(store_with_trades):
    table = trades_table(store_with_trades)
    assert len(table) >= 2  # at least the two opening buys
    assert "symbol" in table.columns and "side" in table.columns


def test_sentiment_distribution_buckets_into_five_named_categories():
    frame = pd.DataFrame({"sentiment": [-0.9, -0.2, 0.0, 0.2, 0.9]})
    result = sentiment_distribution(frame)
    assert set(result["bucket"]) == {
        "very negative", "negative", "neutral", "positive", "very positive"
    }
    assert result["count"].sum() == 5


def test_sentiment_distribution_handles_an_empty_frame():
    result = sentiment_distribution(pd.DataFrame(columns=["sentiment"]))
    assert len(result) == 0


def test_macro_regime_indicator_returns_the_latest_row_for_the_symbol():
    table = pd.DataFrame({
        "symbol": ["RELIANCE", "RELIANCE", "TCS"],
        "date": pd.to_datetime(["2026-09-01", "2026-09-02", "2026-09-02"]),
        "vix": [14.0, 15.0, 15.0],
        "vix_percentile": [0.3, 0.4, 0.4],
        "nifty_mom_20d": [0.01, 0.02, 0.02],
    })
    result = macro_regime_indicator(table, "RELIANCE")
    assert result["vix"] == 15.0  # the later of the two RELIANCE rows, not TCS's


def test_macro_regime_indicator_handles_a_symbol_with_no_rows():
    table = pd.DataFrame({"symbol": ["TCS"], "date": pd.to_datetime(["2026-09-01"]),
                          "vix": [15.0], "vix_percentile": [0.4], "nifty_mom_20d": [0.02]})
    result = macro_regime_indicator(table, "RELIANCE")  # not in the table
    assert result["vix"] is None


def test_backtest_ablation_table_returns_empty_frame_when_summary_is_missing(tmp_path):
    result = backtest_ablation_table(tmp_path)  # no agents/summary.csv under tmp_path
    assert result.empty


def test_backtest_ablation_table_reads_the_real_cached_summary_unmodified(tmp_path):
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()
    summary = pd.DataFrame({"strategy": ["Buy&Hold", "Full+Debate"], "Sharpe(net,excess)": [0.58, 0.13]})
    summary.to_csv(agents_dir / "summary.csv", index=False)

    result = backtest_ablation_table(tmp_path)
    pd.testing.assert_frame_equal(result, summary)
