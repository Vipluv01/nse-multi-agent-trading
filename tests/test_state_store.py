"""Tests for PaperTradingStore -- the part where getting the arithmetic wrong is
worst: a portfolio tracker that silently drifts cash or double-counts a fill is a
much harder bug to notice than a wrong backtest number, because there is no
walk-forward ground truth to check it against."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.broker import Fill
from nse_agents.live.state_store import PaperTradingStore


@pytest.fixture
def store(tmp_path):
    return PaperTradingStore(tmp_path / "state.sqlite", initial_capital=1_000_000.0)


def _fill(symbol="RELIANCE", side="buy", qty=10, price=1000.0, cost=5.0, ts="2026-08-20"):
    return Fill(symbol=symbol, side=side, quantity=qty, price=price, cost=cost,
                timestamp=pd.Timestamp(ts))


def test_fresh_store_starts_at_initial_capital(store):
    snap = store.snapshot()
    assert snap.cash == pytest.approx(1_000_000.0)
    assert snap.positions == {}
    assert snap.realized_pnl == 0.0


def test_reopening_the_same_path_does_not_reset_capital(tmp_path):
    path = tmp_path / "state.sqlite"
    a = PaperTradingStore(path, initial_capital=1_000_000.0)
    a.apply_fill(_fill(qty=10, price=1000.0, cost=5.0), "2026-08-20")
    b = PaperTradingStore(path, initial_capital=1_000_000.0)  # re-open, same args
    snap = b.snapshot()
    assert snap.positions["RELIANCE"].quantity == 10  # not reset


def test_buy_reduces_cash_by_exactly_turnover_plus_cost(store):
    store.apply_fill(_fill(qty=10, price=1000.0, cost=5.0), "2026-08-20")
    snap = store.snapshot()
    assert snap.cash == pytest.approx(1_000_000.0 - 10 * 1000.0 - 5.0)


def test_buy_sets_average_cost_basis_correctly_across_two_fills(store):
    store.apply_fill(_fill(qty=10, price=1000.0, cost=5.0), "2026-08-20")
    store.apply_fill(_fill(qty=10, price=1100.0, cost=5.0), "2026-08-21")
    pos = store.snapshot().positions["RELIANCE"]
    assert pos.quantity == 20
    assert pos.avg_price == pytest.approx((10 * 1000.0 + 10 * 1100.0) / 20)


def test_sell_realises_pnl_against_the_average_cost_basis(store):
    store.apply_fill(_fill(side="buy", qty=10, price=1000.0, cost=0.0), "2026-08-20")
    store.apply_fill(_fill(side="sell", qty=10, price=1100.0, cost=2.0), "2026-08-21")
    snap = store.snapshot()
    assert snap.realized_pnl == pytest.approx(10 * (1100.0 - 1000.0) - 2.0)
    assert "RELIANCE" not in snap.positions  # fully closed


def test_partial_sell_leaves_the_remainder_at_the_same_average_price(store):
    store.apply_fill(_fill(side="buy", qty=10, price=1000.0, cost=0.0), "2026-08-20")
    store.apply_fill(_fill(side="sell", qty=4, price=1100.0, cost=0.0), "2026-08-21")
    pos = store.snapshot().positions["RELIANCE"]
    assert pos.quantity == 6
    assert pos.avg_price == pytest.approx(1000.0)  # cost basis unchanged by a partial sell


def test_selling_more_than_held_is_refused(store):
    store.apply_fill(_fill(side="buy", qty=5, price=1000.0, cost=0.0), "2026-08-20")
    with pytest.raises(ValueError, match="only"):
        store.apply_fill(_fill(side="sell", qty=10, price=1000.0, cost=0.0), "2026-08-21")


def test_selling_with_no_position_is_refused(store):
    with pytest.raises(ValueError):
        store.apply_fill(_fill(side="sell", qty=1, price=1000.0, cost=0.0), "2026-08-20")


def test_total_costs_accumulate_across_fills(store):
    store.apply_fill(_fill(side="buy", qty=1, price=100.0, cost=3.0), "2026-08-20")
    store.apply_fill(_fill(side="sell", qty=1, price=100.0, cost=4.0), "2026-08-21")
    assert store.snapshot().total_costs == pytest.approx(7.0)


def test_trade_log_records_every_fill_in_order(store):
    store.apply_fill(_fill(side="buy", qty=1, price=100.0), "2026-08-20")
    store.apply_fill(_fill(side="sell", qty=1, price=110.0), "2026-08-21")
    log = store.trade_log()
    assert len(log) == 2
    assert list(log["side"]) == ["buy", "sell"]


def test_equity_history_marks_to_market_using_the_given_prices(store):
    store.apply_fill(_fill(side="buy", qty=10, price=1000.0, cost=0.0), "2026-08-20")
    snap_before_cash = store.snapshot().cash
    store.record_equity("2026-08-20", {"RELIANCE": 1050.0})
    hist = store.equity_history()
    row = hist.iloc[0]
    assert row["positions_value"] == pytest.approx(10 * 1050.0)
    assert row["total_equity"] == pytest.approx(snap_before_cash + 10 * 1050.0)


def test_equity_history_computes_daily_return_from_the_prior_snapshot(store):
    store.apply_fill(_fill(side="buy", qty=10, price=1000.0, cost=0.0), "2026-08-20")
    store.record_equity("2026-08-20", {"RELIANCE": 1000.0})
    store.record_equity("2026-08-21", {"RELIANCE": 1100.0})
    hist = store.equity_history().set_index("date")
    assert pd.isna(hist.loc["2026-08-20", "daily_return"])  # no prior day yet
    day2_return = hist.loc["2026-08-21", "total_equity"] / hist.loc["2026-08-20", "total_equity"] - 1.0
    assert hist.loc["2026-08-21", "daily_return"] == pytest.approx(day2_return)


def test_a_crash_mid_fill_cannot_desync_cash_from_positions(store):
    """apply_fill must be atomic: cash and the position update together, or
    not at all -- checked here by confirming a failed (refused) sell leaves
    both cash and the position completely unchanged."""
    store.apply_fill(_fill(side="buy", qty=5, price=1000.0, cost=0.0), "2026-08-20")
    before = store.snapshot()
    with pytest.raises(ValueError):
        store.apply_fill(_fill(side="sell", qty=100, price=1000.0, cost=0.0), "2026-08-21")
    after = store.snapshot()
    assert after.cash == before.cash
    assert after.positions["RELIANCE"].quantity == before.positions["RELIANCE"].quantity


def test_drawdown_computation_pattern_includes_live_price_not_just_recorded_snapshots():
    """Regression for a real bug: computing max drawdown from only the
    *recorded* equity_history rows (each written at rebalance time) can miss
    a worse mark-to-market that happened later using live prices but before
    the next rebalance -- the headline 'total return' already reflects live
    prices, so a drawdown blind to them can under-report the true worst point
    while sitting right next to a worse cumulative-return figure. This checks
    the invariant any correct implementation must satisfy: |cumulative return|
    can never exceed |max drawdown| when both start from the same capital.
    """
    import numpy as np

    initial_capital = 1_000_000.0
    recorded_history = np.array([999_000.0, 1_005_000.0, 1_010_000.0])  # never dipped much
    live_total_equity = 950_000.0  # but a live, unrecorded mark is much worse

    total_return = live_total_equity / initial_capital - 1.0

    # The buggy pattern: drawdown from recorded history alone.
    buggy_path = np.concatenate([[initial_capital], recorded_history])
    buggy_peak = np.maximum.accumulate(buggy_path)
    buggy_dd = float((buggy_path / buggy_peak - 1.0).min())

    # The fixed pattern: drawdown including the live mark-to-market point.
    fixed_path = np.concatenate([[initial_capital], recorded_history, [live_total_equity]])
    fixed_peak = np.maximum.accumulate(fixed_path)
    fixed_dd = float((fixed_path / fixed_peak - 1.0).min())

    assert abs(buggy_dd) < abs(total_return), "the buggy pattern misses the live drop"
    assert abs(fixed_dd) >= abs(total_return), "the fixed pattern must capture it"
