"""Integration test: the full paper-trading loop across 5 real, historical trading
days, using MockBroker for execution.

Uses real cached NSE prices rather than synthetic data, so the fill/slippage/cost
arithmetic runs against real price levels -- but this is still explicitly a pipeline
correctness test, not a trading evaluation: the "signals" driving each day's target
weights are a small, fixed, hand-written schedule, not the study's walk-forward model.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.engine import rebalance
from nse_agents.live.mock_broker import MockBroker
from nse_agents.live.state_store import PaperTradingStore

FIVE_DAYS = ["2026-08-28", "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"]

# A small, fixed weight schedule -- not a live signal -- chosen to exercise
# opening, resizing, rotating out of, and fully closing a position.
SCHEDULE = [
    {"RELIANCE": 0.20, "TCS": 0.20},                         # open two positions
    {"RELIANCE": 0.20, "TCS": 0.20},                         # hold -- must not re-trade
    {"RELIANCE": 0.10, "TCS": 0.20, "INFY": 0.15},            # trim one, open another
    {"RELIANCE": 0.0, "TCS": 0.20, "INFY": 0.15},             # fully close RELIANCE
    {"TCS": 0.30, "INFY": 0.15},                              # top up TCS
]


@pytest.fixture
def store(tmp_path):
    return PaperTradingStore(tmp_path / "state.sqlite", initial_capital=1_000_000.0)


@pytest.fixture
def broker():
    return MockBroker()


def test_five_day_paper_trading_loop_runs_without_error(store, broker):
    results = []
    for date_str, targets in zip(FIVE_DAYS, SCHEDULE):
        result = rebalance(store, broker, targets, pd.Timestamp(date_str))
        results.append(result)

    assert len(results) == 5
    assert all(r.total_equity > 0 for r in results)


def test_holding_the_same_target_weight_only_drift_trades_not_re_trades(store, broker):
    """Day 2's target weights equal day 1's, but a fixed *weight* is a moving
    dollar target as prices move total equity -- so a small rebalancing drift
    trade is correct, expected behaviour (the backtest engine's own weight
    convention works the same way), not a bug. What must NOT happen is a full
    re-trade of the position, which would mean the engine forgot what it
    already holds and rebuilt from zero.
    """
    day1 = rebalance(store, broker, SCHEDULE[0], pd.Timestamp(FIVE_DAYS[0]))
    held_after_day1 = {f.symbol: f.quantity for f in day1.fills}

    day2 = rebalance(store, broker, SCHEDULE[1], pd.Timestamp(FIVE_DAYS[1]))
    for fill in day2.fills:
        # A drift trade must be a small correction, not a full re-purchase --
        # bound it well under the size of the original position.
        assert fill.quantity < 0.1 * held_after_day1[fill.symbol]


def test_target_zero_fully_closes_the_position(store, broker):
    for date_str, targets in zip(FIVE_DAYS[:4], SCHEDULE[:4]):
        rebalance(store, broker, targets, pd.Timestamp(date_str))
    assert "RELIANCE" not in store.snapshot().positions


def test_trade_log_has_one_row_per_actual_fill_across_the_whole_run(store, broker):
    total_fills = 0
    for date_str, targets in zip(FIVE_DAYS, SCHEDULE):
        result = rebalance(store, broker, targets, pd.Timestamp(date_str))
        total_fills += len(result.fills)
    assert len(store.trade_log()) == total_fills
    assert total_fills > 0


def test_equity_history_has_exactly_one_row_per_trading_day(store, broker):
    for date_str, targets in zip(FIVE_DAYS, SCHEDULE):
        rebalance(store, broker, targets, pd.Timestamp(date_str))
    hist = store.equity_history()
    assert len(hist) == 5
    assert list(hist["date"]) == FIVE_DAYS


def test_final_equity_reflects_costs_and_slippage_not_just_price_moves(store, broker):
    """With five rebalances' worth of costs and slippage, ending equity must
    differ from a naive frictionless mark-to-market -- if it didn't, the
    engine would be silently ignoring the broker's own reported costs."""
    for date_str, targets in zip(FIVE_DAYS, SCHEDULE):
        rebalance(store, broker, targets, pd.Timestamp(date_str))
    snap = store.snapshot()
    assert snap.total_costs > 0
    hist = store.equity_history()
    # Total drag from costs alone should show up as a measurable fraction of equity.
    assert snap.total_costs / hist["total_equity"].iloc[-1] > 0.0001


def test_state_persists_across_a_simulated_restart(tmp_path, broker):
    """Re-opening the store mid-run (simulating the process restarting between
    trading days) must not lose any state -- the whole point of persistence."""
    path = tmp_path / "state.sqlite"
    store_a = PaperTradingStore(path, initial_capital=1_000_000.0)
    for date_str, targets in zip(FIVE_DAYS[:3], SCHEDULE[:3]):
        rebalance(store_a, broker, targets, pd.Timestamp(date_str))
    mid_snap = store_a.snapshot()

    store_b = PaperTradingStore(path)  # fresh instance, same file
    resumed_snap = store_b.snapshot()
    assert resumed_snap.cash == pytest.approx(mid_snap.cash)
    assert set(resumed_snap.positions) == set(mid_snap.positions)

    for date_str, targets in zip(FIVE_DAYS[3:], SCHEDULE[3:]):
        rebalance(store_b, broker, targets, pd.Timestamp(date_str))
    assert len(store_b.equity_history()) == 5  # all 5 days present, not just the last 2
