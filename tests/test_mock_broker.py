"""Tests for MockBroker."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import CostModel
from nse_agents.live.mock_broker import MockBroker


def test_buy_fills_above_the_quote_sell_fills_below():
    """A market order always crosses the spread against the trader."""
    broker = MockBroker(slippage_bps=5.0)
    quote = broker.get_quote("RELIANCE", pd.Timestamp("2026-08-27"))
    buy = broker.submit_order("RELIANCE", "buy", 10, pd.Timestamp("2026-08-27"))
    sell = broker.submit_order("RELIANCE", "sell", 10, pd.Timestamp("2026-08-27"))
    assert buy.price > quote
    assert sell.price < quote


def test_slippage_scales_with_the_configured_bps():
    tight = MockBroker(slippage_bps=1.0)
    wide = MockBroker(slippage_bps=50.0)
    quote = tight.get_quote("RELIANCE", pd.Timestamp("2026-08-27"))
    tight_fill = tight.submit_order("RELIANCE", "buy", 1, pd.Timestamp("2026-08-27"))
    wide_fill = wide.submit_order("RELIANCE", "buy", 1, pd.Timestamp("2026-08-27"))
    assert (wide_fill.price - quote) > (tight_fill.price - quote)


def test_cost_uses_the_projects_own_cost_model_not_a_reimplementation():
    """The mock broker must not silently diverge from config.CostModel's rates."""
    costs = CostModel()
    broker = MockBroker(costs=costs, slippage_bps=0.0)
    fill = broker.submit_order("RELIANCE", "buy", 100, pd.Timestamp("2026-08-27"))
    expected_cost = costs.cost(fill.price * 100, "buy")
    assert fill.cost == pytest.approx(expected_cost)


def test_zero_or_negative_quantity_is_refused():
    broker = MockBroker()
    with pytest.raises(ValueError):
        broker.submit_order("RELIANCE", "buy", 0, pd.Timestamp("2026-08-27"))
    with pytest.raises(ValueError):
        broker.submit_order("RELIANCE", "buy", -5, pd.Timestamp("2026-08-27"))


def test_invalid_side_is_refused():
    broker = MockBroker()
    with pytest.raises(ValueError):
        broker.submit_order("RELIANCE", "hold", 1, pd.Timestamp("2026-08-27"))


def test_quote_for_a_non_trading_date_raises_not_silently_returns_stale_price():
    broker = MockBroker()
    with pytest.raises(KeyError):
        broker.get_quote("RELIANCE", pd.Timestamp("2026-08-30"))  # a Sunday
