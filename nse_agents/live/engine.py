"""Rebalance a persistent paper-trading account toward a day's target weights.

The seam between a day's decisions (from the orchestrator, expressed as target
portfolio weights, exactly as the backtest engine consumes them) and the persistent
state store: compute what trades are needed to move from today's actual holdings to
today's targets, execute them through a broker, and record the day's mark-to-market
equity. Small deltas are skipped rather than traded -- a real desk does not re-trade a
position over a rounding difference, and doing so here would manufacture cost drag
purely from floating-point noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .broker import AbstractBroker, Fill
from .state_store import PaperTradingStore


@dataclass(frozen=True)
class RebalanceResult:
    date: str
    fills: list[Fill]
    total_equity: float
    skipped: list[str]  # symbols whose delta was too small to trade


def rebalance(
    store: PaperTradingStore,
    broker: AbstractBroker,
    target_weights: dict[str, float],
    date: pd.Timestamp,
    min_trade_value: float = 100.0,
) -> RebalanceResult:
    """Move the account toward ``target_weights`` (symbol -> fraction of equity,
    0 for "flat"; symbols absent from the dict are treated as target 0).
    """
    date_str = pd.Timestamp(date).date().isoformat()
    snap = store.snapshot()

    prices: dict[str, float] = {}
    for symbol in set(target_weights) | set(snap.positions):
        prices[symbol] = broker.get_quote(symbol, date)

    current_positions_value = sum(
        pos.market_value(prices[sym]) for sym, pos in snap.positions.items()
    )
    total_equity = snap.cash + current_positions_value

    fills: list[Fill] = []
    skipped: list[str] = []

    # Sells first: frees cash that a same-day buy might need.
    all_symbols = sorted(set(target_weights) | set(snap.positions))
    deltas = {}
    for symbol in all_symbols:
        target_value = target_weights.get(symbol, 0.0) * total_equity
        current_value = (
            snap.positions[symbol].market_value(prices[symbol]) if symbol in snap.positions else 0.0
        )
        deltas[symbol] = target_value - current_value

    for symbol in sorted(deltas, key=lambda s: deltas[s]):  # most-negative (sells) first
        delta_value = deltas[symbol]
        if abs(delta_value) < min_trade_value:
            skipped.append(symbol)
            continue
        price = prices[symbol]
        quantity = abs(delta_value) / price
        side = "buy" if delta_value > 0 else "sell"
        if side == "sell":
            held = snap.positions[symbol].quantity if symbol in snap.positions else 0.0
            quantity = min(quantity, held)  # never sell more than is actually held
            if quantity <= 1e-9:
                skipped.append(symbol)
                continue
        fill = broker.submit_order(symbol, side, quantity, date)
        store.apply_fill(fill, date_str)
        fills.append(fill)

    final_snap = store.snapshot()
    final_positions_value = sum(
        pos.market_value(prices[sym]) for sym, pos in final_snap.positions.items() if sym in prices
    )
    mark_prices = {**prices}
    store.record_equity(date_str, mark_prices)
    final_equity = final_snap.cash + final_positions_value

    return RebalanceResult(date=date_str, fills=fills, total_equity=final_equity, skipped=skipped)
