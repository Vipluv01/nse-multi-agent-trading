"""Broker abstraction: the seam between the paper-trading engine and execution.

``AbstractBroker`` is the interface any execution backend implements -- ``MockBroker``
here, and eventually a real one. Nothing above this seam (state_store, the paper-trading
loop) knows or cares which is underneath; that is the point of the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: str          # "buy" | "sell"
    quantity: float     # shares (fractional allowed -- this is a model portfolio, not a real order book)
    price: float         # the price actually paid/received, after slippage
    cost: float           # rupees of transaction cost charged on this fill
    timestamp: pd.Timestamp


class AbstractBroker(ABC):
    @abstractmethod
    def get_quote(self, symbol: str, as_of: pd.Timestamp) -> float:
        """Reference price for ``symbol`` at ``as_of`` -- before slippage/costs."""

    @abstractmethod
    def submit_order(self, symbol: str, side: str, quantity: float, as_of: pd.Timestamp) -> Fill:
        """Execute a market order, returning the realised fill."""
