"""A mock broker for paper-trading tests and demos.

Fills are simulated, not real -- this is the honesty boundary the whole project has
maintained throughout: a mock broker is for testing the *pipeline's* correctness (does
state update right, do costs get charged right, does the report reflect the trade log
right), never for producing a number that could be mistaken for a real trading result.

Reuses the project's own ``CostModel`` for transaction costs rather than
re-encoding the STT/stamp-duty rates a second time -- a second, independent copy
of the same tax schedule is exactly the kind of drift that produces two
different "round-trip cost" numbers in two different files, silently.
"""

from __future__ import annotations

import pandas as pd

from ..config import CostModel, SETTINGS
from ..data.prices import load_prices
from .broker import AbstractBroker, Fill


class MockBroker(AbstractBroker):
    def __init__(
        self,
        costs: CostModel | None = None,
        slippage_bps: float = 5.0,
        price_field: str = "close",
    ):
        """``slippage_bps`` is on top of whatever slippage ``costs`` already
        models (``CostModel.slippage`` is 5bps by default) -- deliberately
        additive here rather than reused, because this class models
        *execution* slippage (the bid-ask spread a market order crosses),
        while ``CostModel.slippage`` models *market-impact* slippage (the
        price moving against a resting order). They are two different real
        costs, not the same one counted twice under two names.
        """
        self.costs = costs or CostModel()
        self.slippage_bps = slippage_bps
        self.price_field = price_field
        self._price_cache: dict[str, pd.DataFrame] = {}

    def _prices(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._price_cache:
            self._price_cache[symbol] = load_prices(symbol, SETTINGS.start, "2100-01-01")
        return self._price_cache[symbol]

    def get_quote(self, symbol: str, as_of: pd.Timestamp) -> float:
        frame = self._prices(symbol)
        row = frame.loc[frame["date"] == pd.Timestamp(as_of)]
        if row.empty:
            raise KeyError(f"no price for {symbol} on {as_of.date()}")
        return float(row[self.price_field].iloc[0])

    def submit_order(self, symbol: str, side: str, quantity: float, as_of: pd.Timestamp) -> Fill:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        reference = self.get_quote(symbol, as_of)
        # A market buy crosses the spread upward, a market sell crosses it
        # downward -- slippage always moves the fill against the trader.
        slip = reference * (self.slippage_bps / 1e4)
        fill_price = reference + slip if side == "buy" else reference - slip

        turnover = fill_price * quantity
        cost = self.costs.cost(turnover, side)

        return Fill(
            symbol=symbol, side=side, quantity=quantity, price=fill_price,
            cost=cost, timestamp=pd.Timestamp(as_of),
        )
