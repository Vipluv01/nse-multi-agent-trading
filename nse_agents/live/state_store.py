"""Persistent paper-trading state: cash, positions, trade log, daily equity.

SQLite, not JSON: the state accumulates incrementally (one trade at a time, one day at
a time) rather than being rewritten wholesale each run, and a trade log that must never
lose a row under a crash mid-write is exactly what a transactional store is for. It is
also the same technology already used elsewhere in this project (``llm/cache.py``), so
this does not introduce a new storage paradigm for no reason.

Nothing here is a backtest. State only ever advances forward from whatever fills the
broker actually reports -- there is no way to feed it a full historical return series
at once, by design, so it structurally cannot be used to reproduce or contaminate the
walk-forward study's numbers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import RESULTS
from .broker import Fill

DEFAULT_DB_PATH = RESULTS / "paper_trading" / "state.sqlite"


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: float
    avg_price: float
    opened_at: str

    def market_value(self, current_price: float) -> float:
        return self.quantity * current_price

    def unrealized_pnl(self, current_price: float) -> float:
        return self.quantity * (current_price - self.avg_price)


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    initial_capital: float
    positions: dict[str, Position]
    realized_pnl: float
    total_costs: float
    last_updated: str


class PaperTradingStore:
    def __init__(self, path: Path | str = DEFAULT_DB_PATH, initial_capital: float | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._create_schema()
        self._ensure_account(initial_capital)

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cash REAL NOT NULL,
                initial_capital REAL NOT NULL,
                realized_pnl REAL NOT NULL DEFAULT 0,
                total_costs REAL NOT NULL DEFAULT 0,
                last_updated TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS positions (
                symbol TEXT PRIMARY KEY,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                opened_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                cost REAL NOT NULL,
                timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS equity_history (
                date TEXT PRIMARY KEY,
                cash REAL NOT NULL,
                positions_value REAL NOT NULL,
                total_equity REAL NOT NULL,
                daily_return REAL
            );
            """
        )
        self._conn.commit()

    def _ensure_account(self, initial_capital: float | None) -> None:
        row = self._conn.execute("SELECT cash FROM account WHERE id = 1").fetchone()
        if row is None:
            capital = initial_capital if initial_capital is not None else 1_000_000.0
            self._conn.execute(
                "INSERT INTO account (id, cash, initial_capital, realized_pnl, total_costs, last_updated) "
                "VALUES (1, ?, ?, 0, 0, ?)",
                (capital, capital, pd.Timestamp.now().isoformat()),
            )
            self._conn.commit()

    # ---- reads -----------------------------------------------------------

    def snapshot(self) -> AccountSnapshot:
        acc = self._conn.execute(
            "SELECT cash, initial_capital, realized_pnl, total_costs, last_updated FROM account WHERE id = 1"
        ).fetchone()
        positions = {
            row[0]: Position(symbol=row[0], quantity=row[1], avg_price=row[2], opened_at=row[3])
            for row in self._conn.execute("SELECT symbol, quantity, avg_price, opened_at FROM positions")
        }
        return AccountSnapshot(
            cash=acc[0], initial_capital=acc[1], positions=positions,
            realized_pnl=acc[2], total_costs=acc[3], last_updated=acc[4],
        )

    def trade_log(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT date, symbol, side, quantity, price, cost, timestamp FROM trades ORDER BY id",
            self._conn,
        )

    def equity_history(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT date, cash, positions_value, total_equity, daily_return "
            "FROM equity_history ORDER BY date",
            self._conn,
        )

    # ---- writes ------------------------------------------------------------

    def apply_fill(self, fill: Fill, date: str) -> None:
        """Record one fill: updates cash, the position (weighted-average cost
        basis on a buy; realises P&L on a sell), and the trade log -- all in
        one transaction, so a crash mid-update cannot leave cash and
        positions inconsistent with each other.
        """
        snap = self.snapshot()
        existing = snap.positions.get(fill.symbol)

        if fill.side == "buy":
            turnover = fill.price * fill.quantity
            new_cash = snap.cash - turnover - fill.cost
            if existing is None:
                new_qty, new_avg = fill.quantity, fill.price
                opened_at = date
            else:
                new_qty = existing.quantity + fill.quantity
                new_avg = (
                    existing.quantity * existing.avg_price + fill.quantity * fill.price
                ) / new_qty
                opened_at = existing.opened_at
            self._conn.execute(
                "INSERT INTO positions (symbol, quantity, avg_price, opened_at) VALUES (?,?,?,?) "
                "ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, avg_price=excluded.avg_price",
                (fill.symbol, new_qty, new_avg, opened_at),
            )
            realized_delta = 0.0
        else:  # sell
            if existing is None or existing.quantity < fill.quantity - 1e-9:
                raise ValueError(
                    f"cannot sell {fill.quantity} {fill.symbol}: only "
                    f"{existing.quantity if existing else 0} held"
                )
            realized_delta = fill.quantity * (fill.price - existing.avg_price) - fill.cost
            remaining = existing.quantity - fill.quantity
            turnover = fill.price * fill.quantity
            new_cash = snap.cash + turnover - fill.cost
            if remaining <= 1e-9:
                self._conn.execute("DELETE FROM positions WHERE symbol = ?", (fill.symbol,))
            else:
                self._conn.execute(
                    "UPDATE positions SET quantity = ? WHERE symbol = ?", (remaining, fill.symbol)
                )

        self._conn.execute(
            "UPDATE account SET cash = ?, realized_pnl = realized_pnl + ?, "
            "total_costs = total_costs + ?, last_updated = ? WHERE id = 1",
            (new_cash, realized_delta, fill.cost, pd.Timestamp.now().isoformat()),
        )
        self._conn.execute(
            "INSERT INTO trades (date, symbol, side, quantity, price, cost, timestamp) VALUES (?,?,?,?,?,?,?)",
            (date, fill.symbol, fill.side, fill.quantity, fill.price, fill.cost, fill.timestamp.isoformat()),
        )
        self._conn.commit()

    def record_equity(self, date: str, prices: dict[str, float]) -> AccountSnapshot:
        """Snapshot today's mark-to-market equity, using ``prices`` (current
        close per held symbol) -- called once per trading day, after any
        fills for that day have been applied.
        """
        snap = self.snapshot()
        positions_value = sum(
            pos.market_value(prices[sym]) for sym, pos in snap.positions.items() if sym in prices
        )
        total_equity = snap.cash + positions_value

        prior = self._conn.execute(
            "SELECT total_equity FROM equity_history ORDER BY date DESC LIMIT 1"
        ).fetchone()
        daily_return = (total_equity / prior[0] - 1.0) if prior and prior[0] > 0 else None

        self._conn.execute(
            "INSERT INTO equity_history (date, cash, positions_value, total_equity, daily_return) "
            "VALUES (?,?,?,?,?) ON CONFLICT(date) DO UPDATE SET "
            "cash=excluded.cash, positions_value=excluded.positions_value, "
            "total_equity=excluded.total_equity, daily_return=excluded.daily_return",
            (date, snap.cash, positions_value, total_equity, daily_return),
        )
        self._conn.commit()
        return snap
