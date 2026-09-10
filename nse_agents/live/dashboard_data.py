"""Pure data-loading and computation for the Streamlit dashboard -- no ``streamlit``
import here, deliberately, so every function is testable with plain pytest and the
dashboard's actual business logic doesn't require a running Streamlit session (or even
streamlit installed) to verify. ``scripts/dashboard.py`` imports from this module and
is the only place ``import streamlit`` appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MIN_DAYS_FOR_SHARPE = 60  # same floor as cli.py and generate_paper_report.py


@dataclass(frozen=True)
class OverviewStats:
    total_equity: float
    total_return: float
    cumulative_return: float
    max_drawdown: float
    days_tracked: int
    sharpe: float | None       # None means withheld -- not enough data, not zero
    buyhold_sharpe: float | None


def load_account_state(db_path: Path | str):
    from .state_store import PaperTradingStore

    return PaperTradingStore(Path(db_path))


def compute_overview_stats(store, current_prices: dict[str, float]) -> OverviewStats:
    """Mirrors the exact computation in ``cli.py``'s ``cmd_paper_status`` --
    kept as one function so the CLI, the Markdown report, and the dashboard
    can never quietly disagree about what "current equity" or "max drawdown"
    means. (The bug this project's own KNOWN_ISSUES.md #10 documents was
    exactly two copies of this computation drifting apart.)
    """
    snap = store.snapshot()
    history = store.equity_history()

    positions_value = sum(
        pos.market_value(current_prices[sym]) for sym, pos in snap.positions.items()
        if sym in current_prices
    )
    total_equity = snap.cash + positions_value
    total_return = total_equity / snap.initial_capital - 1.0 if snap.initial_capital else 0.0

    equity_path = np.concatenate(
        [[snap.initial_capital], history["total_equity"].to_numpy(), [total_equity]]
    )
    peak = np.maximum.accumulate(equity_path)
    max_dd = float((equity_path / peak - 1.0).min()) if len(equity_path) else 0.0

    returns = history["daily_return"].dropna().to_numpy() if len(history) else np.array([])
    sharpe = None
    buyhold_sharpe = None
    if len(returns) >= MIN_DAYS_FOR_SHARPE:
        from ..backtest.metrics import compute_performance

        sharpe = compute_performance(returns).sharpe
        try:
            from ..backtest.baselines import build_baseline_signals
            from ..backtest.engine import backtest_signals

            bh = build_baseline_signals()["Buy&Hold"]
            start = pd.Timestamp(history["date"].iloc[0])
            window = bh.loc[bh["date"] >= start]
            if len(window):
                buyhold_sharpe = backtest_signals(window, "Buy&Hold", threshold=0.5).performance.sharpe
        except Exception:
            buyhold_sharpe = None  # benchmark unavailable must not break the dashboard

    return OverviewStats(
        total_equity=total_equity, total_return=total_return, cumulative_return=total_return,
        max_drawdown=max_dd, days_tracked=len(history), sharpe=sharpe, buyhold_sharpe=buyhold_sharpe,
    )


def positions_table(store, current_prices: dict[str, float]) -> pd.DataFrame:
    snap = store.snapshot()
    rows = []
    for sym, pos in sorted(snap.positions.items()):
        price = current_prices.get(sym)
        rows.append({
            "symbol": sym, "quantity": pos.quantity, "avg_price": pos.avg_price,
            "market_value": pos.market_value(price) if price else None,
            "return": (price / pos.avg_price - 1.0) if price else None,
        })
    return pd.DataFrame(rows, columns=["symbol", "quantity", "avg_price", "market_value", "return"])


def trades_table(store) -> pd.DataFrame:
    return store.trade_log()


def sentiment_distribution(sentiment_daily: pd.DataFrame) -> pd.DataFrame:
    """Histogram-ready bucketed counts from a daily-sentiment frame (the same
    shape ``aggregate_daily`` in ``nse_agents/agents/sentiment.py`` produces).
    """
    if sentiment_daily.empty:
        return pd.DataFrame(columns=["bucket", "count"])
    bins = [-1.0, -0.3, -0.05, 0.05, 0.3, 1.0]
    labels = ["very negative", "negative", "neutral", "positive", "very positive"]
    bucketed = pd.cut(sentiment_daily["sentiment"], bins=bins, labels=labels, include_lowest=True)
    return bucketed.value_counts().reindex(labels).rename_axis("bucket").reset_index(name="count")


def macro_regime_indicator(regime_table: pd.DataFrame, symbol: str) -> dict:
    """Latest VIX/Nifty-momentum reading for one symbol's regime row --
    reused from the same ``build_regime_table`` output the live pipeline
    already computes, not a second, separate macro fetch.
    """
    sub = regime_table[regime_table["symbol"] == symbol].sort_values("date")
    if sub.empty:
        return {"vix": None, "vix_percentile": None, "nifty_mom_20d": None, "date": None}
    row = sub.iloc[-1]
    return {
        "vix": row.get("vix"), "vix_percentile": row.get("vix_percentile"),
        "nifty_mom_20d": row.get("nifty_mom_20d"), "date": row.get("date"),
    }
