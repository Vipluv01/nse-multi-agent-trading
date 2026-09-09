"""Portfolio backtest: signals in, cost-adjusted daily returns out.

Structural choices, each of which changes the answer:

* **Long-only.** Delivery-segment shorting is not available to Indian retail
  beyond the intraday window, so a long/short book would be untradeable at this
  horizon. This caps the achievable Sharpe and is stated rather than hidden.
* **Idle cash earns the risk-free rate.** Without this, a selective strategy is
  punished for being flat, which quietly biases every comparison toward
  always-invested baselines.
* **Costs charged on realised weight changes**, buy and sell legs priced
  separately, because Indian STT and stamp duty are asymmetric across sides.
* **Returns are open-to-open**, matching the ``forward_return`` convention:
  the signal is formed at the close of ``t`` and the fill happens at ``t+1``'s
  open. No part of the pipeline can see the price it trades at.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SETTINGS, CostModel
from .metrics import TRADING_DAYS, RISK_FREE_ANNUAL, Performance, compute_performance


@dataclass
class BacktestResult:
    name: str
    daily: pd.DataFrame          # date, gross_return, cost, net_return, exposure, turnover
    performance: Performance
    performance_gross: Performance

    @property
    def returns(self) -> np.ndarray:
        return self.daily["net_return"].to_numpy()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.daily["date"])

    def cost_drag_annual(self) -> float:
        """Annualised return given up to costs -- the number that decides
        whether a daily-rebalanced signal is tradeable at all."""
        return float(self.daily["cost"].mean() * TRADING_DAYS)


def signals_to_weights(
    signals: pd.DataFrame,
    threshold: float = 0.5,
    max_weight: float = SETTINGS.max_weight_per_name,
    score_column: str = "prob_up",
) -> pd.DataFrame:
    """Map per-name scores to target portfolio weights.

    Equal weight across every name clearing ``threshold``, each capped at
    ``max_weight``; the remainder sits in cash. Sizing by signal strength was
    considered and rejected -- with a near-coin-flip classifier, strength-based
    sizing mostly amplifies noise, and it would confound the architecture
    comparison with a sizing choice.
    """
    frame = signals.copy()
    frame["selected"] = (frame[score_column] > threshold).astype(float)
    counts = frame.groupby("date")["selected"].transform("sum")
    per_name = np.where(counts > 0, 1.0 / counts.clip(lower=1.0), 0.0)
    frame["weight"] = frame["selected"] * np.minimum(per_name, max_weight)
    return frame


def run_backtest(
    weighted: pd.DataFrame,
    name: str,
    costs: CostModel = SETTINGS.costs,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    return_column: str = "fwd_ret",
) -> BacktestResult:
    """Aggregate per-name weights and forward returns into a portfolio track."""
    frame = weighted.sort_values(["date", "symbol"]).copy()
    pivot_w = frame.pivot_table(index="date", columns="symbol", values="weight", fill_value=0.0)
    pivot_r = frame.pivot_table(index="date", columns="symbol", values=return_column)
    pivot_r = pivot_r.reindex(columns=pivot_w.columns)

    weights = pivot_w.to_numpy(dtype=float)
    rets = np.nan_to_num(pivot_r.to_numpy(dtype=float))

    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    invested = weights.sum(axis=1)
    cash = np.clip(1.0 - invested, 0.0, None)
    gross = (weights * rets).sum(axis=1) + cash * daily_rf

    # Turnover: compare each day's target weights with the previous day's, per
    # name, so a name held flat across days is not charged twice.
    previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    delta = weights - previous
    buy_turnover = np.clip(delta, 0.0, None).sum(axis=1)
    sell_turnover = np.clip(-delta, 0.0, None).sum(axis=1)

    unit_buy = costs.cost(1.0, "buy")
    unit_sell = costs.cost(1.0, "sell")
    cost = buy_turnover * unit_buy + sell_turnover * unit_sell

    net = gross - cost
    daily = pd.DataFrame(
        {
            "date": pivot_w.index,
            "gross_return": gross,
            "cost": cost,
            "net_return": net,
            "exposure": invested,
            "turnover": buy_turnover + sell_turnover,
        }
    )
    return BacktestResult(
        name=name,
        daily=daily,
        performance=compute_performance(net, daily["turnover"].to_numpy(), invested, risk_free_annual),
        performance_gross=compute_performance(gross, daily["turnover"].to_numpy(), invested, risk_free_annual),
    )


def backtest_signals(
    signals: pd.DataFrame,
    name: str,
    threshold: float = 0.5,
    score_column: str = "prob_up",
    costs: CostModel = SETTINGS.costs,
    max_weight: float = SETTINGS.max_weight_per_name,
) -> BacktestResult:
    weighted = signals_to_weights(signals, threshold, max_weight, score_column)
    return run_backtest(weighted, name, costs)


def align_results(results: list[BacktestResult]) -> pd.DataFrame:
    """Inner-join every track on date, so comparisons are on identical days."""
    merged: pd.DataFrame | None = None
    for res in results:
        col = res.daily[["date", "net_return"]].rename(columns={"net_return": res.name})
        merged = col if merged is None else merged.merge(col, on="date", how="inner")
    assert merged is not None
    return merged.sort_values("date").reset_index(drop=True)


def summary_table(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    for res in results:
        perf = res.performance
        rows.append(
            {
                "strategy": res.name,
                "CAGR": perf.cagr,
                "Sharpe(net,excess)": perf.sharpe,
                "Sharpe(gross)": res.performance_gross.sharpe,
                "MaxDD": perf.max_drawdown,
                "Calmar": perf.calmar,
                "HitRate": perf.hit_rate,
                "AnnVol": perf.ann_vol,
                "Turnover/yr": perf.turnover_annual,
                "Exposure": perf.exposure,
                "CostDrag/yr": res.cost_drag_annual(),
                "Days": perf.n_days,
            }
        )
    return pd.DataFrame(rows)
