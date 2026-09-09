"""Performance metrics.

Deliberately reports risk-adjusted and drawdown numbers alongside return.
A rupee P&L or a win rate on its own is not evaluable: a strategy that wins
94% of the time and loses everything on the 95th trade produces an excellent
win rate and an unacceptable Sharpe.

Sharpe here is an **excess**-return Sharpe. India's risk-free rate has run
around 6-7%, so a gross Sharpe computed against zero flatters an Indian equity
strategy far more than it would a US one -- a 7% annual drift is free money a
liquid-fund investor gets without taking equity risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.06


@dataclass
class Performance:
    n_days: int
    cagr: float
    ann_return: float
    ann_vol: float
    sharpe: float
    sharpe_gross: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    turnover_annual: float
    exposure: float
    skew: float
    kurtosis: float
    worst_day: float
    best_day: float

    def to_dict(self) -> dict:
        return asdict(self)


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return equity / peak - 1.0


def max_drawdown(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(drawdown_series(returns).min())


def sharpe_ratio(
    returns: np.ndarray, risk_free_annual: float = RISK_FREE_ANNUAL
) -> float:
    if len(returns) < 2:
        return 0.0
    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    excess = returns - daily_rf
    sd = excess.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def compute_performance(
    returns: np.ndarray,
    turnover: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> Performance:
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n == 0:
        return Performance(*([0] * 18))

    years = n / TRADING_DAYS
    equity = float(np.prod(1.0 + returns))
    cagr = equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0

    daily_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    downside = returns[returns < daily_rf] - daily_rf
    dsd = downside.std(ddof=1) if len(downside) > 1 else 0.0

    # A degenerate (near-constant) series makes scipy's moment calculation
    # numerically meaningless; report 0 rather than emit a precision warning.
    degenerate = returns.std(ddof=1) < 1e-12
    skew_val = 0.0 if degenerate else float(sps.skew(returns))
    kurt_val = 0.0 if degenerate else float(sps.kurtosis(returns, fisher=True))

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    mdd = max_drawdown(returns)

    return Performance(
        n_days=n,
        cagr=float(cagr),
        ann_return=float(returns.mean() * TRADING_DAYS),
        ann_vol=float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        sharpe=sharpe_ratio(returns, risk_free_annual),
        sharpe_gross=sharpe_ratio(returns, 0.0),
        sortino=float((returns.mean() - daily_rf) / dsd * np.sqrt(TRADING_DAYS))
        if dsd > 1e-12
        else 0.0,
        max_drawdown=mdd,
        calmar=float(cagr / abs(mdd)) if mdd < -1e-9 else 0.0,
        hit_rate=float((returns > 0).mean()),
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        profit_factor=float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf,
        turnover_annual=float(np.nansum(turnover) / years) if turnover is not None else 0.0,
        exposure=float(np.nanmean(exposure)) if exposure is not None else 1.0,
        skew=skew_val,
        kurtosis=kurt_val,
        worst_day=float(returns.min()),
        best_day=float(returns.max()),
    )


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Directional-accuracy metrics, including MCC and AUC as in PLSTM-TAL."""
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(prob) > 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    # AUC via the Mann-Whitney U identity; avoids a sklearn dependency.
    pos, neg = np.asarray(prob)[y_true == 1], np.asarray(prob)[y_true == 0]
    if len(pos) and len(neg):
        ranks = sps.rankdata(np.concatenate([pos, neg]))
        auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    else:
        auc = 0.5

    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(mcc),
        "auc": float(auc),
        "n": int(len(y_true)),
        "predicted_up_rate": float(pred.mean()),
        "actual_up_rate": float(y_true.mean()),
    }


def equity_curve(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    eq = np.cumprod(1.0 + np.nan_to_num(returns))
    return pd.DataFrame({"date": dates, "equity": eq, "drawdown": drawdown_series(np.nan_to_num(returns))})
