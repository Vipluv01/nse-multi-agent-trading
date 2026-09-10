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
    returns: np.ndarray,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    periods_per_year: float = TRADING_DAYS,
) -> float:
    """Annualised excess Sharpe.

    ``periods_per_year`` must match the rebalancing frequency of the return
    series: 252 for daily bars, 252/5 for non-overlapping weekly periods, and
    so on. Annualising a 20-day return series with sqrt(252) inflates the
    Sharpe by sqrt(20) -- roughly 4.5x -- which would make a longer-horizon
    variant look dramatically better for purely arithmetic reasons.
    """
    if len(returns) < 2:
        return 0.0
    period_rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    excess = returns - period_rf
    sd = excess.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods_per_year))


def compute_performance(
    returns: np.ndarray,
    turnover: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    periods_per_year: float = TRADING_DAYS,
) -> Performance:
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n == 0:
        return Performance(*([0] * 18))

    years = n / periods_per_year
    equity = float(np.prod(1.0 + returns))
    cagr = equity ** (1.0 / years) - 1.0 if years > 0 and equity > 0 else -1.0

    period_rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    downside = returns[returns < period_rf] - period_rf
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
        ann_return=float(returns.mean() * periods_per_year),
        ann_vol=float(returns.std(ddof=1) * np.sqrt(periods_per_year)),
        sharpe=sharpe_ratio(returns, risk_free_annual, periods_per_year),
        sharpe_gross=sharpe_ratio(returns, 0.0, periods_per_year),
        sortino=float((returns.mean() - period_rf) / dsd * np.sqrt(periods_per_year))
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



# ---- benchmark-relative risk attribution -----------------------------------------
#
# Everything above this point is a single-series statistic. These four need a second,
# paired return series (a benchmark) and are kept as standalone functions rather than
# folded into ``Performance`` -- ``compute_performance`` has no benchmark parameter,
# and giving it one would mean every existing call site either passes a benchmark it
# doesn't have or the field silently sits at a placeholder value. A benchmark-aware
# caller asks for these explicitly instead.
#
# All four require ``returns`` and ``benchmark_returns`` to already be aligned to the
# same dates, same length, in order -- callers already do this (e.g. via the aligned
# frame ``align_results`` in engine.py produces), so realigning here would either
# silently assume an alignment that doesn't hold or duplicate that logic a second time.
#
# **The benchmark series must use the same open-to-open convention as the portfolio
# returns** (``nse_agents.data.prices.forward_return``), not a naive close-to-close
# ``pct_change()``. Building this module, a benchmark built the naive way against a
# real Buy&Hold portfolio of the same stocks produced a correlation of 0.002 and a beta
# of 0.002 -- both should be close to 1.0, since the portfolio and the index share most
# of their constituents. Switching to ``forward_return`` fixed it: correlation 0.95,
# beta 1.02. The bug was never in these functions; it is exactly the trap a caller who
# builds their own benchmark series is one line away from falling into, which is why
# this is stated here rather than assumed obvious.


def _paired(returns: np.ndarray, benchmark_returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    returns = np.asarray(returns, dtype=float)
    benchmark_returns = np.asarray(benchmark_returns, dtype=float)
    if returns.shape != benchmark_returns.shape:
        raise ValueError(
            f"returns and benchmark_returns must be the same length and already "
            f"date-aligned; got {returns.shape} vs {benchmark_returns.shape}"
        )
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    return returns[mask], benchmark_returns[mask]


def beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Portfolio beta vs the benchmark: Cov(r, b) / Var(b)."""
    r, b = _paired(returns, benchmark_returns)
    if len(r) < 2:
        return float("nan")
    var_b = float(np.var(b, ddof=1))
    if var_b < 1e-16:
        return float("nan")  # a benchmark with ~zero variance makes beta undefined, not zero
    cov = float(np.cov(r, b, ddof=1)[0, 1])
    return cov / var_b


def treynor_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    periods_per_year: float = TRADING_DAYS,
) -> float:
    """Annualised excess return per unit of *systematic* (market) risk, unlike
    Sharpe's per unit of *total* risk. Undefined (NaN, not zero or infinity)
    when beta is undefined or is ~0 -- dividing by a near-zero beta would
    produce an arbitrarily large, meaningless ratio, exactly the kind of
    silent-precision failure this project's discipline exists to avoid.
    """
    r, b = _paired(returns, benchmark_returns)
    if len(r) < 2:
        return float("nan")
    portfolio_beta = beta(r, b)
    if not np.isfinite(portfolio_beta) or abs(portfolio_beta) < 1e-8:
        return float("nan")
    period_rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0
    ann_excess = float((r - period_rf).mean() * periods_per_year)
    return ann_excess / portfolio_beta


def information_ratio(
    returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: float = TRADING_DAYS,
) -> float:
    """Annualised active return (vs the benchmark) per unit of tracking error.
    Unlike Sharpe, this is benchmark-relative throughout -- no risk-free rate
    involved, since "active return" is already a return in excess of
    something (the benchmark), not of cash.

    Two ~zero-tracking-error cases are handled differently, and conflating
    them was a real bug caught by this module's own tests: *no active
    return either* (the series ARE each other -- zero decisions, IR is
    trivially 0) is not the same case as *a nonzero, near-riskless active
    return* (a constant excess with ~0 variance -- an edge with no measured
    risk, which is an ill-defined ratio to report as a finite number, not a
    genuine zero). The second case returns NaN, matching how
    ``treynor_ratio`` refuses to divide by a near-zero beta.
    """
    r, b = _paired(returns, benchmark_returns)
    if len(r) < 2:
        return float("nan")
    active = r - b
    tracking_error = float(active.std(ddof=1))
    if tracking_error < 1e-12:
        if abs(float(active.mean())) < 1e-12:
            return 0.0  # genuinely tracking the benchmark: no active return, no active risk
        return float("nan")  # a nonzero "riskless" edge is undefined, not a real zero
    return float(active.mean() * periods_per_year) / (tracking_error * np.sqrt(periods_per_year))


def upside_capture_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Mean portfolio return on days the benchmark rose, divided by the
    benchmark's own mean return on those same days. 1.0 = captures the
    market's upside exactly; below 1.0 = gives some of it up.
    """
    r, b = _paired(returns, benchmark_returns)
    up = b > 0
    if not up.any():
        return float("nan")
    bench_up_mean = float(b[up].mean())
    if abs(bench_up_mean) < 1e-12:
        return float("nan")
    return float(r[up].mean()) / bench_up_mean


def downside_capture_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """Mean portfolio return on days the benchmark fell, divided by the
    benchmark's own mean return on those same days. Below 1.0 = the strategy
    lost *less* than the market on down days -- real downside protection, the
    same mechanism the crash-regime analysis (scripts/regime_analysis.py)
    measures directly rather than via this ratio's benchmark-day definition.
    """
    r, b = _paired(returns, benchmark_returns)
    down = b < 0
    if not down.any():
        return float("nan")
    bench_down_mean = float(b[down].mean())
    if abs(bench_down_mean) < 1e-12:
        return float("nan")
    return float(r[down].mean()) / bench_down_mean

def equity_curve(returns: np.ndarray, dates: pd.DatetimeIndex) -> pd.DataFrame:
    eq = np.cumprod(1.0 + np.nan_to_num(returns))
    return pd.DataFrame({"date": dates, "equity": eq, "drawdown": drawdown_series(np.nan_to_num(returns))})
