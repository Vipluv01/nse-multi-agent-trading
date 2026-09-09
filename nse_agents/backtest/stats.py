"""Statistical machinery for comparing strategies.

Three things distinguish this from reporting a bare Sharpe:

1. **Block bootstrap, not i.i.d. bootstrap.** Daily strategy returns are
   autocorrelated and volatility-clustered. Resampling single days destroys
   that structure and produces confidence intervals that are far too narrow.
2. **Paired comparisons.** Two strategies trading the same days share the same
   market shocks; comparing their independent CIs wastes that pairing and
   understates significance in both directions.
3. **A multiple-testing correction, and a Deflated Sharpe Ratio.** Testing 4
   architectures x 5 baselines and reporting the winner's p-value unadjusted
   is how backtest overfitting is manufactured. The DSR additionally asks
   whether the best observed Sharpe is what you would expect from that many
   trials on pure noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .metrics import TRADING_DAYS, RISK_FREE_ANNUAL, sharpe_ratio

EULER_MASCHERONI = 0.5772156649015329


@dataclass
class Interval:
    point: float
    low: float
    high: float
    p_value: float | None = None

    def excludes_zero(self) -> bool:
        return (self.low > 0) or (self.high < 0)

    def __str__(self) -> str:
        base = f"{self.point:+.3f} [{self.low:+.3f}, {self.high:+.3f}]"
        return base if self.p_value is None else f"{base} p={self.p_value:.4f}"


def _block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap index draw."""
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])
    return idx[:n]


def block_bootstrap_sharpe(
    returns: np.ndarray,
    n_boot: int = 5000,
    block: int = 20,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    seed: int = 20260909,
    alpha: float = 0.05,
) -> Interval:
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < block * 2:
        return Interval(sharpe_ratio(returns, risk_free_annual), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = sharpe_ratio(returns[_block_indices(len(returns), block, rng)], risk_free_annual)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # Two-sided bootstrap p-value for "Sharpe <= 0".
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return Interval(sharpe_ratio(returns, risk_free_annual), float(lo), float(hi), float(min(p, 1.0)))


def paired_sharpe_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 5000,
    block: int = 20,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    seed: int = 20260909,
    alpha: float = 0.05,
) -> Interval:
    """Bootstrap CI for Sharpe(a) - Sharpe(b) on commonly-dated returns."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    keep = ~np.isnan(a) & ~np.isnan(b)
    a, b = a[keep], b[keep]
    point = sharpe_ratio(a, risk_free_annual) - sharpe_ratio(b, risk_free_annual)
    if len(a) < block * 2:
        return Interval(point, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = _block_indices(len(a), block, rng)  # same index for both: paired
        draws[i] = sharpe_ratio(a[idx], risk_free_annual) - sharpe_ratio(b[idx], risk_free_annual)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2.0 * min((draws <= 0).mean(), (draws >= 0).mean())
    return Interval(point, float(lo), float(hi), float(min(p, 1.0)))


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_sharpe_annual: float = 0.0,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> float:
    """P(true Sharpe > benchmark), adjusting for skew and fat tails.

    A high Sharpe earned through negatively-skewed, fat-tailed returns -- which
    is what most option-selling and mean-reversion strategies look like -- is
    less trustworthy than the same number from Gaussian returns, and the PSR
    prices that in.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    n = len(returns)
    if n < 3:
        return float("nan")
    sr = sharpe_ratio(returns, risk_free_annual) / np.sqrt(TRADING_DAYS)  # per-period
    sr_star = benchmark_sharpe_annual / np.sqrt(TRADING_DAYS)
    skew = float(sps.skew(returns))
    kurt = float(sps.kurtosis(returns, fisher=False))
    denom = np.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2, 1e-12))
    return float(sps.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / denom))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    all_trial_sharpes: np.ndarray,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> tuple[float, float]:
    """Bailey & Lopez de Prado's DSR.

    Returns ``(dsr, expected_max_sharpe_annual)``. ``all_trial_sharpes`` must
    be the annualised Sharpes of *every* configuration tried, not just the
    reported one -- the whole point is to charge the result for the search.
    """
    trials = np.asarray(all_trial_sharpes, dtype=float)
    trials = trials[~np.isnan(trials)]
    n_trials = max(len(trials), 2)
    sd = trials.std(ddof=1) if len(trials) > 1 else 0.0
    if sd < 1e-12:
        return float("nan"), 0.0
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max = sd * ((1.0 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2)
    return probabilistic_sharpe_ratio(returns, expected_max, risk_free_annual), float(expected_max)


def holm_bonferroni(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Step-down Holm correction. Controls family-wise error at ``alpha``."""
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    prev_adjusted = 0.0
    still_rejecting = True
    for rank, (name, p) in enumerate(ordered):
        adjusted = min(1.0, max(prev_adjusted, (m - rank) * p))
        prev_adjusted = adjusted
        if adjusted > alpha:
            still_rejecting = False
        out[name] = {
            "p_raw": p,
            "p_holm": adjusted,
            "significant": bool(still_rejecting and adjusted <= alpha),
            "rank": rank + 1,
        }
    return out
