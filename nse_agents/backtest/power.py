"""Statistical power analysis for the Sharpe-ratio tests used throughout this study.

Answers a question the null results alone cannot: **how large would a real edge have
had to be for this study's own test to find it?** "No configuration beat Buy&Hold" is a
much weaker claim than "no configuration with a true annualised Sharpe above X beat
Buy&Hold" -- the second is falsifiable and precise, the first invites "did you even look
hard enough."

Method: Monte Carlo, using the **exact same test** (`block_bootstrap_sharpe`, "own
Sharpe's 95% CI excludes zero") that decided every pass/fail verdict elsewhere in this
project, rather than a closed-form approximation that might not match what the study
actually did.

For each candidate true Sharpe, synthetic return series are built by block-resampling the
*demeaned* residuals of a real, representative return series (preserving its volatility,
autocorrelation and fat tails) and injecting a synthetic mean sized to hit that Sharpe
exactly. Detection rate across many simulated series at each candidate Sharpe is the
test's statistical power at that effect size. This deliberately reuses the block-bootstrap
machinery rather than a Gaussian closed-form, because financial returns are not Gaussian
and this study's own test does not assume they are.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import TRADING_DAYS, RISK_FREE_ANNUAL
from .stats import _block_indices, block_bootstrap_sharpe


@dataclass
class PowerCurve:
    target_sharpes: np.ndarray
    power: np.ndarray          # fraction of simulations detected at each target
    n_periods: int
    periods_per_year: float
    n_sims: int

    def minimum_detectable_effect(self, target_power: float = 0.80) -> float:
        """Linear interpolation for the Sharpe at which power first reaches ``target_power``.

        Returns ``inf`` if power never reaches the target across the tested grid --
        meaning the grid needs to be widened, not that no such Sharpe exists.
        """
        if self.power[-1] < target_power:
            return float("inf")
        return float(np.interp(target_power, self.power, self.target_sharpes))


def simulate_detection_power(
    reference_returns: np.ndarray,
    target_sharpes: np.ndarray,
    n_periods: int,
    periods_per_year: float = TRADING_DAYS,
    risk_free_annual: float = RISK_FREE_ANNUAL,
    n_sims: int = 200,
    n_boot: int = 500,
    block: int = 20,
    alpha: float = 0.05,
    seed: int = 20260909,
) -> PowerCurve:
    """Power of ``block_bootstrap_sharpe`` (CI excludes zero) at each target Sharpe.

    ``reference_returns`` supplies the realistic volatility/autocorrelation/fat-tail
    structure (e.g. this study's own Buy&Hold return series at the relevant
    periodicity); only its *demeaned residual* is resampled, so the synthetic series'
    true Sharpe is controlled exactly by the injected mean, not by the reference
    series' own historical drift.
    """
    reference_returns = np.asarray(reference_returns, dtype=float)
    reference_returns = reference_returns[~np.isnan(reference_returns)]
    residual = reference_returns - reference_returns.mean()
    period_vol = residual.std(ddof=1)
    period_rf = (1.0 + risk_free_annual) ** (1.0 / periods_per_year) - 1.0

    rng = np.random.default_rng(seed)
    power = np.empty(len(target_sharpes))

    for i, target in enumerate(target_sharpes):
        # The per-period mean that gives this reference volatility exactly the
        # requested annualised Sharpe in excess of the risk-free rate.
        period_mean = period_rf + target * period_vol / np.sqrt(periods_per_year)

        detections = 0
        for _ in range(n_sims):
            idx = _block_indices(n_periods, min(block, len(residual) // 2 or 1), rng)
            # idx indexes into `residual`, which may be shorter or longer than
            # n_periods; _block_indices already returns exactly n_periods draws.
            resampled = residual[idx % len(residual)]
            synthetic = period_mean + resampled
            interval = block_bootstrap_sharpe(
                synthetic, n_boot=n_boot, block=block,
                risk_free_annual=risk_free_annual, seed=int(rng.integers(0, 2**31 - 1)),
                alpha=alpha, periods_per_year=periods_per_year,
            )
            if interval.low > 0:
                detections += 1
        power[i] = detections / n_sims

    return PowerCurve(
        target_sharpes=np.asarray(target_sharpes, dtype=float),
        power=power,
        n_periods=n_periods,
        periods_per_year=periods_per_year,
        n_sims=n_sims,
    )


def analytic_se_approximation(n_periods: int, periods_per_year: float) -> float:
    """Closed-form asymptotic SE of an annualised Sharpe under i.i.d. Gaussian returns.

    Reported alongside the simulation as a well-known sanity check (Lo, 2002), NOT as
    the study's actual test -- real returns are autocorrelated and fat-tailed, which
    the simulation captures and this formula does not. Expressed in the common
    "roughly 1/sqrt(years)" form: SE(SR_annual) ~ sqrt(1 + SR^2/2) / sqrt(years),
    evaluated at SR=0 since we want the null-hypothesis behaviour of the test.
    """
    years = n_periods / periods_per_year
    return float(1.0 / np.sqrt(years))
