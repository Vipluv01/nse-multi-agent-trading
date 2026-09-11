"""Factor decomposition of strategy returns: how much of a strategy's return is
market beta and price momentum, versus a residual (alpha) that those two
observable, replicable exposures do not explain.

**This is a two-factor model (market, momentum), not the four-factor Fama-French
model the name "Fama-French" usually implies, and that is a deliberate,
documented gap rather than an oversight.** SMB (size) and HML (value) are
defined by a periodic cross-sectional sort on *point-in-time* market
capitalisation and book/earnings yield -- data this project has already
established it does not have and will not fake. `nse_agents/agents/regime.py`'s
own module docstring makes the identical argument for why there is a
`RegimeAgent` and not a fundamental agent: substituting *today's* market cap or
book value into a 2019 sort would be lookahead of the worst kind, since it
embeds every subsequent share issuance and earnings surprise between then and
now. Rather than fabricate SMB/HML from current-snapshot data mislabelled as
historical, `regress_factors` runs the regression on the two factors that
genuinely can be built causally from this project's own price history, and
`UNAVAILABLE_FACTORS` documents why the other two are absent so a reader does
not mistake the omission for an oversight.

Both implemented factors are built strictly causally:

* **Market excess return** (``Rm - Rf``): the Nifty 50's own open-to-open
  forward return (this codebase's one correct return convention -- see
  ``nse_agents/data/prices.py``'s ``forward_return`` and the benchmark-
  construction bug documented in ``nse_agents/backtest/benchmarks.py``), minus
  the per-period risk-free rate.
* **Momentum** (``WML``, Winners Minus Losers): the classic Jegadeesh-Titman
  construction -- rank this study's own 10-name universe by trailing 12-month
  return *skipping the most recent month* (avoiding short-term reversal
  contamination), go long the top three and short the bottom three, equal
  weighted, and read next period's return off exactly the same
  ``forward_return`` series everything else in this project uses. The ranking
  on day *t* uses only prices up to and including day *t*; the forward return
  it is compared against is attributed to the position opened the next session
  -- no lookahead.

With only 10 names, ``WML`` here is a small-portfolio momentum factor specific
to this study's own universe, not the market-wide Fama-French momentum factor
computed over thousands of listed names -- reported and labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.stats as sps

from ..config import BENCHMARK, SETTINGS
from ..data.prices import forward_return, load_prices
from .metrics import RISK_FREE_ANNUAL, TRADING_DAYS

MOMENTUM_LOOKBACK_DAYS = 252
MOMENTUM_SKIP_DAYS = 21
MOMENTUM_LEG_SIZE = 3  # out of a 10-name universe

# Why these two Fama-French factors are not implemented -- read before adding
# them "for completeness": both require a periodic cross-sectional sort on
# point-in-time fundamentals this project does not have and has never fetched
# (see the module docstring). A caller wanting these must supply their own
# point-in-time size/value data; this module will not approximate them from a
# current-snapshot query mislabelled as historical.
UNAVAILABLE_FACTORS: dict[str, str] = {
    "smb": "requires point-in-time market capitalisation history for a periodic "
           "size sort; only a current snapshot is available via this project's "
           "data source, and substituting it into a historical sort is lookahead "
           "(the same reasoning nse_agents/agents/regime.py gives for why there "
           "is no fundamental agent).",
    "hml": "requires point-in-time book value or earnings-yield history for a "
           "periodic value sort; not available from this project's price-only "
           "data pipeline, for the same reason as smb above.",
}


def market_excess_factor(
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
    risk_free_annual: float = RISK_FREE_ANNUAL,
) -> pd.DataFrame:
    """(date, mkt_excess): the Nifty 50's open-to-open forward return minus the
    per-period risk-free rate, exactly the convention every other return
    series in this project uses -- never a naive close-to-close pct_change."""
    nifty = load_prices(BENCHMARK, start, end)
    nifty["fwd_ret"] = forward_return(nifty, horizon=SETTINGS.horizon_days)
    period_rf = (1.0 + risk_free_annual) ** (1.0 / TRADING_DAYS) - 1.0
    nifty["mkt_excess"] = nifty["fwd_ret"] - period_rf
    return nifty[["date", "mkt_excess"]].dropna().reset_index(drop=True)


def _momentum_scores(symbols: tuple[str, ...], start: str, end: str) -> pd.DataFrame:
    """Per (symbol, date): trailing 12-month return skipping the most recent
    month, using only close prices up to and including that date."""
    frames = []
    for symbol in symbols:
        frame = load_prices(symbol, start, end)[["date", "open", "close"]].copy()
        frame["mom_score"] = (
            frame["close"].shift(MOMENTUM_SKIP_DAYS) / frame["close"].shift(MOMENTUM_LOOKBACK_DAYS) - 1.0
        )
        frame["symbol"] = symbol
        frame["fwd_ret"] = forward_return(frame, horizon=SETTINGS.horizon_days)
        frames.append(frame[["date", "symbol", "mom_score", "fwd_ret"]])
    return pd.concat(frames, ignore_index=True)


def momentum_factor(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
    leg_size: int = MOMENTUM_LEG_SIZE,
) -> pd.DataFrame:
    """(date, wml): equal-weight top-``leg_size`` minus bottom-``leg_size`` by
    trailing momentum score, within this study's own universe. A day with
    fewer than ``2 * leg_size`` names carrying a valid score (insufficient
    history) is dropped rather than computed on a smaller, silently
    inconsistent leg size.
    """
    scored = _momentum_scores(symbols, start, end)
    rows = []
    for date, group in scored.groupby("date"):
        valid = group.dropna(subset=["mom_score", "fwd_ret"])
        if len(valid) < leg_size * 2:
            continue
        ranked = valid.sort_values("mom_score")
        losers = ranked.iloc[:leg_size]
        winners = ranked.iloc[-leg_size:]
        rows.append({"date": date, "wml": float(winners["fwd_ret"].mean() - losers["fwd_ret"].mean())})
    return pd.DataFrame(rows)


def build_factors(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
) -> pd.DataFrame:
    """(date, mkt_excess, wml), inner-joined -- only days where both factors
    are defined."""
    market = market_excess_factor(start, end)
    wml = momentum_factor(symbols, start, end)
    return market.merge(wml, on="date", how="inner")


@dataclass(frozen=True)
class FactorRegressionResult:
    n_days: int
    alpha_daily: float
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    beta_pvalues: dict[str, float]
    r_squared: float
    factors_used: tuple[str, ...] = field(default_factory=tuple)
    factors_unavailable: dict[str, str] = field(default_factory=lambda: dict(UNAVAILABLE_FACTORS))

    def to_dict(self) -> dict:
        return {
            "n_days": self.n_days,
            "alpha_daily": self.alpha_daily,
            "alpha_annualized": self.alpha_annualized,
            "alpha_tstat": self.alpha_tstat,
            "alpha_pvalue": self.alpha_pvalue,
            "betas": dict(self.betas),
            "beta_tstats": dict(self.beta_tstats),
            "beta_pvalues": dict(self.beta_pvalues),
            "r_squared": self.r_squared,
            "factors_used": list(self.factors_used),
            "factors_unavailable": dict(self.factors_unavailable),
        }


def regress_factors(
    returns: np.ndarray,
    dates: pd.DatetimeIndex,
    factors: pd.DataFrame,
    factor_columns: tuple[str, ...] = ("mkt_excess", "wml"),
) -> FactorRegressionResult:
    """OLS: strategy return = alpha + sum(beta_k * factor_k) + residual.

    Standard errors come from the classical OLS variance formula
    (``sigma^2 * (X'X)^-1``) rather than a library, consistent with this
    project's own hand-rolled statistical machinery elsewhere
    (``nse_agents/backtest/stats.py``'s bootstrap CIs, the hand-written PLSTM
    cell) -- no new dependency for a computation this well-posed to do by hand.
    """
    merged = pd.DataFrame({"date": pd.DatetimeIndex(dates), "ret": np.asarray(returns, dtype=float)})
    merged = merged.merge(factors, on="date", how="inner").dropna(
        subset=["ret", *factor_columns]
    )
    n = len(merged)
    k = len(factor_columns) + 1  # + intercept
    if n <= k:
        raise ValueError(f"only {n} overlapping days for {k - 1} factors -- too few for OLS")

    y = merged["ret"].to_numpy()
    X = np.column_stack([np.ones(n), *(merged[c].to_numpy() for c in factor_columns)])

    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta_hat
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(sigma2 * xtx_inv), 0.0, None))
    tstats = np.divide(beta_hat, se, out=np.zeros_like(beta_hat), where=se > 1e-15)
    pvalues = 2.0 * (1.0 - sps.t.cdf(np.abs(tstats), dof))

    total_ss = float(((y - y.mean()) ** 2).sum())
    resid_ss = float((resid ** 2).sum())
    r_squared = 1.0 - resid_ss / total_ss if total_ss > 1e-15 else 0.0

    alpha_daily = float(beta_hat[0])
    alpha_annualized = (1.0 + alpha_daily) ** TRADING_DAYS - 1.0

    return FactorRegressionResult(
        n_days=n,
        alpha_daily=alpha_daily,
        alpha_annualized=alpha_annualized,
        alpha_tstat=float(tstats[0]),
        alpha_pvalue=float(pvalues[0]),
        betas={c: float(b) for c, b in zip(factor_columns, beta_hat[1:])},
        beta_tstats={c: float(t) for c, t in zip(factor_columns, tstats[1:])},
        beta_pvalues={c: float(p) for c, p in zip(factor_columns, pvalues[1:])},
        r_squared=r_squared,
        factors_used=factor_columns,
    )
