"""Quantify what this study could and could not have found.

Every backtest here concluded "no configuration beats Buy&Hold." That claim is
incomplete without an answer to: how large would a true edge have had to be for our own
test to reliably see it, given how much data we actually had? This script answers that,
using the real Buy&Hold return series at each tested periodicity (h=1/5/20) as the
volatility/autocorrelation template, and the exact same block-bootstrap test used to
pass/fail every strategy in the study.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals, subsample_periods
from nse_agents.backtest.power import analytic_se_approximation, simulate_detection_power
from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.dataset import build_panel_dataset

OUT = RESULTS / "improvements"
pd.set_option("display.width", 200)

TARGET_SHARPES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])


def buyhold_returns(horizon: int) -> tuple[np.ndarray, float]:
    """Real Buy&Hold return series at this periodicity, restricted to the same
    OOS-era window used throughout the study, so the power analysis reflects the
    actual data available to every experiment reported here."""
    periods = 252.0 / horizon
    if horizon == 1:
        signals = build_baseline_signals()["Buy&Hold"]
        signals = signals.loc[signals["date"] >= "2018-12-07"]
    else:
        ds = build_panel_dataset(horizon=horizon, label="absolute")
        signals = pd.DataFrame({
            "date": ds.dates, "symbol": ds.symbols, "fwd_ret": ds.fwd_returns, "prob_up": 1.0,
        })
        signals = signals.loc[signals["date"] >= "2018-12-07"]
    traded = subsample_periods(signals, horizon)
    result = backtest_signals(traded, "Buy&Hold", threshold=0.5, periods_per_year=periods)
    return result.returns, periods


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = {}

    for horizon in (1, 5, 20):
        returns, periods = buyhold_returns(horizon)
        n = len(returns)
        print(f"\nhorizon={horizon}d: n={n} periods ({n/periods:.1f} years), "
              f"vol={returns.std()*np.sqrt(periods):.1%}/yr", flush=True)

        t0 = time.time()
        curve = simulate_detection_power(
            returns, TARGET_SHARPES, n_periods=n, periods_per_year=periods,
            n_sims=300, n_boot=500, block=max(5, int(20 / horizon)),
        )
        curves[horizon] = curve
        mde = curve.minimum_detectable_effect(0.80)
        analytic = analytic_se_approximation(n, periods)
        print(f"  MDE @ 80% power: {mde:.3f}  (analytic SE sanity check: {analytic:.3f}, "
              f"~2.8x that is the rule-of-thumb MDE = {2.8*analytic:.3f})", flush=True)
        for target, power in zip(curve.target_sharpes, curve.power):
            rows.append({
                "horizon_days": horizon, "n_periods": n, "years": n / periods,
                "target_sharpe": target, "power": power,
            })
        print(f"  ({time.time()-t0:.0f}s)", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "power_curves.csv", index=False)

    summary = pd.DataFrame([
        {
            "horizon_days": h,
            "n_periods": c.n_periods,
            "years": c.n_periods / c.periods_per_year,
            "mde_80pct_power": c.minimum_detectable_effect(0.80),
            "mde_50pct_power": c.minimum_detectable_effect(0.50),
        }
        for h, c in curves.items()
    ])
    summary.to_csv(OUT / "power_summary.csv", index=False)
    print("\n=== summary ===")
    print(summary.round(3).to_string(index=False))
    print(f"\nInterpretation: at the daily (h=1) horizon this study tested primarily, a\n"
          f"true annualised Sharpe below ~{summary.loc[summary.horizon_days==1,'mde_80pct_power'].iloc[0]:.2f}\n"
          f"would have been detected less than 80% of the time by this study's own test.\n"
          f"The null result here should be read as \"no edge above that size was found,\"\n"
          f"not \"there is provably no edge at all.\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
