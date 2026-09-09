"""Regime-stability robustness check.

Every result so far reports one number averaged over 7.6 years. That average can hide
a strategy that only works in one market condition -- or, the more interesting
possibility here, a risk overlay that earns its keep specifically when markets fall,
even while looking unremarkable on average.

The regime classifier is deliberately **market-derived, not strategy-derived**: it is a
mechanical, causal function of the Nifty's own price history (a trailing-60-day return
and distance from its running peak), fixed before any strategy's returns are looked at.
Defining "crash" by which days a particular strategy happened to lose money would be
circular; defining it by the benchmark's own drawdown is not.

Every regime and every strategy is reported here -- this is a diagnostic, not a new
search for a configuration that "wins" in some slice. A crash regime with ~80 days is
too short for a meaningful Sharpe estimate, so it is reported as cumulative return and
drawdown captured, not as a bootstrapped ratio that would carry false precision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals, run_backtest
from nse_agents.backtest.metrics import compute_performance, drawdown_series
from nse_agents.config import BENCHMARK, RESULTS, SETTINGS
from nse_agents.data.prices import load_prices

OUT = RESULTS / "improvements"
OOS_START = "2018-12-07"
pd.set_option("display.width", 220)


def classify_regimes(start: str = OOS_START) -> pd.DataFrame:
    """Mechanical, causal Nifty regime label per trading day.

    crash:  trailing 60-day Nifty return < -10%
    bull:   within 3% of the running all-time high
    choppy: neither
    """
    nifty = load_prices(BENCHMARK, SETTINGS.start, SETTINGS.end)[["date", "close"]]
    nifty = nifty.loc[nifty["date"] >= start].reset_index(drop=True)
    close = nifty["close"]
    ret60 = close.pct_change(60)
    peak = close.cummax()
    dist_from_peak = close / peak - 1.0
    crash = ret60 < -0.10
    bull = dist_from_peak > -0.03
    nifty["regime"] = np.select([crash, bull], ["crash", "bull"], default="choppy")
    return nifty[["date", "regime"]]


def load_all_strategy_returns() -> dict[str, pd.DataFrame]:
    """Every strategy in the final ablation, as a (date, net_return) frame."""
    out: dict[str, pd.DataFrame] = {}

    agents_dir = RESULTS / "agents"
    for name in ("Tech-only", "Tech+Regime", "Tech+Sentiment", "Tech+Sent+Regime", "Full+Debate"):
        path = agents_dir / f"decisions_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
            result = run_backtest(frame, name)
            out[name] = result.daily[["date", "net_return"]]

    start = out["Tech-only"]["date"].min() if "Tech-only" in out else OOS_START
    for name, frame in build_baseline_signals().items():
        window = frame.loc[frame["date"] >= start]
        result = backtest_signals(window, name, threshold=0.5)
        out[name] = result.daily[["date", "net_return"]]

    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    regimes = classify_regimes()
    print("regime day counts:", regimes["regime"].value_counts().to_dict(), flush=True)

    strategies = load_all_strategy_returns()
    print(f"loaded {len(strategies)} strategies: {list(strategies)}", flush=True)

    rows = []
    for name, frame in strategies.items():
        merged = frame.merge(regimes, on="date", how="inner")
        for regime, group in merged.groupby("regime"):
            returns = group["net_return"].to_numpy()
            n = len(returns)
            cum_return = float(np.prod(1.0 + returns) - 1.0)
            dd = drawdown_series(returns)
            max_dd = float(dd.min())
            hit_rate = float((returns > 0).mean())
            row = {
                "strategy": name, "regime": regime, "n_days": n,
                "cumulative_return": cum_return, "max_drawdown": max_dd, "hit_rate": hit_rate,
            }
            # Only report an annualised Sharpe where the sample is long enough that
            # the number means something (>=150 trading days, ~7 months).
            if n >= 150:
                perf = compute_performance(returns)
                row["sharpe_net"] = perf.sharpe
                row["cagr"] = perf.cagr
            else:
                row["sharpe_net"] = np.nan
                row["cagr"] = np.nan
            rows.append(row)

    table = pd.DataFrame(rows)
    order = ["Buy&Hold", "RSI(14)", "Tech+Regime", "Tech+Sent+Regime", "Full+Debate",
             "MACD", "KDJ+RSI", "Tech-only", "MeanReversion", "Tech+Sentiment"]
    table["strategy"] = pd.Categorical(table["strategy"], categories=order, ordered=True)
    table = table.sort_values(["regime", "strategy"]).reset_index(drop=True)
    table.to_csv(OUT / "regime_breakdown.csv", index=False)

    for regime in ("crash", "choppy", "bull"):
        sub = table[table["regime"] == regime].drop(columns="regime")
        n = int(sub["n_days"].iloc[0]) if len(sub) else 0
        print(f"\n=== {regime} ({n} trading days) ===")
        print(sub.round(4).to_string(index=False))

    print("\nInterpretation guide: crash-regime numbers are cumulative return and max\n"
          "drawdown only (n too small for a meaningful Sharpe). Compare each strategy's\n"
          "crash max_drawdown against Buy&Hold's -- that is the direct test of whether\n"
          "the risk overlay (drawdown brake, vol targeting) earns its keep specifically\n"
          "when markets fall, which the full-period averages cannot show.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
