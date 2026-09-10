"""Risk attribution report: Treynor, Information Ratio, upside/downside capture, and
beta for every strategy in the main ablation, against three benchmarks (Nifty 50,
Equal-Weight Universe, Nifty Next 50).

This is a reporting pass over already-computed results -- it trains nothing, re-runs
no walk-forward, and changes no number anywhere else in the study. It answers a
question the existing Sharpe/drawdown tables don't: not just "did a strategy make
money," but "how much of that was market beta, and how much of the market's moves
(up and down) did it actually track."

Usage:  .venv/bin/python scripts/risk_attribution_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.benchmarks import align_benchmark_to_dates, build_extended_benchmarks
from nse_agents.backtest.engine import backtest_signals, run_backtest
from nse_agents.backtest.metrics import (
    beta,
    downside_capture_ratio,
    information_ratio,
    treynor_ratio,
    upside_capture_ratio,
)
from nse_agents.config import RESULTS

OUT = RESULTS / "improvements"
pd.set_option("display.width", 220)


def load_strategy_returns() -> dict:
    """Every strategy from the main agent ablation plus the classical
    baselines -- the same set summarised in results/agents/summary.csv."""
    out = {}
    agents_dir = RESULTS / "agents"
    for name in ("Tech-only", "Tech+Regime", "Tech+Sentiment", "Tech+Sent+Regime", "Full+Debate"):
        path = agents_dir / f"decisions_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
            result = run_backtest(frame, name)
            out[name] = result

    start = min(r.dates.min() for r in out.values()) if out else "2018-12-07"
    for name, frame in build_baseline_signals().items():
        window = frame.loc[frame["date"] >= start]
        out[name] = backtest_signals(window, name, threshold=0.5)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = load_strategy_returns()
    print(f"loaded {len(strategies)} strategies: {list(strategies)}", flush=True)

    benchmarks = build_extended_benchmarks()
    for name, frame in benchmarks.items():
        span = f"{frame['date'].min().date()} -> {frame['date'].max().date()}"
        print(f"  benchmark {name}: {len(frame):,} rows, {span}", flush=True)
    print(
        "  NOTE: NiftyNext50 starts 2020-01-01 (Yahoo's data floor for that ticker), "
        "~2 years short of the other two -- comparisons against it cover a shorter, "
        "non-identical window and are not directly comparable in absolute terms to "
        "the Nifty50 / EqualWeightUniverse comparisons.",
        flush=True,
    )

    rows = []
    for strat_name, result in strategies.items():
        for bench_name, bench_frame in benchmarks.items():
            aligned_bench = align_benchmark_to_dates(bench_frame, result.dates)
            # Restrict to the benchmark's own real date range -- the zero-fill
            # in align_benchmark_to_dates exists so lengths match, but including
            # zero-filled "no data" days as real observations would understate
            # NiftyNext50's true volatility and bias every ratio computed against it.
            bench_dates = set(bench_frame["date"])
            mask = [d in bench_dates for d in result.dates]
            returns = result.returns[mask]
            bench_returns = aligned_bench[mask]
            if len(returns) < 30:
                continue

            rows.append({
                "strategy": strat_name, "benchmark": bench_name, "n_days": len(returns),
                "beta": beta(returns, bench_returns),
                "treynor": treynor_ratio(returns, bench_returns),
                "information_ratio": information_ratio(returns, bench_returns),
                "upside_capture": upside_capture_ratio(returns, bench_returns),
                "downside_capture": downside_capture_ratio(returns, bench_returns),
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "risk_attribution.csv", index=False)

    for bench_name in benchmarks:
        print(f"\n=== vs {bench_name} ===", flush=True)
        sub = table[table["benchmark"] == bench_name].drop(columns="benchmark")
        print(sub.round(4).to_string(index=False), flush=True)

    print(f"\nwrote {OUT / 'risk_attribution.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
