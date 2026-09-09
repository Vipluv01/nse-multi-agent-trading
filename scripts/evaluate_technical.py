"""Evaluate the technical arm: architecture ablation, then cost-aware backtest.

Order is deliberate -- directional accuracy first, P&L second.

Usage:  .venv/bin/python scripts/evaluate_technical.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals, random_control
from nse_agents.backtest.engine import backtest_signals, summary_table
from nse_agents.config import RESULTS
from nse_agents.report.analysis import (
    architecture_summary,
    classification_table,
    compare_to_benchmark,
    ensemble_predictions,
    load_oos_predictions,
    sharpe_intervals,
    threshold_sweep,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

OUT = RESULTS / "technical"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main() -> int:
    runs = load_oos_predictions(OUT)
    if not runs:
        print("no OOS predictions found; run scripts/train_technical.py first")
        return 1

    section("1. Directional accuracy, out-of-sample (walk-forward, purged)")
    per_run = classification_table(runs)
    print(per_run.round(4).to_string(index=False))
    per_run.to_csv(OUT / "classification_per_run.csv", index=False)

    section("2. Architecture ablation (mean over seeds, tested against chance)")
    arch = architecture_summary(per_run)
    print(arch.round(4).to_string(index=False))
    arch.to_csv(OUT / "architecture_summary.csv", index=False)

    best_arch = arch.iloc[0]["architecture"]
    print(f"\nbest architecture by accuracy: {best_arch}")

    section("3. Cost-aware backtest vs classical baselines")
    ensembles = {}
    results = []
    for architecture in arch["architecture"]:
        signals = ensemble_predictions(runs, architecture)
        ensembles[architecture] = signals
        results.append(backtest_signals(signals, architecture, threshold=0.5))

    start = min(s["date"].min() for s in ensembles.values())
    baselines = build_baseline_signals()
    for name, frame in baselines.items():
        window = frame.loc[frame["date"] >= start]
        results.append(backtest_signals(window, name, threshold=0.5))

    # Random control: same number of names held each day as the learned model,
    # chosen at random. Isolates stock selection from exposure and turnover.
    results.append(
        backtest_signals(random_control(ensembles[best_arch]), "RandomControl", threshold=0.5)
    )

    table = summary_table(results).sort_values("Sharpe(net,excess)", ascending=False)
    print(table.round(4).to_string(index=False))
    table.to_csv(OUT / "backtest_summary.csv", index=False)

    section("4. Sharpe confidence intervals and Deflated Sharpe")
    intervals = sharpe_intervals(results)
    print(intervals.round(4).to_string(index=False))
    intervals.to_csv(OUT / "sharpe_intervals.csv", index=False)

    section("5. Paired comparison vs Buy&Hold (Holm-corrected)")
    comparison = compare_to_benchmark(results, "Buy&Hold")
    print(comparison.round(4).to_string(index=False))
    comparison.to_csv(OUT / "vs_buyhold.csv", index=False)

    section(f"6. Decision-threshold sensitivity ({best_arch})")
    sweep = threshold_sweep(ensembles[best_arch], best_arch)
    print(sweep.round(4).to_string(index=False))
    sweep.to_csv(OUT / "threshold_sweep.csv", index=False)

    print("\nwrote tables to", OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
