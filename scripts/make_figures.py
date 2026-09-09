"""Render every figure from the tables the evaluation scripts wrote."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.metrics import equity_curve
from nse_agents.config import RESULTS
from nse_agents.report import figures

FIGS = RESULTS / "figures"


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    made = []

    tech = RESULTS / "technical"
    agents = RESULTS / "agents"

    # Prefer the agent run's tables when present: they cover every strategy.
    source = agents if (agents / "sharpe_intervals.csv").exists() else tech

    if (source / "sharpe_intervals.csv").exists():
        intervals = pd.read_csv(source / "sharpe_intervals.csv")
        made.append(figures.sharpe_forest(
            intervals, FIGS / "sharpe_forest.png",
            "No strategy's Sharpe is distinguishable from zero"))

    if (source / "summary.csv").exists() or (tech / "backtest_summary.csv").exists():
        path = source / "summary.csv"
        if not path.exists():
            path = tech / "backtest_summary.csv"
        summary = pd.read_csv(path)
        made.append(figures.gross_vs_net(
            summary, FIGS / "gross_vs_net_sharpe.png",
            "Transaction costs, not signal quality, decide most of these strategies"))

    if (tech / "architecture_summary.csv").exists():
        arch = pd.read_csv(tech / "architecture_summary.csv")
        made.append(figures.architecture_accuracy(
            arch, FIGS / "architecture_accuracy.png",
            "Peephole cells and temporal attention add nothing out-of-sample"))

    if (tech / "threshold_sweep.csv").exists():
        sweep = pd.read_csv(tech / "threshold_sweep.csv")
        made.append(figures.threshold_sensitivity(
            sweep, FIGS / "threshold_sensitivity.png",
            "The result is not an artefact of one decision threshold"))

    # Equity curves: rebuild from the per-strategy daily returns.
    curves: dict[str, pd.DataFrame] = {}
    for name in ("Full+Debate", "Tech+Sent+Regime", "Tech+Regime", "Tech-only"):
        path = agents / f"decisions_{name}.csv"
        if path.exists():
            from nse_agents.backtest.engine import run_backtest

            frame = pd.read_csv(path, parse_dates=["date"])
            result = run_backtest(frame, name)
            curves[name] = equity_curve(result.returns, result.dates)
            if len(curves) >= 3:
                break
    if curves:
        from nse_agents.backtest.baselines import build_baseline_signals
        from nse_agents.backtest.engine import backtest_signals

        start = min(c["date"].min() for c in curves.values())
        bh = build_baseline_signals()["Buy&Hold"]
        result = backtest_signals(bh.loc[bh["date"] >= start], "Buy&Hold")
        curves["Buy&Hold"] = equity_curve(result.returns, result.dates)
        made.append(figures.equity_curves(
            curves, FIGS / "equity_curves.png",
            "Out-of-sample equity, net of Indian transaction costs"))

    sentiment = RESULTS / "sentiment"
    for tag in ("local", "anthropic"):
        path = sentiment / f"event_buckets_{tag}.csv"
        if path.exists():
            buckets = pd.read_csv(path)
            study = pd.read_csv(sentiment / f"event_study_{tag}.csv")
            p = float(study["p_value"].iloc[0])
            verdict = ("does not predict" if p > 0.05 else "predicts")
            spread = float(buckets["mean_bps"].iloc[-1] - buckets["mean_bps"].iloc[0])
            made.append(figures.sentiment_buckets(
                buckets, FIGS / f"sentiment_buckets_{tag}.png",
                f"LLM headline sentiment {verdict} next-day returns ({tag})",
                f"Regression slope p = {p:.3f}, n = {int(study['n'].iloc[0]):,}; "
                f"positive-minus-negative spread {spread:+.1f} bps"))

    for path in made:
        print("wrote", path, flush=True)
    if not made:
        print("no result tables found; run the evaluation scripts first")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
