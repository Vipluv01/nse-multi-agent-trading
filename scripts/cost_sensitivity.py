"""Intraday-cost sensitivity check.

Every result in README.md uses the delivery cost schedule (32 bps round-trip), the
correct one for every strategy in this study, which rebalances daily and holds
overnight. This script re-runs the same, already-computed strategies under the
intraday schedule (~14 bps round-trip) purely as a labelled sensitivity check --
flagged as an open item in KNOWN_ISSUES.md -- not as a new search for a
configuration that clears the bar under a friendlier cost assumption.

Same pre-registered success criterion as every other attempt in this project: net
Sharpe CI excludes zero AND beats Buy&Hold after Holm correction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals
from nse_agents.backtest.stats import block_bootstrap_sharpe, paired_sharpe_difference
from nse_agents.config import RESULTS, CostModel
from nse_agents.report.analysis import ensemble_predictions, load_oos_predictions

OUT = RESULTS / "improvements"
pd.set_option("display.width", 200)


def main() -> int:
    delivery, intraday = CostModel(), CostModel.intraday()
    print(f"delivery round-trip: {delivery.cost_bps('buy')+delivery.cost_bps('sell'):.2f}bps  "
          f"intraday round-trip: {intraday.cost_bps('buy')+intraday.cost_bps('sell'):.2f}bps\n")

    runs = load_oos_predictions(RESULTS / "technical")
    tech = ensemble_predictions(runs, "LSTM")  # best empirical architecture
    # Intersect on tech's own date set exactly, not just a >= filter: the OOS
    # CSVs were snapshotted from an earlier price-cache pull, and a later
    # cache refresh (e.g. for the live pipeline) can add trailing days a
    # >= filter alone would not exclude, desyncing every comparison by a day.
    tech_dates = set(tech["date"].unique())

    signals = {"Tech-only (LSTM)": tech}
    for name, frame in build_baseline_signals().items():
        signals[name] = frame.loc[frame["date"].isin(tech_dates)].reset_index(drop=True)

    rows = []
    for name, frame in signals.items():
        for label, costs in (("delivery", delivery), ("intraday", intraday)):
            result = backtest_signals(frame, f"{name}_{label}", threshold=0.5, costs=costs)
            interval = block_bootstrap_sharpe(result.returns, n_boot=3000)
            rows.append({
                "strategy": name, "cost_schedule": label,
                "sharpe_net": result.performance.sharpe,
                "ci_low": interval.low, "ci_high": interval.high,
                "cagr": result.performance.cagr, "cost_drag": result.cost_drag_annual(),
                "turnover": result.performance.turnover_annual,
            })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "cost_sensitivity.csv", index=False)
    print(table.round(4).to_string(index=False))

    # Paired comparison vs Buy&Hold under intraday costs only, for whichever
    # strategy's own CI looks most promising under the friendlier schedule.
    bh_intraday = backtest_signals(signals["Buy&Hold"], "Buy&Hold_intraday", threshold=0.5, costs=intraday)
    print("\n=== vs Buy&Hold, intraday costs ===")
    clears = []
    for name, frame in signals.items():
        if name == "Buy&Hold":
            continue
        result = backtest_signals(frame, f"{name}_intraday", threshold=0.5, costs=intraday)
        interval = block_bootstrap_sharpe(result.returns, n_boot=3000)
        versus = paired_sharpe_difference(result.returns, bh_intraday.returns, n_boot=3000)
        passes = interval.low > 0 and versus.point > 0 and versus.p_value < 0.05
        print(f"  {name:20s} sharpe={result.performance.sharpe:+.3f} "
              f"[{interval.low:+.2f},{interval.high:+.2f}]  vs B&H {versus.point:+.3f} "
              f"(p={versus.p_value:.3f})  {'PASSES' if passes else ''}")
        if passes:
            clears.append(name)

    print("\nVERDICT:", "PASSES under intraday costs: " + ", ".join(clears) if clears
          else "no strategy clears the bar even under the friendlier intraday cost schedule")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
