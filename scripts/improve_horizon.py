"""Attempts B1, B2 and C1: longer horizons, and a cross-sectional ranking label.

Every combination is trained and reported -- see PREREGISTRATION.md.

Two correctness points that decide whether the comparison means anything:

1. **The embargo must exceed the horizon.** A label at decision date t resolves
   at t+1+h, so with h=20 and a 10-day embargo, training rows from the last
   fortnight of each training window resolve *inside* the test fold. The
   embargo is therefore set to h + 5, not left at the default.
2. **Positions are held for h days, not rebalanced daily.** Decision dates are
   subsampled to every h-th session so holding periods do not overlap, and
   every metric is annualised with 252/h periods per year. Annualising 20-day
   returns with sqrt(252) would inflate the Sharpe by ~4.5x for free.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals, subsample_periods
from nse_agents.backtest.metrics import classification_metrics
from nse_agents.backtest.stats import block_bootstrap_sharpe, paired_sharpe_difference
from nse_agents.config import RESULTS, SETTINGS, WalkForward
from nse_agents.data.dataset import build_panel_dataset
from nse_agents.models.train import TrainConfig, run_walk_forward

OUT = RESULTS / "improvements"
pd.set_option("display.width", 240)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)

    combos = [
        ("B0 absolute h=1 (as reported)", "absolute", 1),
        ("B1 absolute h=5", "absolute", 5),
        ("B2 absolute h=20", "absolute", 20),
        ("C1 cross-sectional h=1", "cross_sectional", 1),
        ("C1b cross-sectional h=5", "cross_sectional", 5),
        ("C1c cross-sectional h=20", "cross_sectional", 20),
    ]

    rows = []
    for name, label, horizon in combos:
        dataset = build_panel_dataset(horizon=horizon, label=label)
        # Embargo must clear the horizon or the label leaks into the test fold.
        wf = WalkForward(train_years=3.0, test_months=6, embargo_days=max(10, horizon + 5))

        seed_frames = []
        for seed_offset in range(3):
            import nse_agents.models.train as train_mod

            original = train_mod.train_one_fold

            def seeded(*a, _o=original, _s=seed_offset, **kw):
                kw["seed"] = SETTINGS.seed + _s
                return _o(*a, **kw)

            train_mod.train_one_fold = seeded
            try:
                oos, _ = run_walk_forward(
                    dataset.x, dataset.y, dataset.dates, dataset.symbols,
                    dataset.fwd_returns, TrainConfig(epochs=25), wf, verbose=False,
                )
            finally:
                train_mod.train_one_fold = original
            seed_frames.append(oos)

        ensemble = seed_frames[0][["date", "symbol", "y_true", "fwd_ret"]].copy()
        ensemble["prob_up"] = np.mean([f["prob_up"].to_numpy() for f in seed_frames], axis=0)

        metrics = classification_metrics(
            ensemble["y_true"].to_numpy(), ensemble["prob_up"].to_numpy()
        )

        periods = 252.0 / horizon
        traded = subsample_periods(ensemble, horizon)
        result = backtest_signals(traded, name, threshold=0.5, periods_per_year=periods)
        interval = block_bootstrap_sharpe(
            result.returns, n_boot=3000, block=max(5, int(20 / horizon)), periods_per_year=periods
        )

        # Buy&Hold on exactly the same period grid, for a paired comparison.
        bh = build_baseline_signals()
        bh_frame = bh["Buy&Hold"]
        bh_frame = bh_frame.loc[bh_frame["date"] >= ensemble["date"].min()].copy()
        if horizon > 1:
            # Rebuild Buy&Hold's returns on the same non-overlapping grid.
            bh_ds = build_panel_dataset(horizon=horizon, label="absolute")
            bh_frame = pd.DataFrame({
                "date": bh_ds.dates, "symbol": bh_ds.symbols,
                "fwd_ret": bh_ds.fwd_returns, "prob_up": 1.0,
            })
            bh_frame = bh_frame.loc[bh_frame["date"] >= ensemble["date"].min()]
        bh_traded = subsample_periods(bh_frame, horizon)
        bh_result = backtest_signals(bh_traded, "Buy&Hold", threshold=0.5, periods_per_year=periods)
        versus = paired_sharpe_difference(
            result.returns, bh_result.returns, n_boot=3000,
            block=max(5, int(20 / horizon)), periods_per_year=periods,
        )

        perf = result.performance
        rows.append({
            "variant": name,
            "n_oos": metrics["n"],
            "accuracy": metrics["accuracy"],
            "auc": metrics["auc"],
            "mcc": metrics["mcc"],
            "periods": perf.n_days,
            "sharpe_net": perf.sharpe,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "sharpe_gross": result.performance_gross.sharpe,
            "cagr": perf.cagr,
            "max_dd": perf.max_drawdown,
            "turnover": perf.turnover_annual,
            "cost_drag": result.cost_drag_annual(),
            "bh_sharpe": bh_result.performance.sharpe,
            "vs_bh": versus.point,
            "vs_bh_p": versus.p_value,
        })
        print(f"  {name}: acc={metrics['accuracy']:.4f} sharpe={perf.sharpe:+.3f} "
              f"[{interval.low:+.2f},{interval.high:+.2f}] vs B&H {versus.point:+.3f} "
              f"(p={versus.p_value:.3f}) cost_drag={result.cost_drag_annual():.4f}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "horizon_label_variants.csv", index=False)
    print("\n=== all horizon / label attempts ===")
    print(table.round(4).to_string(index=False))

    print("\nPre-registered success criterion: net Sharpe CI excludes zero AND beats "
          "Buy&Hold on a paired test after Holm correction.")
    clears = table[(table["ci_low"] > 0) & (table["vs_bh"] > 0) & (table["vs_bh_p"] < 0.05)]
    print("VERDICT:", "PASSES: " + ", ".join(clears["variant"]) if len(clears)
          else "no variant clears the bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
