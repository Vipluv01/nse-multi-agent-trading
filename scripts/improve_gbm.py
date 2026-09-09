"""Attempts E1 + E2: a gradient-boosted-trees baseline on a wider cross-sectional panel.

Pre-registered in PREREGISTRATION.md before this was run. Uses only price data (no
news), so it does not touch anything about the sentiment findings.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.ensemble import HistGradientBoostingClassifier

from nse_agents.backtest.engine import backtest_signals, subsample_periods
from nse_agents.backtest.metrics import classification_metrics
from nse_agents.backtest.stats import block_bootstrap_sharpe, paired_sharpe_difference
from nse_agents.config import RESULTS, SETTINGS, WalkForward
from nse_agents.data.dataset import build_panel_dataset
from nse_agents.models.train import walk_forward_folds

OUT = RESULTS / "improvements"
pd.set_option("display.width", 240)

# 30 additional liquid NSE large/mid caps, spanning sectors not already covered by
# SETTINGS.universe (banking, IT, energy, telecom, FMCG, cement, autos, pharma,
# metals, capital goods). Chosen for liquidity before any result was seen -- not
# selected because of how they performed.
EXTRA_UNIVERSE = (
    "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "MARUTI", "ASIANPAINT", "TITAN",
    "ULTRACEMCO", "SUNPHARMA", "WIPRO", "HCLTECH", "NESTLEIND", "POWERGRID",
    "NTPC", "ONGC", "COALINDIA", "DABUR", "TATASTEEL", "JSWSTEEL",
    "ADANIPORTS", "BAJAJFINSV", "DRREDDY", "CIPLA", "DIVISLAB", "EICHERMOT",
    "HEROMOTOCO", "BRITANNIA", "GRASIM", "SHREECEM", "INDUSINDBK", "TECHM",
)
WIDE_UNIVERSE = SETTINGS.universe + EXTRA_UNIVERSE


def flatten_for_gbm(x: np.ndarray) -> np.ndarray:
    """(N, lookback, F) -> (N, lookback*F). Trees have no notion of sequence
    order, so the last-step values and the lookback window are both handed to
    it as plain columns; it can learn to weight recent vs older steps itself."""
    return x.reshape(x.shape[0], -1)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"fetching/caching prices for {len(WIDE_UNIVERSE)} symbols "
          f"({len(EXTRA_UNIVERSE)} new)...", flush=True)
    t0 = time.time()

    combos = [
        ("E0 GBM absolute h=1, narrow (10)", "absolute", 1, SETTINGS.universe),
        ("E1 GBM absolute h=1, wide (40)", "absolute", 1, WIDE_UNIVERSE),
        ("E1b GBM cross-sectional h=1, wide (40)", "cross_sectional", 1, WIDE_UNIVERSE),
        ("E2 GBM cross-sectional h=5, wide (40)", "cross_sectional", 5, WIDE_UNIVERSE),
        ("E2b GBM cross-sectional h=20, wide (40)", "cross_sectional", 20, WIDE_UNIVERSE),
    ]

    rows = []
    for name, label, horizon, universe in combos:
        dataset = build_panel_dataset(symbols=universe, horizon=horizon, label=label)
        wf = WalkForward(train_years=3.0, test_months=6, embargo_days=max(10, horizon + 5))
        folds = walk_forward_folds(dataset.dates, wf)

        x_flat = flatten_for_gbm(dataset.x)
        oos_frames = []
        for fold in folds:
            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=4, learning_rate=0.05,
                l2_regularization=1.0, random_state=SETTINGS.seed,
                early_stopping=True, validation_fraction=0.15,
            )
            model.fit(x_flat[fold.train_idx], dataset.y[fold.train_idx])
            probs = model.predict_proba(x_flat[fold.test_idx])[:, 1]
            oos_frames.append(pd.DataFrame({
                "date": dataset.dates[fold.test_idx],
                "symbol": dataset.symbols[fold.test_idx],
                "y_true": dataset.y[fold.test_idx],
                "fwd_ret": dataset.fwd_returns[fold.test_idx],
                "prob_up": probs,
            }))
        if not oos_frames:
            print(f"  {name}: no folds produced, skipping", flush=True)
            continue
        oos = pd.concat(oos_frames, ignore_index=True)

        metrics = classification_metrics(oos["y_true"].to_numpy(), oos["prob_up"].to_numpy())
        periods = 252.0 / horizon
        traded = subsample_periods(oos, horizon)
        result = backtest_signals(traded, name, threshold=0.5, periods_per_year=periods)
        interval = block_bootstrap_sharpe(
            result.returns, n_boot=3000, block=max(5, int(20 / horizon)), periods_per_year=periods
        )

        bh_ds = build_panel_dataset(symbols=universe, horizon=horizon, label="absolute")
        bh_frame = pd.DataFrame({
            "date": bh_ds.dates, "symbol": bh_ds.symbols,
            "fwd_ret": bh_ds.fwd_returns, "prob_up": 1.0,
        })
        bh_frame = bh_frame.loc[bh_frame["date"] >= oos["date"].min()]
        bh_traded = subsample_periods(bh_frame, horizon)
        bh_result = backtest_signals(bh_traded, "Buy&Hold", threshold=0.5, periods_per_year=periods)
        versus = paired_sharpe_difference(result.returns, bh_result.returns, n_boot=3000,
                                          block=max(5, int(20 / horizon)), periods_per_year=periods)

        perf = result.performance
        rows.append({
            "variant": name, "universe_size": len(universe), "n_oos": metrics["n"],
            "accuracy": metrics["accuracy"], "auc": metrics["auc"], "mcc": metrics["mcc"],
            "periods": perf.n_days, "sharpe_net": perf.sharpe,
            "ci_low": interval.low, "ci_high": interval.high,
            "sharpe_gross": result.performance_gross.sharpe, "cagr": perf.cagr,
            "max_dd": perf.max_drawdown, "turnover": perf.turnover_annual,
            "cost_drag": result.cost_drag_annual(), "bh_sharpe": bh_result.performance.sharpe,
            "vs_bh": versus.point, "vs_bh_p": versus.p_value,
        })
        print(f"  {name}: n={metrics['n']:,} acc={metrics['accuracy']:.4f} "
              f"auc={metrics['auc']:.4f} mcc={metrics['mcc']:+.4f} "
              f"sharpe={perf.sharpe:+.3f} [{interval.low:+.2f},{interval.high:+.2f}] "
              f"vs B&H {versus.point:+.3f} (p={versus.p_value:.3f})", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "gbm_variants.csv", index=False)
    print(f"\n(total wall time {time.time()-t0:.0f}s)")
    print("\n=== all GBM / wide-universe attempts ===")
    print(table.round(4).to_string(index=False))

    clears = table[(table["ci_low"] > 0) & (table["vs_bh"] > 0) & (table["vs_bh_p"] < 0.05)]
    print("\nVERDICT:", "PASSES: " + ", ".join(clears["variant"]) if len(clears)
          else "no variant clears the pre-registered bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
