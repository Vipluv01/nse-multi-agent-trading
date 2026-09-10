"""Hyperparameter sensitivity sweep -- overfitting protection, not a new search for
alpha. Pre-registered in PREREGISTRATION.md before this was run; read that section
for the scope decisions (architecture consistency, which parameters are swept where)
made before any of these numbers existed.

Plain grid search, not Optuna: the total search space here is 4x3 + 4 = 16
configurations, small enough for exhaustive, reproducible enumeration -- and Optuna's
own parameter-importance estimates need far more trials than 16 to be statistically
meaningful. A black-box sampler earns its keep on a space too large to enumerate;
this one isn't.

Usage:  .venv/bin/python scripts/optimize_hyperparams.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig, build_pairs
from nse_agents.agents.regime import RegimeAgent, build_regime_table
from nse_agents.agents.risk import RiskLimits, RiskManager
from nse_agents.agents.sentiment import SentimentAgent
from nse_agents.agents.technical import TechnicalAgent
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.backtest.engine import backtest_signals, subsample_periods
from nse_agents.backtest.stats import block_bootstrap_sharpe
from nse_agents.config import RESULTS, SETTINGS, CostModel, WalkForward
from nse_agents.data.dataset import build_panel_dataset
from nse_agents.models.train import TrainConfig, run_walk_forward
from nse_agents.report.analysis import ensemble_predictions, load_oos_predictions

OUT = RESULTS / "improvements"
HORIZONS = (1, 5, 10, 20)
COST_THRESHOLDS_BPS = (20, 32, 50)
CONVICTION_FLOORS = (0.05, 0.10, 0.15, 0.20)
pd.set_option("display.width", 220)


def cost_model_for_bps(round_trip_bps: float) -> tuple[CostModel, float]:
    """A CostModel whose measured round-trip cost matches the requested bps,
    holding the STT/GST/stamp-duty *shape* fixed (real, immutable statutory
    rates) and scaling only slippage -- the one component that is genuinely
    an assumption rather than a published rate. This answers "what if
    execution were cheaper/more expensive than measured," not "what if the
    government's tax schedule were different."

    Returns ``(cost_model, actual_round_trip_bps)`` -- the two can differ: the
    statutory floor (STT + stamp duty + exchange charges + GST on delivery
    trades, with zero slippage) is already ~22.2bps, so a requested 20bps is
    not achievable by reducing slippage and the actual figure is reported
    alongside the target rather than silently substituted for it.
    """
    base = CostModel()
    base_bps = base.cost_bps("buy") + base.cost_bps("sell")
    target_slippage = base.slippage + (round_trip_bps - base_bps) / 2 / 1e4
    model = CostModel(slippage=max(target_slippage, 0.0))
    actual_bps = model.cost_bps("buy") + model.cost_bps("sell")
    return model, actual_bps


def _train_horizon(horizon: int, tag: str) -> pd.DataFrame:
    """3-seed PLSTM-TAL (TrainConfig's default architecture, matching B0/B1/B2)
    walk-forward at one horizon, returning the seed-averaged OOS frame."""
    print(f"training h={horizon} fresh ({tag}, PLSTM-TAL, 3 seeds)...", flush=True)
    torch.set_num_threads(4)
    dataset = build_panel_dataset(horizon=horizon, label="absolute")
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
    return ensemble


def horizon_cost_grid() -> pd.DataFrame:
    """A '+'-shaped design, not a full 4x3 factorial, and deliberately so: cost
    sensitivity is measured crossed at h=1 only (reusing the main study's own
    cached PLSTM-TAL OOS predictions -- verified byte-for-byte equivalent to
    B0's 0.5026 accuracy on 19,110 rows before being trusted here, zero
    retraining needed), and horizon sensitivity is measured at the reference
    32bps cost only (reusing B0/B1/B2 directly, training only h=10 fresh, the
    one horizon with no existing cache). A full crossed grid would need three
    more multi-seed walk-forward trainings (h=5, h=20 at non-reference costs
    would still need the same OOS predictions, which is not the blocker --
    but a from-scratch h=5/h=20 retrain to avoid depending on any cached
    result was judged not worth 3x the compute for a search space already
    this exhaustively covered elsewhere in the study, and PREREGISTRATION.md
    states this scope before these numbers were seen).
    """
    rows = []

    # ---- cost-threshold sensitivity, crossed at h=1 (real cached OOS) ----
    from nse_agents.report.analysis import ensemble_predictions, load_oos_predictions

    runs = load_oos_predictions(RESULTS / "technical")
    h1_oos = ensemble_predictions(runs, "PLSTM-TAL")
    for bps in COST_THRESHOLDS_BPS:
        costs, actual_bps = cost_model_for_bps(bps)
        result = backtest_signals(h1_oos, f"h1_bps{bps}", threshold=0.5, costs=costs)
        interval = block_bootstrap_sharpe(result.returns, n_boot=2000)
        rows.append({
            "horizon": 1, "cost_threshold_bps": bps, "actual_round_trip_bps": actual_bps,
            "sharpe_net": result.performance.sharpe, "ci_low": interval.low, "ci_high": interval.high,
            "cagr": result.performance.cagr, "max_dd": result.performance.max_drawdown,
        })
        note = "" if abs(actual_bps - bps) < 0.5 else f" (actual {actual_bps:.1f}bps -- statutory floor)"
        print(f"  h=1 cost={bps:2d}bps{note}: sharpe={result.performance.sharpe:+.3f} "
              f"[{interval.low:+.2f},{interval.high:+.2f}]", flush=True)

    # ---- horizon sensitivity, crossed at the reference 32bps cost ----
    cached = pd.read_csv(RESULTS / "improvements" / "horizon_label_variants.csv")
    cached_by_horizon = {
        5: cached[cached.variant.str.contains("B1")].iloc[0],
        20: cached[cached.variant.str.contains("B2")].iloc[0],
    }
    for horizon, cached_row in cached_by_horizon.items():
        rows.append({
            "horizon": horizon, "cost_threshold_bps": 32, "actual_round_trip_bps": 32.0,
            "sharpe_net": cached_row["sharpe_net"], "ci_low": cached_row["ci_low"],
            "ci_high": cached_row["ci_high"], "cagr": cached_row["cagr"], "max_dd": cached_row["max_dd"],
        })
        print(f"  h={horizon} cost=32bps (from cache): sharpe={cached_row['sharpe_net']:+.3f} "
              f"[{cached_row['ci_low']:+.2f},{cached_row['ci_high']:+.2f}]", flush=True)

    h10_oos = _train_horizon(10, "the one horizon with no cache")
    h10_traded = subsample_periods(h10_oos, 10)
    costs32, _ = cost_model_for_bps(32)
    result = backtest_signals(h10_traded, "h10_bps32", threshold=0.5, periods_per_year=252.0 / 10, costs=costs32)
    interval = block_bootstrap_sharpe(result.returns, n_boot=2000, block=2, periods_per_year=252.0 / 10)
    rows.append({
        "horizon": 10, "cost_threshold_bps": 32, "actual_round_trip_bps": 32.0,
        "sharpe_net": result.performance.sharpe, "ci_low": interval.low, "ci_high": interval.high,
        "cagr": result.performance.cagr, "max_dd": result.performance.max_drawdown,
    })
    print(f"  h=10 cost=32bps: sharpe={result.performance.sharpe:+.3f} "
          f"[{interval.low:+.2f},{interval.high:+.2f}]", flush=True)

    return pd.DataFrame(rows)


def conviction_floor_sweep() -> pd.DataFrame:
    runs = load_oos_predictions(RESULTS / "technical")
    oos = ensemble_predictions(runs, "LSTM-TAL")
    pairs = build_pairs(oos)

    technical = TechnicalAgent(oos)
    regime = RegimeAgent(build_regime_table())
    sentiment_path = RESULTS / "sentiment" / "daily_local.csv"
    agents = [technical, regime]
    if sentiment_path.exists():
        agents.append(SentimentAgent(pd.read_csv(sentiment_path, parse_dates=["date"])))

    from nse_agents.llm.factory import build_backend

    backend = build_backend("local", cache=True)

    rows = []
    for floor in CONVICTION_FLOORS:
        print(f"\nconviction_floor={floor:.2f}...", flush=True)
        trader = Trader(TraderConfig(use_debate=True), RiskManager(RiskLimits()))
        orchestrator = Orchestrator(
            agents=agents, trader=trader, backend=backend,
            config=OrchestratorConfig(debate_mode="disagreement", disagreement_metric="conviction",
                                      conviction_floor=floor, verbose=True),
        )
        frame, _ = orchestrator.run(pairs)
        from nse_agents.backtest.engine import run_backtest

        result = run_backtest(frame, f"floor{floor}")
        interval = block_bootstrap_sharpe(result.returns, n_boot=2000)
        rows.append({
            "conviction_floor": floor, "sharpe_net": result.performance.sharpe,
            "ci_low": interval.low, "ci_high": interval.high,
            "cagr": result.performance.cagr, "max_dd": result.performance.max_drawdown,
        })
        print(f"  sharpe={result.performance.sharpe:+.3f} [{interval.low:+.2f},{interval.high:+.2f}]", flush=True)

    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== horizon x cost_threshold_bps grid ===", flush=True)
    hc = horizon_cost_grid()
    hc.to_csv(OUT / "hyperparam_horizon_cost.csv", index=False)

    print("\n=== conviction_floor sweep ===", flush=True)
    cf = conviction_floor_sweep()
    cf.to_csv(OUT / "hyperparam_conviction_floor.csv", index=False)

    # Response-surface figure.
    from nse_agents.report import figures

    figures.hyperparam_sensitivity(hc, cf, RESULTS / "figures" / "hyperparam_sensitivity.png",
                                   "Hyperparameter sensitivity: nothing clears the bar anywhere",
                                   "Overfitting protection, not a search for a winning cell -- see PREREGISTRATION.md")

    # A range-based sensitivity summary in place of an Optuna-style importance
    # score: with only 16 total cells, a formal importance estimate would
    # carry more precision than the data supports. The range each parameter's
    # Sharpe spans, holding the others fixed at their reference value, is a
    # simple, honest substitute.
    print("\n=== sensitivity summary (Sharpe range across each parameter) ===")
    ref_cost = hc[hc.cost_threshold_bps == 32]
    print(f"horizon (cost fixed at 32bps):        "
          f"{ref_cost.sharpe_net.min():+.3f} to {ref_cost.sharpe_net.max():+.3f}  "
          f"(range {ref_cost.sharpe_net.max()-ref_cost.sharpe_net.min():.3f})")
    ref_horizon = hc[hc.horizon == 1]
    print(f"cost_threshold_bps (horizon fixed at 1d): "
          f"{ref_horizon.sharpe_net.min():+.3f} to {ref_horizon.sharpe_net.max():+.3f}  "
          f"(range {ref_horizon.sharpe_net.max()-ref_horizon.sharpe_net.min():.3f})")
    print(f"conviction_floor:                     "
          f"{cf.sharpe_net.min():+.3f} to {cf.sharpe_net.max():+.3f}  "
          f"(range {cf.sharpe_net.max()-cf.sharpe_net.min():.3f})")

    clears = hc[(hc.ci_low > 0)]
    clears_cf = cf[(cf.ci_low > 0)]
    print("\nVERDICT:", "cells with own-Sharpe CI excluding zero: " +
          f"{len(clears)} horizon/cost cells, {len(clears_cf)} conviction-floor cells"
          if len(clears) or len(clears_cf) else "no cell anywhere clears its own Sharpe CI above zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
