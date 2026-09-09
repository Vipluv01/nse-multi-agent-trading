"""Turning raw runs into the tables the thesis reports.

The ordering enforced here is the point: **classification quality is examined
before any P&L is computed, and P&L is only ever reported net of costs.** A
directional model that is at chance out-of-sample cannot have a real edge, and
seeing an attractive equity curve before checking that is how a researcher
talks themselves into a curve that is really a market beta plus a lucky
sequence of turnover.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestResult, align_results, backtest_signals, summary_table
from ..backtest.metrics import classification_metrics
from ..backtest.stats import (
    block_bootstrap_sharpe,
    deflated_sharpe_ratio,
    holm_bonferroni,
    paired_sharpe_difference,
)


def load_oos_predictions(directory: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted(Path(directory).glob("oos_*.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        out[path.stem.replace("oos_", "")] = frame
    return out


def classification_table(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-run directional metrics, in the PLSTM-TAL benchmark's vocabulary."""
    rows = []
    for name, frame in runs.items():
        metrics = classification_metrics(frame["y_true"].to_numpy(), frame["prob_up"].to_numpy())
        metrics["run"] = name
        metrics["architecture"] = name.rsplit("_seed", 1)[0]
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    return frame[
        ["run", "architecture", "n", "accuracy", "precision", "recall", "f1", "mcc", "auc",
         "predicted_up_rate", "actual_up_rate"]
    ]


def architecture_summary(classification: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread across seeds, with a t-test against 50% accuracy.

    Reported across seeds rather than for a single run because a 1-seed
    architecture comparison at this effect size is noise: the seed-to-seed
    spread here is comparable to the between-architecture spread, which is
    itself the finding.
    """
    from scipy import stats as sps

    rows = []
    for arch, group in classification.groupby("architecture"):
        acc = group["accuracy"].to_numpy()
        n_obs = int(group["n"].iloc[0])
        # Binomial test of the pooled accuracy against chance.
        mean_acc = float(acc.mean())
        se = np.sqrt(0.25 / n_obs)
        z = (mean_acc - 0.5) / se
        rows.append(
            {
                "architecture": arch,
                "seeds": len(acc),
                "accuracy_mean": mean_acc,
                "accuracy_std": float(acc.std(ddof=1)) if len(acc) > 1 else 0.0,
                "auc_mean": float(group["auc"].mean()),
                "mcc_mean": float(group["mcc"].mean()),
                "z_vs_chance": float(z),
                "p_vs_chance": float(2 * (1 - sps.norm.cdf(abs(z)))),
            }
        )
    return pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)


def ensemble_predictions(runs: dict[str, pd.DataFrame], architecture: str) -> pd.DataFrame:
    """Average ``prob_up`` across seeds of one architecture.

    Seed-averaging is the fairest single representative of an architecture:
    picking the best seed would be selecting on the test set.
    """
    frames = [f for name, f in runs.items() if name.rsplit("_seed", 1)[0] == architecture]
    if not frames:
        raise KeyError(architecture)
    base = frames[0][["date", "symbol", "y_true", "fwd_ret", "fold"]].copy()
    probs = np.mean([f["prob_up"].to_numpy() for f in frames], axis=0)
    base["prob_up"] = probs
    if "attn_recent5" in frames[0]:
        base["attn_recent5"] = np.mean([f["attn_recent5"].to_numpy() for f in frames], axis=0)
    return base


def compare_to_benchmark(
    results: list[BacktestResult], benchmark_name: str, alpha: float = 0.05
) -> pd.DataFrame:
    """Paired Sharpe differences vs one benchmark, Holm-corrected."""
    aligned = align_results(results)
    bench = aligned[benchmark_name].to_numpy()

    intervals, p_values = {}, {}
    for res in results:
        if res.name == benchmark_name:
            continue
        interval = paired_sharpe_difference(aligned[res.name].to_numpy(), bench)
        intervals[res.name] = interval
        p_values[res.name] = interval.p_value if interval.p_value is not None else 1.0

    corrected = holm_bonferroni(p_values, alpha)
    rows = []
    for name, interval in intervals.items():
        info = corrected[name]
        rows.append(
            {
                "strategy": name,
                "vs": benchmark_name,
                "d_sharpe": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "p_raw": info["p_raw"],
                "p_holm": info["p_holm"],
                "significant": info["significant"],
            }
        )
    return pd.DataFrame(rows).sort_values("d_sharpe", ascending=False).reset_index(drop=True)


def sharpe_intervals(results: list[BacktestResult]) -> pd.DataFrame:
    rows = []
    all_sharpes = [r.performance.sharpe for r in results]
    for res in results:
        interval = block_bootstrap_sharpe(res.returns)
        dsr, expected_max = deflated_sharpe_ratio(res.returns, np.asarray(all_sharpes))
        rows.append(
            {
                "strategy": res.name,
                "sharpe": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
                "p_sharpe_gt_0": interval.p_value,
                "deflated_sharpe": dsr,
                "expected_max_sharpe_from_search": expected_max,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def threshold_sweep(
    signals: pd.DataFrame, name: str, thresholds=(0.50, 0.52, 0.54, 0.56, 0.58, 0.60)
) -> pd.DataFrame:
    """Sensitivity of the result to the decision threshold.

    Included because a single hand-picked threshold is the easiest place to
    hide an overfit. If the conclusion only holds at one value, it is not a
    conclusion.
    """
    rows = []
    for threshold in thresholds:
        res = backtest_signals(signals, f"{name}@{threshold:.2f}", threshold=threshold)
        perf = res.performance
        rows.append(
            {
                "threshold": threshold,
                "sharpe_net": perf.sharpe,
                "sharpe_gross": res.performance_gross.sharpe,
                "cagr": perf.cagr,
                "max_dd": perf.max_drawdown,
                "exposure": perf.exposure,
                "turnover": perf.turnover_annual,
                "cost_drag": res.cost_drag_annual(),
            }
        )
    return pd.DataFrame(rows)
