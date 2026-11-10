"""Final-evaluation ablation: the mid-semester system plus everything in the November plan.

Two experiments, each one identical execution + cost model across configurations:

* ``core``     -- the mid-semester 10 names and period, so every new component
                  (fundamentals agent, cost-aware sizing, no-trade band, larger
                  LLM, conviction-gated debate) is measured against exactly the
                  numbers in the mid-semester report.
* ``extended`` -- 30 names, prices to Nov 2026, freshly re-trained technical
                  agent. No news corpus exists for the 20 added names, so
                  sentiment/debate configurations are core-only; that is a scope
                  limit, stated, not a silent omission.

Pre-registered choices (fixed before any result was seen, never swept to the answer):
  rebalance band 0.05 (one quarter of the 20% per-name cap); conviction floor
  0.10 (the mid-semester gate); cost-aware edge test as already implemented.

Writes results/final/<experiment>/{summary,sharpe_intervals,vs_buyhold,incremental,
subperiods}.csv and decisions_<config>.csv.

Usage:
  .venv/bin/python scripts/final_run_ablation.py --experiment core
  .venv/bin/python scripts/final_run_ablation.py --experiment core --debate      # needs the LLM (MPS)
  .venv/bin/python scripts/final_run_ablation.py --experiment extended
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("--experiment", choices=["core", "extended"], default="core")
ap.add_argument("--debate", action="store_true", help="include the LLM debate configurations")
ap.add_argument("--band", type=float, default=0.05)
ap.add_argument("--architecture", default="LSTM-TAL")
ARGS = ap.parse_args()

from nse_agents.final.config import (  # noqa: E402
    EXT_END, EXT_START, EXT_UNIVERSE, FINAL, use_extended_cache,
)

if ARGS.experiment == "extended":
    use_extended_cache()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig, build_pairs  # noqa: E402
from nse_agents.agents.regime import RegimeAgent, build_regime_table  # noqa: E402
from nse_agents.agents.risk import RiskLimits, RiskManager  # noqa: E402
from nse_agents.agents.sentiment import SentimentAgent  # noqa: E402
from nse_agents.agents.technical import TechnicalAgent  # noqa: E402
from nse_agents.agents.trader import Trader, TraderConfig  # noqa: E402
from nse_agents.backtest.baselines import build_baseline_signals  # noqa: E402
from nse_agents.backtest.engine import backtest_signals, run_backtest, summary_table  # noqa: E402
from nse_agents.config import RESULTS, SETTINGS  # noqa: E402
from nse_agents.final.fundamentals_data import build_agent  # noqa: E402
from nse_agents.report.analysis import (  # noqa: E402
    compare_to_benchmark, ensemble_predictions, load_oos_predictions, sharpe_intervals,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

SUBPERIODS = {
    "COVID crash (2020-03..2020-06)": ("2020-03-01", "2020-06-30"),
    "2022 rate-hike drawdown (2022-01..2022-06)": ("2022-01-01", "2022-06-30"),
    "post-mid-sem holdout (2026-09..2026-11)": ("2026-09-01", "2026-11-07"),
}


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def subperiod_table(results) -> pd.DataFrame:
    rows = []
    for res in results:
        daily = res.daily.copy()
        daily["date"] = pd.to_datetime(daily["date"])
        for label, (a, b) in SUBPERIODS.items():
            part = daily[(daily["date"] >= a) & (daily["date"] <= b)]
            if len(part) < 5:
                continue
            curve = (1.0 + part["net_return"]).cumprod()
            rows.append({
                "strategy": res.name, "period": label, "days": len(part),
                "total_return": float(curve.iloc[-1] - 1.0),
                "max_drawdown": float((curve / curve.cummax() - 1.0).min()),
                "ann_vol": float(part["net_return"].std() * np.sqrt(252)),
            })
    return pd.DataFrame(rows)


def main() -> int:
    exp = ARGS.experiment
    out = FINAL / exp
    out.mkdir(parents=True, exist_ok=True)

    if exp == "core":
        symbols, start, end = SETTINGS.universe, SETTINGS.start, SETTINGS.end
        tech_dir = RESULTS / "technical"
    else:
        symbols, start, end = EXT_UNIVERSE, EXT_START, EXT_END
        tech_dir = FINAL / "extended" / "technical"

    oos = ensemble_predictions(load_oos_predictions(tech_dir), ARGS.architecture)
    pairs = build_pairs(oos)
    pairs = pairs[pairs["symbol"].isin(symbols)].reset_index(drop=True)
    print(f"[{exp}] {len(pairs):,} decisions, {pairs['symbol'].nunique()} names, "
          f"{pairs['date'].min().date()}..{pairs['date'].max().date()}", flush=True)

    technical = TechnicalAgent(oos, architecture=ARGS.architecture)
    regime_table = build_regime_table(symbols, start, end)
    regime = RegimeAgent(regime_table)
    vol_table = regime_table[["symbol", "date", "vol_20"]].copy()
    fundamentals, _ = build_agent(symbols, pairs["date"].unique())

    sentiments = {}
    if exp == "core":
        for tag, path in (("1.5B", RESULTS / "sentiment" / "daily_local.csv"),
                          ("3B", FINAL / "daily_qwen3b.csv")):
            if path.exists():
                sentiments[tag] = SentimentAgent(pd.read_csv(path, parse_dates=["date"]))
        print("sentiment models available:", list(sentiments), flush=True)

    plain = RiskLimits()
    aware = RiskLimits(cost_aware=True)
    band = ARGS.band
    # (name, agents, debate_mode, risk_limits, band)
    configs = [
        ("Tech-only", [technical], "never", plain, 0.0),
        ("Tech+Regime", [technical, regime], "never", plain, 0.0),
        ("Tech+Regime+Fund", [technical, regime, fundamentals], "never", plain, 0.0),
        ("Regime+Fund", [regime, fundamentals], "never", plain, 0.0),
        ("Regime+Fund+Band", [regime, fundamentals], "never", plain, band),
        ("Tech+Regime+Fund+CostAware+Band", [technical, regime, fundamentals], "never", aware, band),
    ]
    if "1.5B" in sentiments:
        s = sentiments["1.5B"]
        configs += [
            ("Tech+Sent+Regime", [technical, s, regime], "never", plain, 0.0),
            ("Tech+Sent+Regime+Fund", [technical, s, regime, fundamentals], "never", plain, 0.0),
            ("All4+CostAware+Band", [technical, s, regime, fundamentals], "never", aware, band),
        ]
    if "3B" in sentiments:
        s3 = sentiments["3B"]
        configs += [
            ("Tech+Sent3B+Regime", [technical, s3, regime], "never", plain, 0.0),
            ("Tech+Sent3B+Regime+Fund", [technical, s3, regime, fundamentals], "never", plain, 0.0),
        ]

    backend = None
    if ARGS.debate and "1.5B" in sentiments:
        from nse_agents.llm.factory import build_backend

        backend = build_backend("local", cache=True)
        s = sentiments["1.5B"]
        configs += [
            ("Full+Debate(conviction)", [technical, s, regime], "disagreement", plain, 0.0),
            ("All4+Debate(conviction)", [technical, s, regime, fundamentals], "disagreement", plain, 0.0),
        ]

    results, frames, gate_stats = [], {}, {}
    for name, agents, debate_mode, limits, band_value in configs:
        section(f"{name}: agents={','.join(a.name for a in agents)} debate={debate_mode} "
                f"cost_aware={limits.cost_aware} band={band_value}")
        orchestrator = Orchestrator(
            agents=agents,
            trader=Trader(TraderConfig(use_debate=debate_mode != "never"), RiskManager(limits)),
            backend=backend if debate_mode != "never" else None,
            config=OrchestratorConfig(
                debate_mode=debate_mode, disagreement_metric="conviction", conviction_floor=0.10,
                rebalance_band=band_value,
            ),
        )
        frame, _ = orchestrator.run(pairs, vol_table)
        frame.to_csv(out / f"decisions_{name}.csv", index=False)
        frames[name] = frame
        if debate_mode != "never":
            gate_stats[name] = float(frame["debated"].mean())
        result = run_backtest(frame, name)
        results.append(result)
        p = result.performance
        print(f"  -> Sharpe(net) {p.sharpe:+.3f}  gross {result.performance_gross.sharpe:+.3f}  "
              f"CAGR {p.cagr:+.3f}  MaxDD {p.max_drawdown:.3f}  turnover {p.turnover_annual:.1f}x/yr", flush=True)

    section("Baselines (same dates, same costs)")
    first = pairs["date"].min()
    last = pairs["date"].max()
    # Every strategy trades on the *identical* return series: the decision
    # universe's own forward returns. Baselines only contribute their signal.
    study_returns = pairs[["date", "symbol", "fwd_ret"]]
    for name, frame in build_baseline_signals(symbols, start, end).items():
        frame = frame.drop(columns="fwd_ret").merge(study_returns, on=["date", "symbol"], how="inner")
        results.append(backtest_signals(frame, name, threshold=0.5))

    section("Summary")
    table = summary_table(results).sort_values("Sharpe(net,excess)", ascending=False)
    print(table.round(4).to_string(index=False), flush=True)
    table.to_csv(out / "summary.csv", index=False)

    intervals = sharpe_intervals(results)
    intervals.to_csv(out / "sharpe_intervals.csv", index=False)
    section("Sharpe CIs + Deflated Sharpe")
    print(intervals.round(4).to_string(index=False), flush=True)

    comparison = compare_to_benchmark(results, "Buy&Hold")
    comparison.to_csv(out / "vs_buyhold.csv", index=False)
    section("vs Buy&Hold (Holm-corrected)")
    print(comparison.round(4).to_string(index=False), flush=True)

    agent_results = [r for r in results if r.name in frames]
    incremental = compare_to_benchmark(agent_results, "Tech-only")
    incremental.to_csv(out / "incremental.csv", index=False)

    sub = subperiod_table(results)
    sub.to_csv(out / "subperiods.csv", index=False)
    section("Sub-periods")
    print(sub.round(4).to_string(index=False), flush=True)

    # Equity curves, for the figures/report.
    curves = pd.DataFrame({"date": pd.to_datetime(results[0].daily["date"])})
    for res in results:
        d = res.daily[["date", "net_return"]].copy()
        d["date"] = pd.to_datetime(d["date"])
        curves = curves.merge(d.rename(columns={"net_return": res.name}), on="date", how="outer")
    curves.sort_values("date").to_csv(out / "daily_net_returns.csv", index=False)

    meta = {"experiment": exp, "n_decisions": len(pairs), "symbols": list(symbols),
            "first": str(first.date()), "last": str(last.date()), "band": band,
            "architecture": ARGS.architecture, "debate_escalation_rate": gate_stats}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
