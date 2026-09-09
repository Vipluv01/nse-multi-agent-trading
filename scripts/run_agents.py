"""Run the multi-agent system, as an ablation over which agents are present.

The ablation is the experiment. Comparing "the multi-agent system" against
baselines answers nothing on its own -- if it wins, the win could come from the
technical model, the sentiment signal, the regime filter, the debate, or the
risk layer. Each configuration below adds exactly one component, so the
contribution of each is separately visible, and every configuration runs the
identical execution and cost model.

Usage:
  .venv/bin/python scripts/run_agents.py --backend local
  .venv/bin/python scripts/run_agents.py --backend local --debate-mode disagreement
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig, build_pairs
from nse_agents.agents.regime import RegimeAgent, build_regime_table
from nse_agents.agents.researchers import format_findings, generate_transcript
from nse_agents.agents.risk import RiskLimits, RiskManager
from nse_agents.agents.sentiment import SentimentAgent
from nse_agents.agents.technical import TechnicalAgent
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals, run_backtest, summary_table
from nse_agents.config import RESULTS, SETTINGS
from nse_agents.report.analysis import (
    compare_to_benchmark,
    ensemble_predictions,
    load_oos_predictions,
    sharpe_intervals,
)

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

OUT = RESULTS / "agents"
TECH = RESULTS / "technical"


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local", choices=["local", "anthropic", "echo"])
    ap.add_argument("--architecture", default="LSTM-TAL")
    ap.add_argument("--debate-mode", default="disagreement", choices=["always", "disagreement", "never"])
    ap.add_argument("--sentiment-tag", default="local")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--transcripts", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    runs = load_oos_predictions(TECH)
    if not runs:
        print("run scripts/train_technical.py first")
        return 1
    oos = ensemble_predictions(runs, args.architecture)
    pairs = build_pairs(oos)
    print(f"decision universe: {len(pairs):,} (date, symbol) pairs, "
          f"{pairs['date'].min().date()}..{pairs['date'].max().date()}", flush=True)

    technical = TechnicalAgent(oos)

    regime_table = build_regime_table()
    regime = RegimeAgent(regime_table)
    vol_table = regime_table[["symbol", "date", "vol_20"]].copy()

    sentiment_path = RESULTS / "sentiment" / f"daily_{args.sentiment_tag}.csv"
    sentiment = None
    if sentiment_path.exists():
        daily = pd.read_csv(sentiment_path, parse_dates=["date"])
        sentiment = SentimentAgent(daily)
        overlap = pairs.merge(daily[["symbol", "date"]], on=["symbol", "date"], how="inner")
        print(f"sentiment coverage: {len(overlap):,}/{len(pairs):,} decision points "
              f"({len(overlap) / len(pairs):.1%})", flush=True)
    else:
        print(f"no sentiment file at {sentiment_path}; sentiment configs will be skipped", flush=True)

    from nse_agents.llm.factory import build_backend

    backend = build_backend(args.backend, cache=True) if args.debate_mode != "never" else None

    configurations = [
        ("Tech-only", [technical], "never"),
        ("Tech+Regime", [technical, regime], "never"),
    ]
    if sentiment is not None:
        configurations += [
            ("Tech+Sentiment", [technical, sentiment], "never"),
            ("Tech+Sent+Regime", [technical, sentiment, regime], "never"),
            ("Full+Debate", [technical, sentiment, regime], args.debate_mode),
        ]

    results, decision_frames = [], {}
    for name, agents, debate_mode in configurations:
        section(f"configuration: {name}  (agents: {', '.join(a.name for a in agents)}; "
                f"debate: {debate_mode})")
        orchestrator = Orchestrator(
            agents=agents,
            trader=Trader(TraderConfig(use_debate=debate_mode != "never"), RiskManager(RiskLimits())),
            backend=backend if debate_mode != "never" else None,
            config=OrchestratorConfig(debate_mode=debate_mode, batch_size=args.batch_size),
        )
        frame, decisions = orchestrator.run(pairs, vol_table)
        frame.to_csv(OUT / f"decisions_{name}.csv", index=False)
        decision_frames[name] = frame

        result = run_backtest(frame, name)
        results.append(result)
        perf = result.performance
        print(f"  -> Sharpe(net) {perf.sharpe:+.4f}  CAGR {perf.cagr:+.4f}  "
              f"MaxDD {perf.max_drawdown:.4f}  exposure {perf.exposure:.3f}  "
              f"turnover {perf.turnover_annual:.1f}x", flush=True)
        print(f"     actions: {frame['action'].value_counts().to_dict()}", flush=True)

    section("Baselines, on the same dates and the same cost model")
    start = pairs["date"].min()
    for name, frame in build_baseline_signals().items():
        results.append(backtest_signals(frame.loc[frame["date"] >= start], name, threshold=0.5))

    section("Summary")
    table = summary_table(results).sort_values("Sharpe(net,excess)", ascending=False)
    print(table.round(4).to_string(index=False), flush=True)
    table.to_csv(OUT / "summary.csv", index=False)

    section("Sharpe intervals and Deflated Sharpe")
    intervals = sharpe_intervals(results)
    print(intervals.round(4).to_string(index=False), flush=True)
    intervals.to_csv(OUT / "sharpe_intervals.csv", index=False)

    section("Paired comparison vs Buy&Hold (Holm-corrected)")
    comparison = compare_to_benchmark(results, "Buy&Hold")
    print(comparison.round(4).to_string(index=False), flush=True)
    comparison.to_csv(OUT / "vs_buyhold.csv", index=False)

    section("Incremental contribution of each component (paired vs Tech-only)")
    agent_results = [r for r in results if r.name in decision_frames]
    if len(agent_results) > 1:
        incremental = compare_to_benchmark(agent_results, "Tech-only")
        print(incremental.round(4).to_string(index=False), flush=True)
        incremental.to_csv(OUT / "incremental.csv", index=False)

    # ---- explainability artefact ----------------------------------------
    if backend is not None and args.transcripts > 0 and sentiment is not None:
        section(f"Sample debate transcripts ({args.transcripts})")
        agents = [technical, sentiment, regime]
        sampled = pairs.sample(n=min(args.transcripts, len(pairs)), random_state=SETTINGS.seed)
        transcripts = []
        for row in sampled.itertuples(index=False):
            opinions = [a.opine(row.symbol, row.date) for a in agents]
            findings = format_findings(opinions)
            text = generate_transcript(
                backend, row.symbol, pd.Timestamp(row.date).date().isoformat(), findings
            )
            transcripts.append(
                {
                    "date": pd.Timestamp(row.date).date().isoformat(),
                    "symbol": row.symbol,
                    "findings": findings,
                    "bull": text["bull"],
                    "bear": text["bear"],
                }
            )
            print(f"\n--- {row.symbol} {pd.Timestamp(row.date).date()} ---", flush=True)
            print(findings, flush=True)
            print(f"BULL: {text['bull'][:400]}", flush=True)
            print(f"BEAR: {text['bear'][:400]}", flush=True)
        (OUT / "transcripts.json").write_text(json.dumps(transcripts, indent=2))

    print(f"\nwrote results to {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
