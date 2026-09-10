"""Run the full agent pipeline on today's real market and news data.

**This is a demonstration of the pipeline running live, not a trading recommendation
and not a new evaluation.** Every accuracy and Sharpe number in this repository comes
from the out-of-sample walk-forward study; this script's model has no held-out
performance of its own (see nse_agents/models/checkpoint.py). The walk-forward study's
own conclusion -- no configuration here has a demonstrated market-beating edge -- applies
to this run exactly as it does to every other one.

What this genuinely adds: the technical model, sentiment agent, regime agent (with the
optional India VIX / Nifty-momentum macro features), debate layer, and the cost-aware
risk manager all run against real, current data end-to-end, producing the same
structured, auditable rationale the backtest produces -- proof the system runs as a
live pipeline, not only as a historical replay.

Usage:
  .venv/bin/python scripts/train_production_model.py   # once, or whenever retraining
  .venv/bin/python scripts/run_live_signal.py --backend local
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.orchestrator import Orchestrator, OrchestratorConfig
from nse_agents.agents.regime import RegimeAgent
from nse_agents.agents.risk import RiskLimits, RiskManager
from nse_agents.agents.sentiment import SentimentAgent
from nse_agents.agents.technical import TechnicalAgent
from nse_agents.agents.trader import Trader, TraderConfig
from nse_agents.config import RESULTS
from nse_agents.live.engine import rebalance
from nse_agents.live.mock_broker import MockBroker
from nse_agents.live.snapshot import build_live_snapshot
from nse_agents.live.state_store import DEFAULT_DB_PATH, PaperTradingStore

OUT = RESULTS / "paper_trading"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(RESULTS / "production" / "technical_model.pt"))
    ap.add_argument("--backend", default="local", choices=["local", "anthropic", "openai", "echo", "none"])
    ap.add_argument("--debate-mode", default="disagreement",
                     choices=["always", "disagreement", "never"])
    ap.add_argument("--disagreement-metric", default="conviction", choices=["raw", "zscore", "conviction"])
    ap.add_argument("--cost-aware", action="store_true", default=True)
    ap.add_argument("--no-cost-aware", dest="cost_aware", action="store_false")
    ap.add_argument("--persist", action="store_true",
                     help="Apply today's target weights to the persistent paper-trading "
                          "account (via MockBroker) and update its state. Without this "
                          "flag, the run only prints and writes the signal JSON -- "
                          "nothing about the account changes.")
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--initial-capital", type=float, default=1_000_000.0)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    from nse_agents.llm.factory import build_backend

    backend = None if args.backend == "none" else build_backend(args.backend, cache=True)

    print("building today's snapshot (prices, features, headlines)...", flush=True)
    snapshot = build_live_snapshot(Path(args.model), backend=backend)
    print(f"as of {snapshot.as_of.date()}  "
          f"(model trained through {snapshot.checkpoint_meta['trained_through']}, "
          f"architecture {snapshot.checkpoint_meta['architecture']})", flush=True)

    technical = TechnicalAgent(snapshot.technical_oos, architecture=snapshot.checkpoint_meta["architecture"])
    regime = RegimeAgent(snapshot.regime_table, use_macro=True)
    agents = [technical, regime]
    if len(snapshot.sentiment_daily):
        agents.append(SentimentAgent(snapshot.sentiment_daily))
        print(f"sentiment: {len(snapshot.sentiment_daily)} (symbol,day) views from today's headlines", flush=True)
    else:
        print("sentiment: no backend / no headlines -- technical + regime only", flush=True)

    risk = RiskManager(RiskLimits(cost_aware=args.cost_aware))
    trader = Trader(TraderConfig(use_debate=args.debate_mode != "never"), risk)
    orchestrator = Orchestrator(
        agents=agents, trader=trader, backend=backend,
        config=OrchestratorConfig(
            debate_mode=args.debate_mode, disagreement_metric=args.disagreement_metric,
            verbose=False,
        ),
    )

    pairs = snapshot.technical_oos[["date", "symbol"]].copy()
    pairs["fwd_ret"] = float("nan")  # unknown -- today's outcome hasn't happened yet
    frame, decisions = orchestrator.run(pairs)

    report = []
    for decision in sorted(decisions, key=lambda d: -d.score):
        report.append({
            "symbol": decision.symbol, "action": decision.action,
            "score": round(decision.score, 4), "size": round(decision.size, 4),
            "rationale": decision.rationale,
        })

    out_path = OUT / f"signal_{snapshot.as_of.date()}.json"
    out_path.write_text(json.dumps({
        "as_of": str(snapshot.as_of.date()),
        "disclaimer": "Demonstration only. No configuration in this study has a "
                       "demonstrated market-beating edge -- see README.md. This is not "
                       "a trading recommendation.",
        "model": snapshot.checkpoint_meta,
        "decisions": report,
    }, indent=2))

    print(f"\n{'symbol':12s} {'action':6s} {'score':>7s} {'size':>7s}")
    print("-" * 40)
    for row in report:
        print(f"{row['symbol']:12s} {row['action']:6s} {row['score']:+7.3f} {row['size']:7.3f}")
    print(f"\nfull rationale written to {out_path}")

    if args.persist:
        # Target weights straight from the trader's own sizing -- a FLAT/HOLD
        # decision already carries size=0.0, so it needs no special-casing here:
        # rebalance() treats an absent or zero weight identically.
        target_weights = {row["symbol"]: row["size"] for row in report}
        store = PaperTradingStore(Path(args.db), initial_capital=args.initial_capital)
        broker = MockBroker()
        result = rebalance(store, broker, target_weights, snapshot.as_of)

        print(f"\n--- paper account updated ({len(result.fills)} fill(s)) ---")
        for fill in result.fills:
            print(f"  {fill.symbol:12s} {fill.side:4s} {fill.quantity:8.2f} @ "
                  f"Rs {fill.price:8.2f}  (cost Rs {fill.cost:.2f})")
        print(f"  total equity: Rs {result.total_equity:,.2f}")
        print(f"  state: {args.db}")
        print(f"  see current status with: python -m nse_agents.cli paper-status --db {args.db}")

    print("\nDEMONSTRATION ONLY -- see README.md: no configuration in this study has a "
          "demonstrated market-beating edge net of real costs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
