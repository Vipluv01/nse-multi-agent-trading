"""Factor regression report: for every strategy in the main ablation plus the
classical baselines, decompose net returns into a market-excess-return
exposure, a momentum (WML) exposure, and a residual alpha -- answering
"is this strategy's Sharpe explained by beta and momentum tilt, or is there a
residual the two-factor model doesn't explain" for every configuration this
study reports, not just the flagship.

Two factors, not four -- see nse_agents/backtest/factor_model.py's module
docstring for why SMB and HML are not implemented (they need point-in-time
fundamentals this project does not have and will not fake).

This is a reporting pass over already-computed results -- it trains nothing
and changes no number reported anywhere else in the study.

Usage:  .venv/bin/python scripts/factor_regression_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.engine import backtest_signals, run_backtest
from nse_agents.backtest.factor_model import UNAVAILABLE_FACTORS, build_factors, regress_factors
from nse_agents.config import RESULTS

OUT = RESULTS / "improvements"
pd.set_option("display.width", 220)


def load_strategy_returns() -> dict:
    out = {}
    agents_dir = RESULTS / "agents"
    for name in ("Tech-only", "Tech+Regime", "Tech+Sentiment", "Tech+Sent+Regime", "Full+Debate"):
        path = agents_dir / f"decisions_{name}.csv"
        if path.exists():
            frame = pd.read_csv(path, parse_dates=["date"])
            out[name] = run_backtest(frame, name)

    start = min(r.dates.min() for r in out.values()) if out else "2018-12-07"
    for name, frame in build_baseline_signals().items():
        window = frame.loc[frame["date"] >= start]
        out[name] = backtest_signals(window, name, threshold=0.5)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    strategies = load_strategy_returns()
    print(f"loaded {len(strategies)} strategies: {list(strategies)}", flush=True)

    factors = build_factors()
    print(f"factor window: {factors['date'].min().date()} -> {factors['date'].max().date()} "
          f"({len(factors):,} days)", flush=True)
    for name, reason in UNAVAILABLE_FACTORS.items():
        print(f"  NOTE: {name.upper()} not implemented -- {reason}", flush=True)

    rows = []
    for name, result in strategies.items():
        try:
            reg = regress_factors(result.returns, result.dates, factors)
        except ValueError as exc:
            print(f"  {name}: skipped -- {exc}", flush=True)
            continue
        rows.append({
            "strategy": name, "n_days": reg.n_days,
            "alpha_annualized": reg.alpha_annualized, "alpha_tstat": reg.alpha_tstat,
            "alpha_pvalue": reg.alpha_pvalue,
            "beta_mkt": reg.betas["mkt_excess"], "beta_mkt_tstat": reg.beta_tstats["mkt_excess"],
            "beta_wml": reg.betas["wml"], "beta_wml_tstat": reg.beta_tstats["wml"],
            "r_squared": reg.r_squared,
        })

    table = pd.DataFrame(rows).sort_values("alpha_annualized", ascending=False).reset_index(drop=True)
    table.to_csv(OUT / "factor_regression.csv", index=False)
    print("\n" + table.round(4).to_string(index=False), flush=True)

    significant = table[table["alpha_pvalue"] < 0.05]
    print(f"\n{len(significant)}/{len(table)} strategies show a statistically significant "
          f"(p<0.05) two-factor alpha.", flush=True)
    print("Interpretation guide: a significant positive alpha here means the strategy's return "
          "is not explained by market beta and this universe's own momentum tilt alone -- it does "
          "NOT by itself mean the alpha is large enough to clear transaction costs (see "
          "results/agents/summary.csv's CostDrag/yr) or survive the study's own Holm-corrected "
          "significance bar (results/agents/sharpe_intervals.csv) applied to the raw Sharpe ratio.",
          flush=True)
    print(f"\nwrote {OUT / 'factor_regression.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
