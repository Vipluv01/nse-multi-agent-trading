"""Institutional-style strategy factsheet for the flagship multi-agent system
(``Full+Debate`` -- the one configuration that runs every specialist plus the
debate pass; see results/agents/summary.csv for the full ablation).

**This factsheet does not launder the study's own headline finding.** Full+Debate
does not beat Buy&Hold net of costs (see README.md), and every number here is
computed from the same cached decisions/prices every other report in this study
reads -- nothing here is a new backtest or a differently-selected window. The
factsheet format (key metrics, monthly heatmap, benchmark comparison table, LLM
usage stats) is institutional convention; the content is this project's real,
already-reported result, reformatted for that convention rather than reshaped to
flatter it.

Usage:  .venv/bin/python scripts/generate_factsheet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.backtest.baselines import build_baseline_signals
from nse_agents.backtest.benchmarks import align_benchmark_to_dates, build_extended_benchmarks
from nse_agents.backtest.engine import backtest_signals, run_backtest
from nse_agents.backtest.metrics import (
    Performance,
    beta,
    compute_performance,
    information_ratio,
)
from nse_agents.config import RESULTS, SETTINGS

OUT_MD = RESULTS / "FACTSHEET.md"
HEATMAP_PATH = RESULTS / "figures" / "monthly_return_heatmap.png"
STRATEGY = "Full+Debate"
pd.set_option("display.width", 220)


def load_headline_result():
    path = RESULTS / "agents" / f"decisions_{STRATEGY}.csv"
    frame = pd.read_csv(path, parse_dates=["date"])
    return run_backtest(frame, STRATEGY)


def key_metrics_table(result, nifty_returns: np.ndarray) -> pd.DataFrame:
    perf: Performance = result.performance
    b = beta(result.returns, nifty_returns)
    ir = information_ratio(result.returns, nifty_returns)
    rows = [
        ("CAGR", f"{perf.cagr:+.2%}"),
        ("Sharpe (net, excess)", f"{perf.sharpe:+.3f}"),
        ("Sortino", f"{perf.sortino:+.3f}"),
        ("Calmar", f"{perf.calmar:+.3f}"),
        ("Information Ratio (vs Nifty 50)", f"{ir:+.3f}" if np.isfinite(ir) else "n/a"),
        ("Beta (vs Nifty 50)", f"{b:.3f}" if np.isfinite(b) else "n/a"),
        ("Max Drawdown", f"{perf.max_drawdown:.2%}"),
        ("Annualised Volatility", f"{perf.ann_vol:.2%}"),
        ("Hit Rate", f"{perf.hit_rate:.1%}"),
        ("Trading Days", f"{perf.n_days:,}"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def monthly_return_matrix(daily: pd.DataFrame) -> pd.DataFrame:
    """Year x month matrix of compounded monthly returns, 2018-present."""
    frame = daily.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.loc[frame["date"].dt.year >= 2018]
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    monthly = frame.groupby(["year", "month"])["net_return"].apply(
        lambda r: float(np.prod(1.0 + r) - 1.0)
    )
    return monthly.unstack("month")


def benchmark_comparison_table(result) -> pd.DataFrame:
    """Full+Debate, Buy&Hold, and the three raw benchmark indices, each
    reported on its own (possibly shorter, for Nifty Next 50) real window."""
    rows = []
    perf = result.performance
    rows.append({
        "Series": STRATEGY, "CAGR": perf.cagr, "Sharpe": perf.sharpe,
        "MaxDD": perf.max_drawdown, "Days": perf.n_days,
    })

    start = result.dates.min()
    bh = build_baseline_signals()["Buy&Hold"]
    bh_window = bh.loc[bh["date"] >= start]
    bh_result = backtest_signals(bh_window, "Buy&Hold", threshold=0.5)
    bh_perf = bh_result.performance
    rows.append({
        "Series": "Buy&Hold", "CAGR": bh_perf.cagr, "Sharpe": bh_perf.sharpe,
        "MaxDD": bh_perf.max_drawdown, "Days": bh_perf.n_days,
    })

    for name, bench in build_extended_benchmarks(start=str(start.date()), end=SETTINGS.end).items():
        bperf = compute_performance(bench["fwd_ret"].to_numpy())
        rows.append({
            "Series": name, "CAGR": bperf.cagr, "Sharpe": bperf.sharpe,
            "MaxDD": bperf.max_drawdown, "Days": bperf.n_days,
        })
    return pd.DataFrame(rows)


def llm_stats() -> dict:
    decisions_path = RESULTS / "agents" / f"decisions_{STRATEGY}.csv"
    decisions = pd.read_csv(decisions_path)
    total_decisions = len(decisions)
    escalation_rate = float(decisions["debated"].mean()) if "debated" in decisions else float("nan")

    scored_path = RESULTS / "sentiment" / "scored_local.csv"
    label_dist = None
    n_headlines = 0
    if scored_path.exists():
        scored = pd.read_csv(scored_path)
        n_headlines = len(scored)
        probs = scored[["p_good", "p_bad", "p_unknown"]].to_numpy()
        labels = np.array(["Good", "Bad", "Unknown"])[probs.argmax(axis=1)]
        label_dist = pd.Series(labels).value_counts(normalize=True)

    return {
        "total_decisions": total_decisions,
        "escalation_rate": escalation_rate,
        "n_headlines": n_headlines,
        "label_dist": label_dist,
    }


def _to_markdown_table(frame: pd.DataFrame) -> str:
    """A minimal Markdown table renderer, in place of ``DataFrame.to_markdown``
    (which pulls in the ``tabulate`` package, not otherwise a dependency of
    this project) -- every value here is already pre-formatted as a display
    string by the caller, so no alignment or numeric formatting logic is
    needed here."""
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep] + body)


def render_markdown(
    metrics: pd.DataFrame, benchmarks: pd.DataFrame, llm: dict, heatmap_rel_path: str,
) -> str:
    lines = []
    lines.append("# Strategy Factsheet -- Full+Debate\n")
    lines.append(
        "**This is not a positive track record.** Full+Debate does not beat "
        "Buy&Hold net of real Indian transaction costs over this study's "
        "7.6-year out-of-sample window (95% CI on the Sharpe gap includes "
        "zero -- see README.md). This factsheet reports the same result in "
        "institutional format, not a different or more favourable one.\n"
    )
    lines.append("## Key Metrics\n")
    lines.append(_to_markdown_table(metrics))
    lines.append("")

    lines.append("## Monthly Return Heatmap (2018-present)\n")
    lines.append(f"![Monthly returns]({heatmap_rel_path})\n")

    lines.append("## Benchmark Comparison\n")
    display = benchmarks.copy()
    display["CAGR"] = display["CAGR"].map(lambda v: f"{v:+.2%}")
    display["Sharpe"] = display["Sharpe"].map(lambda v: f"{v:+.3f}")
    display["MaxDD"] = display["MaxDD"].map(lambda v: f"{v:.2%}")
    lines.append(_to_markdown_table(display))
    lines.append(
        "\n*Nifty Next 50 starts 2020-01-01 (Yahoo's data floor for that "
        "ticker) -- its row covers a shorter window than the others and is "
        "not directly comparable in absolute terms.*\n"
    )

    lines.append("## LLM Debate & Sentiment Stats\n")
    lines.append(f"- **Total decisions**: {llm['total_decisions']:,}")
    lines.append(f"- **Debate escalation rate**: {llm['escalation_rate']:.1%} of decisions "
                  f"went to the bull/bear debate pass")
    if llm["label_dist"] is not None:
        lines.append(f"- **Sentiment label distribution** ({llm['n_headlines']:,} scored headlines):")
        for label, share in llm["label_dist"].items():
            lines.append(f"  - {label}: {share:.1%}")
    else:
        lines.append("- Sentiment label distribution: no scored-headline cache found")
    lines.append("")

    lines.append(
        "---\n*Generated by `scripts/generate_factsheet.py`. See README.md for the full "
        "study, KNOWN_ISSUES.md for what remains unverified, and PREREGISTRATION.md for "
        "the pre-registered attempts this result was checked against.*"
    )
    return "\n".join(lines)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    result = load_headline_result()
    print(f"loaded {STRATEGY}: {result.performance.n_days} days, "
          f"Sharpe {result.performance.sharpe:+.3f}", flush=True)

    benchmarks = build_extended_benchmarks()
    nifty_returns = align_benchmark_to_dates(benchmarks["Nifty50"], result.dates)
    metrics = key_metrics_table(result, nifty_returns)
    print("\n=== Key Metrics ===")
    print(metrics.to_string(index=False), flush=True)

    monthly = monthly_return_matrix(result.daily)
    from nse_agents.report import figures

    figures.monthly_return_heatmap(
        monthly, HEATMAP_PATH,
        f"{STRATEGY}: monthly returns, 2018-present",
        "Net of Indian transaction costs -- see README.md for the full-period result",
    )
    print(f"\nwrote {HEATMAP_PATH}", flush=True)

    bench_table = benchmark_comparison_table(result)
    print("\n=== Benchmark Comparison ===")
    print(bench_table.round(4).to_string(index=False), flush=True)

    llm = llm_stats()
    print("\n=== LLM Debate & Sentiment Stats ===")
    print(f"total decisions: {llm['total_decisions']:,}", flush=True)
    print(f"debate escalation rate: {llm['escalation_rate']:.1%}", flush=True)
    if llm["label_dist"] is not None:
        print(llm["label_dist"].to_string(), flush=True)

    markdown = render_markdown(metrics, bench_table, llm, "figures/monthly_return_heatmap.png")
    OUT_MD.write_text(markdown)
    print(f"\nwrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
