"""Render the paper-trading state into a readable Markdown report.

Reads only the persisted state (state.sqlite) -- it never recomputes or re-simulates
anything, so the report cannot disagree with the account it describes. An equity-curve
chart is written alongside it and embedded.

Usage:
  .venv/bin/python scripts/generate_paper_report.py
  .venv/bin/python scripts/generate_paper_report.py --db path/to/state.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS
from nse_agents.live.state_store import DEFAULT_DB_PATH, PaperTradingStore
from nse_agents.report import figures

OUT_DIR = RESULTS / "paper_trading"
MIN_DAYS_FOR_SHARPE = 60


def _latest_price(symbol: str) -> float:
    from nse_agents.config import SETTINGS
    from nse_agents.data.prices import load_prices

    return float(load_prices(symbol, SETTINGS.start, "2100-01-01")["close"].iloc[-1])


def render_equity_chart(history: pd.DataFrame, path: Path) -> Path | None:
    if len(history) < 2:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    dates = pd.to_datetime(history["date"])
    # Markers, because this is a handful of discrete daily observations, not
    # a continuous series -- a bare line implies a resolution the data doesn't
    # have. Day-level ticks for the same reason: the default locator/formatter
    # picks sub-day ticks (e.g. "08-28 12:00") for a date range this short,
    # which implies intraday granularity that was never recorded.
    ax.plot(dates, history["total_equity"], color=figures.SERIES[0], lw=2,
            marker="o", ms=5, mec=figures.SURFACE, mew=1.2)
    ax.axhline(history["total_equity"].iloc[0], color=figures.GRID, lw=1, ls=(0, (4, 3)))
    ax.set_ylabel("Account equity (Rs)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    figures._finish(ax, "Paper-trading account equity",
                    "Simulated fills via MockBroker -- demonstration only, not a live track record")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--out", default=str(OUT_DIR / "DAILY_REPORT.md"))
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no paper-trading state at {db_path}; nothing to report.")
        return 1

    store = PaperTradingStore(db_path)
    snap = store.snapshot()
    history = store.equity_history()
    trades = store.trade_log()

    prices = {sym: _latest_price(sym) for sym in snap.positions}
    positions_value = sum(pos.market_value(prices[sym]) for sym, pos in snap.positions.items())
    total_equity = snap.cash + positions_value
    total_return = total_equity / snap.initial_capital - 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path = render_equity_chart(history, out_path.parent / "equity_curve.png")

    lines: list[str] = []
    lines.append("# Paper Trading Report")
    lines.append("")
    lines.append(f"_Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} "
                 f"from `{db_path.name}`._")
    lines.append("")
    lines.append("> **Demonstration only.** Fills are simulated through `MockBroker`, not "
                 "executed with a real broker, and no configuration in this study has a "
                 "demonstrated market-beating edge net of real costs — see "
                 "[README.md](../../README.md). This report describes a simulated account, "
                 "not a track record.")
    lines.append("")

    lines.append("## Account")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Initial capital | Rs {snap.initial_capital:,.2f} |")
    lines.append(f"| Cash | Rs {snap.cash:,.2f} |")
    lines.append(f"| Positions value | Rs {positions_value:,.2f} |")
    lines.append(f"| **Total equity** | **Rs {total_equity:,.2f}** |")
    lines.append(f"| Total return | {total_return:+.2%} |")
    lines.append(f"| Realised P&L | Rs {snap.realized_pnl:,.2f} |")
    lines.append(f"| Transaction costs paid | Rs {snap.total_costs:,.2f} |")
    lines.append(f"| Trading days tracked | {len(history)} |")
    lines.append("")

    if chart_path is not None:
        lines.append(f"![Equity curve]({chart_path.name})")
        lines.append("")

    lines.append("## Open positions")
    lines.append("")
    if snap.positions:
        lines.append("| Symbol | Quantity | Avg cost | Market value | Return |")
        lines.append("|---|---:|---:|---:|---:|")
        for sym, pos in sorted(snap.positions.items()):
            price = prices[sym]
            lines.append(
                f"| {sym} | {pos.quantity:,.2f} | Rs {pos.avg_price:,.2f} | "
                f"Rs {pos.market_value(price):,.2f} | {price / pos.avg_price - 1.0:+.2%} |"
            )
    else:
        lines.append("_No open positions._")
    lines.append("")

    lines.append("## Performance")
    lines.append("")
    returns = history["daily_return"].dropna().to_numpy() if len(history) else np.array([])
    if len(history) >= 1:
        # Drawdown must run over the full equity path *including* the starting
        # capital and today's live mark-to-market -- computing it from just the
        # daily_return series drops day one (no prior day to difference against),
        # and computing it from only the *recorded* rebalance-day snapshots drops
        # today if positions haven't been rebalanced yet: "Total return" above
        # already uses today's live prices, so a drawdown that ignores them could
        # under-report the true worst point while the headline return shows it.
        equity_path = np.concatenate(
            [[snap.initial_capital], history["total_equity"].to_numpy(), [total_equity]]
        )
        peak = np.maximum.accumulate(equity_path)
        max_dd = float((equity_path / peak - 1.0).min())
        lines.append(f"- Cumulative return: **{total_return:+.2%}**")
        lines.append(f"- Maximum drawdown: **{max_dd:.2%}**")
    if len(returns) >= MIN_DAYS_FOR_SHARPE:
        from nse_agents.backtest.metrics import compute_performance

        perf = compute_performance(returns)
        lines.append(f"- Sharpe (net, excess): **{perf.sharpe:+.3f}**")
    else:
        lines.append(
            f"- Sharpe: _withheld_ — an annualised Sharpe from {len(returns)} day(s) is "
            f"not a small estimate but a meaningless one. Reported once "
            f"{MIN_DAYS_FOR_SHARPE}+ days are tracked."
        )
    lines.append("")

    lines.append("## Trade log")
    lines.append("")
    if len(trades):
        lines.append("| Date | Symbol | Side | Quantity | Price | Cost |")
        lines.append("|---|---|---|---:|---:|---:|")
        for row in trades.itertuples(index=False):
            lines.append(
                f"| {row.date} | {row.symbol} | {row.side} | {row.quantity:,.2f} | "
                f"Rs {row.price:,.2f} | Rs {row.cost:,.2f} |"
            )
    else:
        lines.append("_No trades recorded._")
    lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")
    if chart_path:
        print(f"wrote {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
