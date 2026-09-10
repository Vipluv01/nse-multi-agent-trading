"""Command-line entry point: ``python -m nse_agents.cli <subcommand>``.

A subcommand dispatcher, not a growing pile of scripts -- so a future command shares
this file's argument-parsing shell rather than each script reinventing one. Only
``paper-status`` exists today; add a new subcommand by registering a parser and a
handler in ``_SUBCOMMANDS`` below.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.baselines import build_baseline_signals
from .backtest.engine import backtest_signals
from .backtest.metrics import compute_performance
from .live.state_store import DEFAULT_DB_PATH, PaperTradingStore


def _latest_price(symbol: str, as_of: pd.Timestamp | None = None) -> float:
    from .data.prices import load_prices
    from .config import SETTINGS

    frame = load_prices(symbol, SETTINGS.start, "2100-01-01")
    if as_of is not None:
        frame = frame.loc[frame["date"] <= as_of]
    return float(frame["close"].iloc[-1])


def cmd_paper_status(args: argparse.Namespace) -> int:
    path = Path(args.db)
    if not path.exists():
        print(f"no paper-trading state at {path} -- run scripts/run_live_signal.py "
              f"(or the paper-trading engine) at least once first.")
        return 1

    store = PaperTradingStore(path)
    snap = store.snapshot()
    history = store.equity_history()

    prices = {sym: _latest_price(sym) for sym in snap.positions}
    positions_value = sum(pos.market_value(prices[sym]) for sym, pos in snap.positions.items())
    total_equity = snap.cash + positions_value
    total_return = total_equity / snap.initial_capital - 1.0

    print(f"{'PAPER TRADING STATUS':^60}")
    print("=" * 60)
    print(f"  Initial capital  : Rs {snap.initial_capital:>14,.2f}")
    print(f"  Cash             : Rs {snap.cash:>14,.2f}")
    print(f"  Positions value  : Rs {positions_value:>14,.2f}")
    print(f"  Total equity     : Rs {total_equity:>14,.2f}")
    print(f"  Total return     :    {total_return:>+14.2%}")
    print(f"  Realized P&L     : Rs {snap.realized_pnl:>14,.2f}")
    print(f"  Total costs paid : Rs {snap.total_costs:>14,.2f}")
    print(f"  Last updated     :    {snap.last_updated}")

    if snap.positions:
        print(f"\n  {'OPEN POSITIONS':^56}")
        print(f"  {'-'*56}")
        print(f"  {'symbol':<12}{'qty':>10}{'avg cost':>12}{'mkt value':>14}{'return':>10}")
        for sym, pos in sorted(snap.positions.items()):
            price = prices[sym]
            ret = price / pos.avg_price - 1.0
            print(f"  {sym:<12}{pos.quantity:>10.2f}{pos.avg_price:>12.2f}"
                  f"{pos.market_value(price):>14,.2f}{ret:>+10.2%}")
    else:
        print("\n  No open positions.")

    # An annualised Sharpe from a handful of days is not a small estimate, it is
    # a meaningless one -- annualising a 2-day sample produced "+62.33" in
    # testing. This project reports drawdown and cumulative return on short
    # samples and withholds the ratio until there is enough data for it to
    # mean anything, exactly as the crash-regime analysis does (see
    # scripts/regime_analysis.py).
    MIN_DAYS_FOR_SHARPE = 60
    returns = history["daily_return"].dropna().to_numpy() if len(history) else np.array([])

    print(f"\n  {'PERFORMANCE (paper account)':^56}")
    print(f"  {'-'*56}")
    print(f"  Days tracked          : {len(history)}")
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
        print(f"  Cumulative return     : {total_return:+.2%}")
        print(f"  Max drawdown          : {max_dd:.2%}")

    if len(returns) >= MIN_DAYS_FOR_SHARPE:
        perf = compute_performance(returns)
        print(f"  Sharpe (net, excess)  : {perf.sharpe:+.3f}")
        bh = build_baseline_signals()["Buy&Hold"]
        start = pd.Timestamp(history["date"].iloc[0])
        bh_window = bh.loc[bh["date"] >= start]
        if len(bh_window):
            bh_result = backtest_signals(bh_window, "Buy&Hold", threshold=0.5)
            print(f"  Buy&Hold Sharpe (same) : {bh_result.performance.sharpe:+.3f}")
    else:
        print(f"  Sharpe                : withheld -- needs {MIN_DAYS_FOR_SHARPE}+ days "
              f"({len(returns)} so far)")
        print(f"  Nifty comparison      : withheld until then, for the same reason")

    print("\n  DEMONSTRATION ONLY -- see README.md: no configuration in this study")
    print("  has a demonstrated market-beating edge net of real costs.")
    print("=" * 60)
    return 0


def cmd_healthcheck(args: argparse.Namespace) -> int:
    from .live.healthcheck import run_all_checks

    results = run_all_checks(db_path=args.db, ping_llm=args.ping_llm)

    print(f"{'PRE-MARKET HEALTH CHECK':^60}")
    print("=" * 60)
    all_ok = True
    for result in results:
        status = "OK  " if result.ok else "FAIL"
        print(f"  [{status}] {result.name}")
        print(f"          {result.detail}")
        all_ok = all_ok and result.ok
    print("=" * 60)

    if all_ok:
        print("  All checks passed.")
        return 0
    failed = [r.name for r in results if not r.ok]
    print(f"  {len(failed)} check(s) failed: {', '.join(failed)}")
    return 1


_SUBCOMMANDS = {
    "paper-status": (
        "Show current paper-trading account state, positions and performance.",
        cmd_paper_status,
        lambda p: p.add_argument("--db", default=str(DEFAULT_DB_PATH)),
    ),
    "healthcheck": (
        "Validate price feed freshness, LLM key config, RSS availability, and "
        "paper-trading DB integrity before a market-open run. Exit code 0 iff all pass.",
        cmd_healthcheck,
        lambda p: (
            p.add_argument("--db", default=str(DEFAULT_DB_PATH)),
            p.add_argument("--ping-llm", action="store_true",
                           help="Also make a real, minimal call to any configured "
                                "LLM provider to confirm the key actually works "
                                "(off by default -- costs a token, not just a check)."),
        ),
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m nse_agents.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _handler, add_args) in _SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        add_args(sub)

    args = parser.parse_args(argv)
    _, handler, _ = _SUBCOMMANDS[args.command]
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
