"""Command-line entry point: ``python -m nse_agents.cli <subcommand>``.

A subcommand dispatcher, not a growing pile of scripts -- so a future command shares
this file's argument-parsing shell rather than each script reinventing one. Only
``paper-status`` exists today; add a new subcommand by registering a parser and a
handler in ``_SUBCOMMANDS`` below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.baselines import build_baseline_signals
from .backtest.engine import backtest_signals
from .backtest.metrics import compute_performance
from .config import RESULTS
from .live.state_store import DEFAULT_DB_PATH, PaperTradingStore

DEFAULT_EXPORT_PATH = RESULTS / "export.json"


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


def cmd_db_backup(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no paper-trading state at {db_path} -- nothing to back up.")
        return 1
    store = PaperTradingStore(db_path)
    backup_path = store.backup(args.backup_dir)
    size_kb = backup_path.stat().st_size / 1024
    print(f"backed up {db_path} -> {backup_path} ({size_kb:.1f} KB, gzip-compressed)")
    return 0


def cmd_db_vacuum(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no paper-trading state at {db_path} -- nothing to vacuum.")
        return 1
    store = PaperTradingStore(db_path)
    result = store.vacuum()
    print(f"{'DB VACUUM':^50}")
    print("=" * 50)
    print(f"  size before checkpoint : {result['size_before_checkpoint_bytes']:,} bytes")
    print(f"  size after checkpoint  : {result['size_after_checkpoint_bytes']:,} bytes "
          f"({result['checkpoint_grew_file_by_bytes']:+,} bytes -- WAL content absorbed)")
    print(f"  size after VACUUM      : {result['size_after_vacuum_bytes']:,} bytes "
          f"(-{result['vacuum_reclaimed_bytes']:,} bytes reclaimed)")
    print(f"  WAL frames checkpointed: {result['wal_checkpointed_frames']}")
    print("=" * 50)
    return 0


def cmd_audit_data(args: argparse.Namespace) -> int:
    from .data.audit import LOG_PATH, audit_universe, findings_to_frame, write_report

    print("running price-data audit...", flush=True)
    findings = audit_universe()
    table = findings_to_frame(findings)
    path = write_report(findings, Path(args.log) if args.log else LOG_PATH)

    print(f"{'DATA AUDIT':^60}")
    print("=" * 60)
    if table.empty:
        print("  no anomalies found")
    else:
        for kind, group in table.groupby("kind"):
            print(f"  {kind:<36}{len(group):>6} finding(s)")
    print("=" * 60)
    print(f"  full report: {path}")
    return 0


def cmd_db_migrate(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    existed_before = db_path.exists()
    # Connecting already runs every pending migration (see state_store.py) --
    # this command exists so an operator can apply and see them explicitly,
    # on their own schedule, rather than have migration happen silently as a
    # side effect of the next incidental paper-status/run_live_signal call.
    store = PaperTradingStore(db_path)
    version = store.schema_version()
    remaining = store.pending_migrations()

    print(f"{'DB MIGRATE':^60}")
    print("=" * 60)
    if not existed_before:
        print(f"  initialized fresh database at {db_path}")
    print(f"  schema version: {version}")
    if remaining:
        print(f"  {len(remaining)} migration(s) still pending (unexpected -- report this):")
        for m in remaining:
            print(f"    v{m.version}: {m.description}")
    else:
        print("  up to date -- no pending migrations")
    print("=" * 60)
    return 0


def _json_default(obj):
    """Handles the two object types that leak out of pandas/numpy round-trips
    and are not natively JSON-serialisable: numpy scalar types (cast to the
    equivalent Python type, so a number stays a JSON number, not a string) and
    pandas Timestamps (ISO-formatted). Anything else falls back to ``str()``
    rather than raising, so an export never crashes on one unexpected field --
    it degrades to a readable string instead.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return str(obj)


def _load_strategy_for_export(name: str):
    """Every strategy this study reports carries either a cached per-decision
    log (the multi-agent ablation) or a cached signal frame (the classical
    baselines) -- this mirrors the identical loading pattern already used by
    scripts/risk_attribution_report.py and scripts/factor_regression_report.py,
    kept local rather than imported from scripts/ since nse_agents/ is the
    package those scripts themselves depend on, not the reverse."""
    from .backtest.engine import backtest_signals, run_backtest

    decisions_path = RESULTS / "agents" / f"decisions_{name}.csv"
    if decisions_path.exists():
        decisions = pd.read_csv(decisions_path, parse_dates=["date"])
        result = run_backtest(decisions, name)
        log_columns = [c for c in ("date", "symbol", "action", "score", "size", "debated") if c in decisions]
        per_symbol_log = decisions[log_columns]
        log_kind = "per_symbol_decision_log"
        return result, per_symbol_log, log_kind

    baselines = build_baseline_signals()
    if name in baselines:
        signals = baselines[name]
        result = backtest_signals(signals, name, threshold=0.5)
        log_columns = [c for c in ("date", "symbol", "prob_up", "fwd_ret") if c in signals]
        per_symbol_log = signals[log_columns]
        log_kind = "per_symbol_signal_log"  # baselines have no agent-style Decision
        return result, per_symbol_log, log_kind

    available = sorted({p.stem.replace("decisions_", "") for p in (RESULTS / "agents").glob("decisions_*.csv")}
                        | set(baselines))
    raise KeyError(f"unknown strategy {name!r}; available: {available}")


def cmd_export_metrics(args: argparse.Namespace) -> int:
    from .backtest.factor_model import build_factors, regress_factors
    from .backtest.metrics import drawdown_series

    try:
        result, per_symbol_log, log_kind = _load_strategy_for_export(args.strategy)
    except KeyError as exc:
        print(exc)
        return 1

    equity = np.cumprod(1.0 + result.returns)
    drawdown = drawdown_series(result.returns)
    equity_curve = pd.DataFrame({"date": result.dates, "equity": equity, "drawdown": drawdown})

    try:
        factors = build_factors()
        factor_result = regress_factors(result.returns, result.dates, factors).to_dict()
    except ValueError as exc:
        # Too few overlapping days against the factor window -- a real,
        # reportable outcome for a short-lived strategy, not a crash.
        factor_result = {"error": str(exc)}

    performance = {
        "gross": result.performance_gross.to_dict(),
        "net": result.performance.to_dict(),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        payload = {
            "strategy": args.strategy,
            "generated_at": pd.Timestamp.now().isoformat(),
            "performance": performance,
            "factor_regression": factor_result,
            "equity_curve": equity_curve.to_dict(orient="records"),
            log_kind: per_symbol_log.to_dict(orient="records"),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    else:  # csv -- one table per concern, since nested factor betas and a
           # per-symbol log do not share row shape and forcing them into one
           # table would mean fabricating a join key none of them actually share.
        stem = out_path.with_suffix("")
        perf_row = {"strategy": args.strategy}
        for scope, d in performance.items():
            perf_row.update({f"{scope}_{k}": v for k, v in d.items()})
        pd.DataFrame([perf_row]).to_csv(f"{stem}_performance.csv", index=False)
        equity_curve.to_csv(f"{stem}_equity_curve.csv", index=False)
        per_symbol_log.to_csv(f"{stem}_{log_kind}.csv", index=False)
        factor_row = {"strategy": args.strategy}
        for k, v in factor_result.items():
            if isinstance(v, dict):
                factor_row.update({f"{k}_{kk}": vv for kk, vv in v.items()})
            elif not isinstance(v, (list, tuple)):
                factor_row[k] = v
        pd.DataFrame([factor_row]).to_csv(f"{stem}_factor_regression.csv", index=False)
        print(f"wrote {stem}_performance.csv, {stem}_equity_curve.csv, "
              f"{stem}_{log_kind}.csv, {stem}_factor_regression.csv")
    return 0


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
    "db-backup": (
        "Create a timestamped, gzip-compressed backup of the paper-trading database "
        "in results/paper_trading/backups/ (via SQLite's online backup API, not a "
        "plain file copy -- safe under WAL mode's split main-file/-wal-file storage).",
        cmd_db_backup,
        lambda p: (
            p.add_argument("--db", default=str(DEFAULT_DB_PATH)),
            p.add_argument("--backup-dir", default=None,
                           help="Defaults to <db's parent dir>/backups."),
        ),
    ),
    "db-vacuum": (
        "WAL-checkpoint and VACUUM the paper-trading database to reclaim space "
        "and keep query performance from degrading as the trade log grows.",
        cmd_db_vacuum,
        lambda p: p.add_argument("--db", default=str(DEFAULT_DB_PATH)),
    ),
    "audit-data": (
        "Scan cached price series for possible unadjusted corporate actions, "
        "zero-volume/missing-trading-day gaps, stale repeated prices, and "
        "outlier return spikes. Writes a diagnostic report to logs/data_audit.log.",
        cmd_audit_data,
        lambda p: p.add_argument("--log", default=None,
                                  help="Defaults to logs/data_audit.log."),
    ),
    "db-migrate": (
        "Apply any pending paper-trading database schema migrations (tracked via "
        "PRAGMA user_version), without touching existing trade logs or account state.",
        cmd_db_migrate,
        lambda p: p.add_argument("--db", default=str(DEFAULT_DB_PATH)),
    ),
    "export-metrics": (
        "Export one strategy's backtest statistics (gross and net), per-symbol "
        "decision/signal log, equity curve, and factor-regression metrics as "
        "structured JSON (one file) or CSV (four sibling tables) for downstream "
        "visualisation. Trains and computes nothing new -- reads only already-cached "
        "results.",
        cmd_export_metrics,
        lambda p: (
            p.add_argument("--strategy", default="Full+Debate",
                            help="Any strategy from the main ablation or classical baselines "
                                 "(e.g. Full+Debate, Tech+Regime, Buy&Hold, MACD)."),
            p.add_argument("--format", choices=["json", "csv"], default="json"),
            p.add_argument("--output", default=str(DEFAULT_EXPORT_PATH)),
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
