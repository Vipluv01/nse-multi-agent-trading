"""Build and send the end-of-day execution summary (P&L, trades, positions, drawdown).

Meant to be invoked by a scheduler (cron/launchd) around market close -- see the cron
example in README.md.

Usage:
  .venv/bin/python scripts/send_eod_summary.py
  .venv/bin/python scripts/send_eod_summary.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import SETTINGS
from nse_agents.live.notifier import (
    TelegramNotifier,
    WebhookNotifier,
    build_eod_summary,
    send_with_fallback,
)
from nse_agents.live.state_store import DEFAULT_DB_PATH, PaperTradingStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no paper-trading state at {db_path} -- run scripts/run_live_signal.py "
              f"--persist at least once first.")
        return 1

    store = PaperTradingStore(db_path)
    snap = store.snapshot()

    from nse_agents.data.prices import load_prices

    prices = {}
    for symbol in snap.positions:
        try:
            prices[symbol] = float(load_prices(symbol, SETTINGS.start, "2100-01-01")["close"].iloc[-1])
        except Exception:
            continue

    positions_detail = [
        {"symbol": sym, "quantity": pos.quantity, "avg_price": pos.avg_price,
         "market_value": pos.market_value(prices[sym]) if sym in prices else 0.0,
         "return": (prices[sym] / pos.avg_price - 1.0) if sym in prices else 0.0}
        for sym, pos in snap.positions.items()
    ]

    today = pd.Timestamp.now().date().isoformat()
    trades = store.trade_log()
    fills_today = trades[trades["date"] == today].to_dict("records") if len(trades) else []

    text = build_eod_summary(snap, positions_detail, fills_today, today)

    if args.dry_run:
        print(text)
        return 0

    notifiers = []
    try:
        notifiers.append(TelegramNotifier())
    except RuntimeError as exc:
        print(f"Telegram not configured: {exc}", file=sys.stderr)
    try:
        notifiers.append(WebhookNotifier())
    except RuntimeError as exc:
        print(f"Webhook not configured: {exc}", file=sys.stderr)

    if not notifiers:
        print("\nNo notifier configured (set TELEGRAM_BOT_TOKEN+TELEGRAM_CHAT_ID or "
              "NOTIFY_WEBHOOK_URL) -- printing instead:\n")
        print(text)
        return 1

    result = send_with_fallback(notifiers, text)
    print(f"{'sent' if result.ok else 'FAILED'}: {result.detail}")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
