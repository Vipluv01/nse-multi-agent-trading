"""Build and send the pre-market briefing (VIX/Nifty regime + top headlines).

Meant to be invoked by a scheduler (cron/launchd) around market open -- this script
itself does not schedule anything; see the cron example in README.md.

Usage:
  .venv/bin/python scripts/send_premarket_briefing.py                 # local backend
  .venv/bin/python scripts/send_premarket_briefing.py --dry-run       # print, don't send
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS, SETTINGS
from nse_agents.live.notifier import (
    TelegramNotifier,
    WebhookNotifier,
    build_premarket_briefing,
    send_with_fallback,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the briefing instead of sending it -- always safe, "
                          "needs no credentials.")
    args = ap.parse_args()

    from nse_agents.agents.regime import build_regime_table

    watch_symbol = SETTINGS.universe[0]
    regime = build_regime_table(symbols=(watch_symbol,), start=SETTINGS.start, end="2100-01-01")
    latest = regime.iloc[-1].to_dict() if len(regime) else {}

    sentiment_path = RESULTS / "sentiment" / "daily_local.csv"
    top_headlines: list[dict] = []
    if sentiment_path.exists():
        daily = pd.read_csv(sentiment_path, parse_dates=["date"])
        today = daily[daily["date"] == daily["date"].max()]
        top_headlines = [
            {"symbol": row.symbol, "title": f"mean tone {row.sentiment:+.2f} over {row.n_headlines} headline(s)",
             "sentiment": row.sentiment}
            for row in today.sort_values("sentiment", ascending=False).itertuples()
        ]

    as_of = pd.Timestamp.now().date().isoformat()
    text = build_premarket_briefing(latest, top_headlines, as_of)

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
