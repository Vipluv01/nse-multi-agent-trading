"""Daily notification dispatch: Telegram Bot API and generic webhooks (Slack/Discord).

Two things kept deliberately separate:

* **Sending** (``TelegramNotifier``, ``WebhookNotifier``) -- thin, testable wrappers
  around one HTTP POST each. Neither can be exercised against a real endpoint in this
  environment (no bot token, no webhook URL configured), so each raises a clear,
  specific "not configured" error rather than failing silently or pretending to send.
* **Content** (``build_premarket_briefing``, ``build_eod_summary``) -- pure functions
  that take already-computed data and return a formatted message. These need no
  network access to test, and are exactly what a caller wires up to a scheduler (cron,
  launchd, a CI job) -- **this module does not schedule anything itself.** "Daily at
  9:00 AM IST" and "daily at 3:30 PM IST" are the operating system's job, not this
  project's; see the cron/launchd snippet in README.md rather than a background daemon
  silently started here.

Uses ``urllib`` (stdlib), not a new ``requests`` dependency -- matching the convention
already used throughout ``nse_agents/data/`` rather than introducing a second HTTP
library for one more feature.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DispatchResult:
    ok: bool
    detail: str


class TelegramNotifier:
    def __init__(self, bot_token: str | None = None, chat_id: str | None = None):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        if not self.bot_token or not self.chat_id:
            raise RuntimeError(
                "TelegramNotifier needs TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID "
                "(env vars or constructor args). Neither is set in this environment, "
                "so this has never been run against a real Telegram bot -- verify the "
                "message renders correctly (Markdown escaping in particular) on the "
                "first real send."
            )

    def send(self, text: str, timeout: int = 15) -> DispatchResult:
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({
            "chat_id": self.chat_id, "text": text, "parse_mode": "Markdown",
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = json.loads(resp.read())
            if not body.get("ok"):
                return DispatchResult(False, f"Telegram API returned ok=false: {body}")
            return DispatchResult(True, f"sent to chat {self.chat_id}")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return DispatchResult(False, f"{type(exc).__name__}: {exc}")


class WebhookNotifier:
    """Generic incoming-webhook sender -- Slack and Discord both accept a JSON
    POST with a ``content`` (Discord) or ``text`` (Slack) field; ``platform``
    picks which shape to send, since the two are not interchangeable.
    """

    def __init__(self, webhook_url: str | None = None, platform: str = "slack"):
        self.webhook_url = webhook_url or os.environ.get("NOTIFY_WEBHOOK_URL")
        if platform not in ("slack", "discord"):
            raise ValueError(f"platform must be 'slack' or 'discord', got {platform!r}")
        self.platform = platform
        if not self.webhook_url:
            raise RuntimeError(
                "WebhookNotifier needs NOTIFY_WEBHOOK_URL (env var or constructor arg) "
                "-- not set in this environment, so this has never been run against a "
                "real Slack/Discord webhook."
            )

    def send(self, text: str, timeout: int = 15) -> DispatchResult:
        key = "text" if self.platform == "slack" else "content"
        payload = json.dumps({key: text}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url, data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
            ok = 200 <= status < 300
            return DispatchResult(ok, f"HTTP {status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            return DispatchResult(False, f"{type(exc).__name__}: {exc}")


def send_with_fallback(notifiers: list, text: str) -> DispatchResult:
    """Try each notifier in order; return the first success. Mirrors
    ``EnsembleBackend``'s fallback shape (nse_agents/agents/ensemble.py) for the
    same reason: one channel being down (a bad webhook, a revoked bot token)
    must not silently drop the whole day's notification if a second channel
    is configured and working.
    """
    last: DispatchResult | None = None
    for notifier in notifiers:
        try:
            result = notifier.send(text)
        except Exception as exc:  # noqa: BLE001
            result = DispatchResult(False, f"{type(exc).__name__}: {exc}")
        if result.ok:
            return result
        last = result
    return last or DispatchResult(False, "no notifiers configured")


# ---- message content (pure functions, no network) -------------------------------

DISCLAIMER = (
    "_Demonstration only. No configuration in this study has a demonstrated "
    "market-beating edge net of real costs -- see README.md._"
)


def build_premarket_briefing(regime_row: dict, top_headlines: list[dict], as_of: str) -> str:
    """``regime_row``: one row's worth of macro fields (vix, vix_percentile,
    nifty_mom_20d) as already computed by ``build_regime_table``.
    ``top_headlines``: list of {symbol, title, sentiment} dicts, already scored
    -- this function formats, it does not fetch or score anything itself.
    """
    lines = [f"*Pre-Market Briefing -- {as_of}*", ""]

    vix = regime_row.get("vix")
    vix_pct = regime_row.get("vix_percentile")
    nifty_mom = regime_row.get("nifty_mom_20d")
    if vix is not None:
        pct_str = f" ({vix_pct:.0%} percentile)" if vix_pct is not None and not pd.isna(vix_pct) else ""
        lines.append(f"India VIX: {vix:.2f}{pct_str}")
    if nifty_mom is not None and not pd.isna(nifty_mom):
        lines.append(f"Nifty 20-day momentum: {nifty_mom:+.2%}")
    lines.append("")

    if top_headlines:
        lines.append("*Top sentiment headlines:*")
        for item in top_headlines[:5]:
            sentiment = item.get("sentiment")
            tag = "+" if sentiment is not None and sentiment > 0 else "-" if sentiment is not None and sentiment < 0 else "?"
            lines.append(f"[{tag}] {item.get('symbol', '?')}: {item.get('title', '')}")
    else:
        lines.append("_No headlines scored for today._")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def build_eod_summary(snapshot, positions_detail: list[dict], fills_today: list[dict], as_of: str) -> str:
    """``snapshot``: an ``AccountSnapshot`` (nse_agents.live.state_store).
    ``positions_detail``: [{symbol, quantity, avg_price, market_value, return}].
    ``fills_today``: [{symbol, side, quantity, price, cost}] -- may be empty.
    """
    total_equity = snapshot.cash + sum(p["market_value"] for p in positions_detail)
    total_return = total_equity / snapshot.initial_capital - 1.0

    lines = [f"*EOD Execution Summary -- {as_of}*", ""]
    lines.append(f"Total equity: Rs {total_equity:,.2f} ({total_return:+.2%})")
    lines.append(f"Cash: Rs {snapshot.cash:,.2f}")
    lines.append(f"Realised P&L: Rs {snapshot.realized_pnl:,.2f}")
    lines.append(f"Costs paid today's cumulative: Rs {snapshot.total_costs:,.2f}")
    lines.append("")

    if fills_today:
        lines.append("*Executed today:*")
        for fill in fills_today:
            lines.append(
                f"  {fill['side'].upper()} {fill['quantity']:.2f} {fill['symbol']} "
                f"@ Rs {fill['price']:.2f}"
            )
    else:
        lines.append("_No trades executed today._")
    lines.append("")

    if positions_detail:
        lines.append("*Open positions:*")
        for pos in positions_detail:
            lines.append(f"  {pos['symbol']}: {pos['quantity']:.2f} @ Rs {pos['avg_price']:.2f} "
                        f"(mkt Rs {pos['market_value']:,.2f}, {pos['return']:+.2%})")
    else:
        lines.append("_No open positions._")

    lines.append("")
    lines.append(DISCLAIMER)
    return "\n".join(lines)
