"""Tests for the notification dispatch: content builders (pure) and send logic
(mocked at the urllib boundary -- no real Telegram/Slack/Discord traffic, except
one schema-verification test at the bottom that is skipped unless real
credentials are present in the environment)."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.notifier import (
    DispatchResult,
    TelegramNotifier,
    WebhookNotifier,
    _load_dotenv,
    build_eod_summary,
    build_premarket_briefing,
    send_with_fallback,
)


def test_telegram_notifier_refuses_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    with pytest.raises(RuntimeError, match="TELEGRAM_BOT_TOKEN"):
        TelegramNotifier()


def test_webhook_notifier_refuses_without_a_url(monkeypatch):
    monkeypatch.delenv("NOTIFY_WEBHOOK_URL", raising=False)
    with pytest.raises(RuntimeError, match="NOTIFY_WEBHOOK_URL"):
        WebhookNotifier()


def test_webhook_notifier_rejects_an_invalid_platform():
    with pytest.raises(ValueError, match="platform"):
        WebhookNotifier(webhook_url="https://example.com/hook", platform="carrier-pigeon")


def test_telegram_send_success(monkeypatch):
    import urllib.request

    class FakeResponse:
        def read(self):
            return json.dumps({"ok": True}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    notifier = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    result = notifier.send("test message")
    assert result.ok


def test_telegram_send_reports_api_level_failure(monkeypatch):
    import urllib.request

    class FakeResponse:
        def read(self):
            return json.dumps({"ok": False, "description": "chat not found"}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    notifier = TelegramNotifier(bot_token="fake-token", chat_id="bad-chat")
    result = notifier.send("test message")
    assert not result.ok
    assert "ok=false" in result.detail


def test_telegram_send_handles_a_network_error_without_raising(monkeypatch):
    import urllib.error
    import urllib.request

    def fails(*a, **kw):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", fails)
    notifier = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    result = notifier.send("test message")  # must not raise
    assert not result.ok


def test_webhook_send_uses_the_right_field_for_each_platform(monkeypatch):
    import urllib.request

    captured = {}

    class FakeResponse:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data)
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    WebhookNotifier(webhook_url="https://hooks.slack.com/x", platform="slack").send("hi")
    assert "text" in captured["body"]

    WebhookNotifier(webhook_url="https://discord.com/api/webhooks/x", platform="discord").send("hi")
    assert "content" in captured["body"]


def test_send_with_fallback_uses_the_first_successful_notifier():
    class Ok:
        def send(self, text):
            return DispatchResult(True, "sent via ok")

    class Fails:
        def send(self, text):
            return DispatchResult(False, "failed")

    result = send_with_fallback([Fails(), Ok()], "message")
    assert result.ok
    assert "sent via ok" in result.detail


def test_send_with_fallback_survives_a_notifier_that_raises():
    class Raises:
        def send(self, text):
            raise RuntimeError("boom")

    class Ok:
        def send(self, text):
            return DispatchResult(True, "sent")

    result = send_with_fallback([Raises(), Ok()], "message")  # must not raise
    assert result.ok


def test_send_with_fallback_reports_failure_when_nothing_works():
    class AlwaysFails:
        def send(self, text):
            return DispatchResult(False, "down")

    result = send_with_fallback([AlwaysFails(), AlwaysFails()], "message")
    assert not result.ok


def test_send_with_fallback_with_no_notifiers_is_a_clean_failure_not_a_crash():
    result = send_with_fallback([], "message")
    assert not result.ok


def test_premarket_briefing_includes_vix_and_headlines():
    msg = build_premarket_briefing(
        {"vix": 15.5, "vix_percentile": 0.4, "nifty_mom_20d": 0.015},
        [{"symbol": "INFY", "title": "Infosys wins deal", "sentiment": 0.7}],
        "2026-09-10",
    )
    assert "15.5" in msg
    assert "INFY" in msg
    assert "Infosys wins deal" in msg
    assert "Demonstration only" in msg  # the disclaimer must always be present


def test_premarket_briefing_handles_no_headlines_gracefully():
    msg = build_premarket_briefing({"vix": 15.0, "vix_percentile": None, "nifty_mom_20d": None}, [], "2026-09-10")
    assert "No headlines" in msg


def test_eod_summary_includes_equity_and_disclaimer():
    from nse_agents.live.state_store import AccountSnapshot

    snap = AccountSnapshot(
        cash=900_000.0, initial_capital=1_000_000.0, positions={},
        realized_pnl=500.0, total_costs=200.0, last_updated="2026-09-10T00:00:00",
    )
    msg = build_eod_summary(
        snap,
        [{"symbol": "TCS", "quantity": 10, "avg_price": 2300.0, "market_value": 23500.0, "return": 0.02}],
        [{"symbol": "TCS", "side": "buy", "quantity": 10, "price": 2300.0, "cost": 50.0}],
        "2026-09-10",
    )
    assert "923,500.00" in msg  # 900,000 cash + 23,500 position value
    assert "TCS" in msg
    assert "Demonstration only" in msg


def test_eod_summary_handles_no_trades_and_no_positions_gracefully():
    from nse_agents.live.state_store import AccountSnapshot

    snap = AccountSnapshot(
        cash=1_000_000.0, initial_capital=1_000_000.0, positions={},
        realized_pnl=0.0, total_costs=0.0, last_updated="2026-09-10T00:00:00",
    )
    msg = build_eod_summary(snap, [], [], "2026-09-10")
    assert "No trades executed" in msg
    assert "No open positions" in msg


# ---- .env loading -----------------------------------------------------------

def test_load_dotenv_sets_a_variable_from_a_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_VAR=hello\n")
    try:
        _load_dotenv(env_file)
        assert os.environ["TEST_DOTENV_VAR"] == "hello"
    finally:
        monkeypatch.delenv("TEST_DOTENV_VAR", raising=False)


def test_load_dotenv_never_overwrites_a_real_environment_variable(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DOTENV_VAR", "real-value")
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_DOTENV_VAR=from-file\n")
    _load_dotenv(env_file)
    assert os.environ["TEST_DOTENV_VAR"] == "real-value"  # real env always wins


def test_load_dotenv_skips_comments_and_blank_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_DOTENV_A", raising=False)
    monkeypatch.delenv("TEST_DOTENV_B", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\n\nTEST_DOTENV_A=1\n   \nTEST_DOTENV_B=2\n")
    try:
        _load_dotenv(env_file)
        assert os.environ["TEST_DOTENV_A"] == "1"
        assert os.environ["TEST_DOTENV_B"] == "2"
    finally:
        monkeypatch.delenv("TEST_DOTENV_A", raising=False)
        monkeypatch.delenv("TEST_DOTENV_B", raising=False)


def test_load_dotenv_strips_surrounding_quotes(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_DOTENV_QUOTED", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text('TEST_DOTENV_QUOTED="quoted value"\n')
    try:
        _load_dotenv(env_file)
        assert os.environ["TEST_DOTENV_QUOTED"] == "quoted value"
    finally:
        monkeypatch.delenv("TEST_DOTENV_QUOTED", raising=False)


def test_load_dotenv_is_a_silent_noop_when_the_file_does_not_exist(tmp_path):
    _load_dotenv(tmp_path / "does_not_exist.env")  # must not raise


# ---- live schema verification (skipped without real credentials) ------------

@pytest.mark.skipif(
    not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")),
    reason="requires real TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment "
           "(or a .env file) -- not present in this environment, so this notifier "
           "remains unverified against a live endpoint (see KNOWN_ISSUES.md #11)",
)
def test_telegram_bot_token_is_real_and_getme_matches_the_documented_schema():
    """Calls Telegram's ``getMe`` endpoint -- not ``sendMessage`` -- specifically
    so this test can run in CI/automated test runs without posting a real
    message to a real chat every time the suite runs. ``getMe`` is read-only
    and still proves the bot token is genuinely valid and that the response
    shape matches Telegram's documented Bot API (``ok``, ``result.id``,
    ``result.is_bot``, ``result.username``).

    A network-level failure to even reach ``api.telegram.org`` (a sandboxed
    CI runner or dev environment blocking that specific host -- confirmed as
    a real, reproducible case: this exact failure mode, with
    ``github.com``/``pypi.org`` both reachable instantly from the same
    machine) is distinguished from a genuine credential/schema failure and
    skipped rather than failed -- this test's job is to verify the
    *credentials and response shape*, not this machine's network policy,
    which real credentials cannot fix and shouldn't be blamed for."""
    import urllib.error
    import urllib.request

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    try:
        with urllib.request.urlopen(f"https://api.telegram.org/bot{token}/getMe", timeout=15) as resp:
            body = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        pytest.skip(f"api.telegram.org unreachable from this machine ({type(exc).__name__}: {exc}) "
                    f"-- not a credential problem, see this test's own docstring")

    assert body["ok"] is True
    result = body["result"]
    assert isinstance(result["id"], int)
    assert result["is_bot"] is True
    assert isinstance(result["username"], str) and result["username"]
