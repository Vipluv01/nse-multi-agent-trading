"""Tests for the pre-market health checks."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.live.healthcheck import (
    check_llm_api_keys,
    check_paper_trading_db_integrity,
    check_price_freshness,
    check_rss_feed_availability,
)


def test_price_freshness_passes_for_a_recent_bar(monkeypatch):
    import nse_agents.live.healthcheck as hc

    fake = pd.DataFrame({"date": pd.bdate_range("2026-08-01", periods=30)})
    monkeypatch.setattr(hc, "load_prices", lambda *a, **kw: fake, raising=False)
    import nse_agents.data.prices as prices_mod

    monkeypatch.setattr(prices_mod, "load_prices", lambda *a, **kw: fake)

    now = fake["date"].max() + pd.Timedelta(days=1)  # 1 day after the latest bar
    result = check_price_freshness(now=now)
    assert result.ok


def test_price_freshness_fails_for_a_stale_bar(monkeypatch):
    import nse_agents.data.prices as prices_mod

    fake = pd.DataFrame({"date": pd.bdate_range("2026-08-01", periods=30)})
    monkeypatch.setattr(prices_mod, "load_prices", lambda *a, **kw: fake)

    now = fake["date"].max() + pd.Timedelta(days=10)  # well past the 4-day threshold
    result = check_price_freshness(now=now, max_stale_days=4)
    assert not result.ok


def test_price_freshness_does_not_false_fail_over_a_weekend(monkeypatch):
    """A Friday close checked on Monday morning (3 calendar days later) must
    pass -- this is the exact false-positive a naive '24h' bar would produce
    every single week."""
    import nse_agents.data.prices as prices_mod

    friday = pd.Timestamp("2026-09-04")  # a Friday
    fake = pd.DataFrame({"date": pd.bdate_range("2026-08-01", end=friday)})
    monkeypatch.setattr(prices_mod, "load_prices", lambda *a, **kw: fake)

    monday_morning = pd.Timestamp("2026-09-07")  # the following Monday
    result = check_price_freshness(now=monday_morning)
    assert result.ok


def test_llm_api_key_absence_is_not_a_failure(monkeypatch):
    """No key configured must report ok=True -- the local backend runs the
    whole pipeline without either key, exactly as the study itself does."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    results = check_llm_api_keys()
    assert all(r.ok for r in results)
    assert any("not set" in r.detail for r in results)


def test_llm_api_key_presence_is_reported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-for-test")
    results = check_llm_api_keys()
    anthropic_result = next(r for r in results if "anthropic" in r.name)
    assert anthropic_result.ok
    assert "is set" in anthropic_result.detail


def test_rss_feed_check_reports_working_feeds(monkeypatch):
    import urllib.request

    class FakeResponse:
        def read(self, n=None):
            return b"<rss></rss>"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: FakeResponse())
    result = check_rss_feed_availability()
    assert result.ok
    assert "economic_times" in result.detail


def test_rss_feed_check_fails_when_every_feed_is_unreachable(monkeypatch):
    import urllib.error
    import urllib.request

    def always_fails(*a, **kw):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(urllib.request, "urlopen", always_fails)
    result = check_rss_feed_availability()
    assert not result.ok


def test_db_integrity_check_passes_when_no_db_exists_yet(tmp_path):
    result = check_paper_trading_db_integrity(tmp_path / "nonexistent.sqlite")
    assert result.ok
    assert "not yet run" in result.detail


def test_db_integrity_check_passes_for_a_real_store(tmp_path):
    from nse_agents.live.state_store import PaperTradingStore

    path = tmp_path / "state.sqlite"
    PaperTradingStore(path, initial_capital=1_000_000.0)
    result = check_paper_trading_db_integrity(path)
    assert result.ok


def test_db_integrity_check_fails_for_a_corrupted_file(tmp_path):
    path = tmp_path / "corrupt.sqlite"
    path.write_bytes(b"not a real sqlite file")
    result = check_paper_trading_db_integrity(path)
    assert not result.ok


def test_db_integrity_check_fails_when_a_required_table_is_missing(tmp_path):
    import sqlite3

    path = tmp_path / "incomplete.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE account (id INTEGER)")
    conn.commit()
    conn.close()
    result = check_paper_trading_db_integrity(path)
    assert not result.ok
    assert "missing" in result.detail
