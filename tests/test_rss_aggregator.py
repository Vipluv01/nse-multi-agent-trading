"""Tests for LocalRSSAggregator and its dedup cache.

Network-independent: feedparser's own parsing is trusted (it is a well-tested
library), so what's tested here is this project's code -- company-name
filtering, actionable-date attribution, and dedup -- against a synthetic feed
response shaped like the real one.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.data.news import LocalRSSAggregator, NewsItem, load_rss_cache, save_rss_cache


def test_moneycontrol_is_registered_but_marked_broken():
    """The feed exists in the source list (documented, not silently dropped)
    but is excluded from fetch_all -- it returns HTTP 403 in practice."""
    assert "moneycontrol_markets" in LocalRSSAggregator.FEEDS
    assert "moneycontrol_markets" in LocalRSSAggregator.BROKEN_SOURCES


def test_fetch_all_filters_by_company_name(monkeypatch):
    agg = LocalRSSAggregator(pause=0)

    def fake_fetch_one(self, source, url):
        return [
            {"source": source, "title": "Infosys Q2 profit beats estimates",
             "link": "https://x", "published_utc": datetime(2026, 9, 9, 10, 0)},
            {"source": source, "title": "Gold prices rally on Fed rate cut hopes",
             "link": "https://y", "published_utc": datetime(2026, 9, 9, 10, 0)},
        ]

    monkeypatch.setattr(LocalRSSAggregator, "_fetch_one", fake_fetch_one)
    items = agg.fetch_all(symbols=("INFY", "RELIANCE"))
    assert len(items) >= 1
    assert all(i.symbol == "INFY" for i in items)
    assert all("Infosys" in i.title for i in items)


def test_fetch_all_attributes_after_close_utc_timestamps_to_next_day(monkeypatch):
    """published_utc is already UTC (feedparser normalises the feed's own
    timezone) -- actionable_date must still apply the same >=10:00 GMT cutoff
    used throughout the rest of the project."""
    agg = LocalRSSAggregator(pause=0)

    def fake_fetch_one(self, source, url):
        return [{"source": source, "title": "Infosys wins large deal", "link": "https://x",
                  "published_utc": datetime(2026, 9, 9, 11, 30)}]  # after the 10:00 GMT cutoff

    monkeypatch.setattr(LocalRSSAggregator, "_fetch_one", fake_fetch_one)
    items = agg.fetch_all(symbols=("INFY",))
    assert items[0].date == "2026-09-10"  # next day


def test_fetch_all_records_errors_without_crashing(monkeypatch, capsys):
    agg = LocalRSSAggregator(pause=0)

    def failing_fetch_one(self, source, url):
        return [{"_error": f"{source}: simulated failure"}]

    monkeypatch.setattr(LocalRSSAggregator, "_fetch_one", failing_fetch_one)
    items = agg.fetch_all(symbols=("INFY",))
    assert items == []  # no crash, just nothing matched


def test_rss_cache_deduplicates_on_symbol_title_and_timestamp(tmp_path):
    path = tmp_path / "cache.json"
    item = NewsItem(symbol="INFY", date="2026-09-09", title="Infosys wins deal",
                     source="economic_times_stocks", url="https://x",
                     published_utc="2026-09-09T10:00:00")
    save_rss_cache([item], path)
    save_rss_cache([item], path)  # identical item again
    assert len(load_rss_cache(path)) == 1


def test_rss_cache_keeps_distinct_items_with_the_same_title(tmp_path):
    """Two genuinely different headlines published hours apart, with titles
    that happen to collide, must not be collapsed into one."""
    path = tmp_path / "cache.json"
    early = NewsItem(symbol="INFY", date="2026-09-09", title="Market update",
                      source="economic_times_markets", url="https://a",
                      published_utc="2026-09-09T04:00:00")
    later = NewsItem(symbol="INFY", date="2026-09-09", title="Market update",
                      source="economic_times_markets", url="https://b",
                      published_utc="2026-09-09T09:00:00")
    save_rss_cache([early, later], path)
    assert len(load_rss_cache(path)) == 2
