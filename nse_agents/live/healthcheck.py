"""Pre-market system health check: the individual, independently-testable checks
``nse_agents.cli healthcheck`` runs.

Kept as pure functions returning a ``CheckResult`` (never printing, never calling
``sys.exit``) so each check is unit-testable without touching a real price cache,
network, or database file -- the CLI command in ``cli.py`` is the only place that
prints them and decides the process exit code.
"""

from __future__ import annotations

import os
import sqlite3
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def check_price_freshness(
    symbol: str = "RELIANCE",
    max_stale_days: int = 4,
    now: pd.Timestamp | None = None,
) -> CheckResult:
    """The cached price series' most recent date must not be too far behind today.

    ``max_stale_days`` defaults to 4, not 1 -- "updated within the last 24h" is the
    wrong bar for market data: NSE does not trade on weekends, so a Monday-morning
    check against a naive 24h window would false-fail every single week, and a
    check run the day after a market holiday would too. 4 calendar days covers a
    long weekend (Friday close -> Tuesday morning) without covering up a real,
    multi-day staleness problem.
    """
    from ..config import SETTINGS
    from ..data.prices import load_prices

    now = now or pd.Timestamp.now().normalize()
    try:
        frame = load_prices(symbol, SETTINGS.start, "2100-01-01")
    except Exception as exc:
        return CheckResult("price_feed_freshness", False, f"fetch failed: {type(exc).__name__}: {exc}")

    if frame.empty:
        return CheckResult("price_feed_freshness", False, "cached price series is empty")

    latest = pd.Timestamp(frame["date"].max())
    age_days = (now - latest).days
    ok = age_days <= max_stale_days
    detail = (
        f"{symbol}: latest cached bar {latest.date()} ({age_days}d old, "
        f"threshold {max_stale_days}d)"
    )
    return CheckResult("price_feed_freshness", ok, detail)


def check_llm_api_keys() -> list[CheckResult]:
    """Configuration presence, not a live ping -- pinging a real endpoint from a
    healthcheck that runs before every market open would burn API quota on every
    single invocation for a question ("is a key configured") that doesn't need
    one. A live reachability check belongs behind an explicit, opt-in flag, not
    the default fast path.
    """
    results = []
    for env_var, provider in (("ANTHROPIC_API_KEY", "anthropic"), ("OPENAI_API_KEY", "openai")):
        present = bool(os.environ.get(env_var))
        detail = f"{env_var} is set" if present else f"{env_var} not set -- local backend only"
        # Not configured is not a failure: the local Qwen backend runs the whole
        # pipeline without either key, exactly as the walk-forward study does.
        results.append(CheckResult(f"llm_api_key_{provider}", True, detail))
    return results


def check_llm_api_reachable(provider: str, timeout: int = 10) -> CheckResult:
    """An actual, opt-in reachability ping -- a minimal real call, not just a
    "key is set" check, since a present-but-revoked or present-but-wrong key
    would pass ``check_llm_api_keys`` and then fail on the first real use.
    """
    from ..llm.factory import build_backend

    env_var = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(env_var):
        return CheckResult(f"llm_api_reachable_{provider}", False, f"{env_var} not set")
    try:
        backend = build_backend(provider, cache=False)
        backend.complete("Reply with the single word: OK", "ping", max_tokens=5)
        return CheckResult(f"llm_api_reachable_{provider}", True, f"{provider}: reachable")
    except Exception as exc:
        return CheckResult(f"llm_api_reachable_{provider}", False,
                           f"{provider}: {type(exc).__name__}: {exc}")


def check_rss_feed_availability(timeout: int = 15) -> CheckResult:
    from ..data.news import LocalRSSAggregator

    ua = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
    working, broken = [], []
    for source, url in LocalRSSAggregator.FEEDS.items():
        if source in LocalRSSAggregator.BROKEN_SOURCES:
            continue  # already known-broken; not part of the health signal
        try:
            req = urllib.request.Request(url, headers=ua)
            urllib.request.urlopen(req, timeout=timeout).read(256)  # only need headers+a few bytes
            working.append(source)
        except (urllib.error.URLError, TimeoutError) as exc:
            broken.append(f"{source} ({type(exc).__name__})")

    ok = len(working) > 0  # at least one working feed is enough to proceed
    detail = f"working: {working or 'none'}" + (f"; unreachable: {broken}" if broken else "")
    return CheckResult("rss_feed_availability", ok, detail)


def check_paper_trading_db_integrity(db_path: Path | str | None = None) -> CheckResult:
    """SQLite's own ``PRAGMA integrity_check``, plus a check that the three
    tables this project actually relies on exist. A missing DB is not treated
    as a failure -- a fresh checkout with no paper-trading run yet is a valid,
    healthy state, not corruption.
    """
    from .state_store import DEFAULT_DB_PATH

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        return CheckResult("paper_trading_db_integrity", True,
                           f"no DB at {path} yet (not yet run -- not a failure)")

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        (status,) = conn.execute("PRAGMA integrity_check").fetchone()
        if status != "ok":
            return CheckResult("paper_trading_db_integrity", False, f"integrity_check: {status}")

        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        required = {"account", "positions", "trades", "equity_history"}
        missing = required - tables
        if missing:
            return CheckResult("paper_trading_db_integrity", False, f"missing tables: {missing}")
        return CheckResult("paper_trading_db_integrity", True, f"{path}: OK, tables present")
    except sqlite3.Error as exc:
        return CheckResult("paper_trading_db_integrity", False, f"{type(exc).__name__}: {exc}")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_all_checks(db_path: Path | str | None = None, ping_llm: bool = False) -> list[CheckResult]:
    """Every check in the default fast path -- no live LLM pings unless
    ``ping_llm`` is explicitly requested (see ``check_llm_api_reachable``)."""
    results = [check_price_freshness()]
    results.extend(check_llm_api_keys())
    if ping_llm:
        for provider in ("anthropic", "openai"):
            if os.environ.get(f"{provider.upper()}_API_KEY"):
                results.append(check_llm_api_reachable(provider))
    results.append(check_rss_feed_availability())
    results.append(check_paper_trading_db_integrity(db_path))
    return results
