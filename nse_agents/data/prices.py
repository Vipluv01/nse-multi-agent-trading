"""NSE OHLCV loader.

Uses the public Yahoo Finance chart endpoint directly rather than depending on
``yfinance``: one HTTP call with a documented JSON shape is easier to pin,
cache and reason about than a scraping library that changes under us.

All series are **split/bonus adjusted**. Indian large caps bonus-issue often
enough (RELIANCE 1:1 in 2017, INFY 1:1 in 2018) that an unadjusted price series
manufactures fake -50% single-day returns, which a directional model will
happily learn.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_CACHE

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

# The study's price cache is a frozen snapshot. Yahoo rewrites *past* adjusted
# prices whenever a dividend is paid (and has also removed phantom holiday bars
# it once served), so refreshing this folder in place silently changes the data
# behind every published number. Other consumers use their own folder, selected
# per call by NSE_AGENTS_PRICE_CACHE:
#   data_cache/prices       -- the walk-forward study (default; never refreshed)
#   data_cache/prices_ext   -- the final-evaluation extended universe/period
#   data_cache/prices_live  -- the live/paper-trading pipeline (refreshed daily)
STUDY_PRICE_CACHE = DATA_CACHE / "prices"
LIVE_PRICE_CACHE = DATA_CACHE / "prices_live"
PRICE_CACHE = Path(os.environ.get("NSE_AGENTS_PRICE_CACHE", STUDY_PRICE_CACHE))
PRICE_CACHE.mkdir(parents=True, exist_ok=True)


def price_cache_dir() -> Path:
    """The cache folder for this call (env is read per call, not at import)."""
    path = Path(os.environ.get("NSE_AGENTS_PRICE_CACHE", STUDY_PRICE_CACHE))
    path.mkdir(parents=True, exist_ok=True)
    return path


def use_live_price_cache() -> None:
    """Called by every live entry point before it loads any price."""
    os.environ["NSE_AGENTS_PRICE_CACHE"] = str(LIVE_PRICE_CACHE)


class FrozenCacheError(RuntimeError):
    """A refresh was requested against the frozen study cache."""


def _yahoo_symbol(symbol: str) -> str:
    """NSE tickers need the ``.NS`` suffix; indices like ``^NSEI`` do not."""
    return symbol if symbol.startswith("^") else f"{symbol}.NS"


def _fetch(symbol: str, start: str, end: str, retries: int = 5) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = (
        _CHART.format(sym=_yahoo_symbol(symbol))
        + f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            payload = json.loads(urllib.request.urlopen(req, timeout=30).read())
            break
        # IncompleteRead / ConnectionError: a truncated chunked transfer is a
        # transient network fault exactly like a timeout, and must be retried
        # rather than abort a 30-symbol fetch on its first flaky response.
        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
            http.client.IncompleteRead,
            ConnectionError,
        ) as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))
    else:
        raise RuntimeError(f"failed to fetch {symbol}: {last}")

    result = payload["chart"]["result"][0]
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert(
                "Asia/Kolkata"
            ).normalize().tz_localize(None),
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        }
    )
    adj = result["indicators"].get("adjclose")
    if adj:
        frame["adj_close"] = adj[0]["adjclose"]
    else:
        frame["adj_close"] = frame["close"]

    frame = frame.dropna(subset=["open", "high", "low", "close", "adj_close"])
    # Scale the whole bar by the close->adj_close ratio so that OHLC stay
    # internally consistent (high >= open/close >= low) after adjustment.
    ratio = frame["adj_close"] / frame["close"]
    for col in ("open", "high", "low"):
        frame[col] = frame[col] * ratio
    frame["close"] = frame["adj_close"]
    frame = frame.drop(columns=["adj_close"])
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.sort_values("date").reset_index(drop=True)


def load_prices(
    symbol: str, start: str, end: str, refresh: bool = False
) -> pd.DataFrame:
    """Return adjusted daily OHLCV for one symbol, caching to disk."""
    cache = price_cache_dir()
    path: Path = cache / f"{symbol.replace('^', 'IDX_')}.csv"
    if refresh and path.exists() and cache.resolve() == STUDY_PRICE_CACHE.resolve():
        # This happened once (live/snapshot.py, Nov 2026) and rewrote all ten
        # study files; see KNOWN_ISSUES.md #18. Refusing is the only safe default.
        raise FrozenCacheError(
            f"refusing to refresh {path}: the study cache is frozen. Live code must "
            "call nse_agents.data.prices.use_live_price_cache() first."
        )
    if path.exists() and not refresh:
        frame = pd.read_csv(path, parse_dates=["date"])
    else:
        frame = _fetch(symbol, start, end)
        frame.to_csv(path, index=False)
    mask = (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    return frame.loc[mask].reset_index(drop=True)


def load_panel(
    symbols: tuple[str, ...] | list[str], start: str, end: str, refresh: bool = False
) -> pd.DataFrame:
    """Long-format panel of every symbol, with a ``symbol`` column."""
    frames = []
    for sym in symbols:
        frame = load_prices(sym, start, end, refresh=refresh)
        frame = frame.assign(symbol=sym)
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def forward_return(frame: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """Open-to-open return over ``horizon`` days, aligned to the decision date.

    The value at row ``t`` is the return earned by a position opened at the
    open of ``t+1`` and closed at the open of ``t+1+horizon``. Decisions are
    made on information available at the close of ``t``, so this is the only
    return series that is actually attainable. Using close-to-close here --
    the common shortcut -- silently assumes execution at a price that was
    already known when the signal fired.
    """
    open_ = frame["open"].to_numpy(dtype=float)
    fwd = np.full(len(frame), np.nan)
    entry = open_[1:]
    if horizon < len(open_) - 1:
        exit_ = open_[1 + horizon :]
        n = len(exit_)
        fwd[:n] = exit_ / entry[:n] - 1.0
    return pd.Series(fwd, index=frame.index, name=f"fwd_ret_{horizon}d")
