"""Indian financial news headlines, per symbol, per day.

Two providers behind one protocol:

* ``GoogleNewsRSS`` -- free, no key, real Indian-source headlines. Queried in
  month-long windows using the ``after:``/``before:`` search operators, which
  reaches further back than an undated query, though Google still caps how
  much history it will surface.
* ``CachedCorpus`` -- replays a JSONL file written by a previous fetch, so a
  backtest is reproducible and does not re-hit the network.

The headline corpus is the binding constraint on this project's sentiment
evaluation, and its true extent is measured in ``coverage_report`` rather than
assumed. Nothing here synthesises a headline: if no news exists for a symbol on
a date, the sentiment agent abstains and says so.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from ..config import DATA_CACHE

NEWS_CACHE = DATA_CACHE / "news"
NEWS_CACHE.mkdir(parents=True, exist_ok=True)

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Query strings, not just tickers: "SBIN" returns nothing useful, "State Bank
# of India" returns the actual news flow.
COMPANY_NAMES: dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "BHARTIARTL": "Bharti Airtel",
    "ITC": "ITC Limited",
    "LT": "Larsen and Toubro",
    "AXISBANK": "Axis Bank",
}


# Headlines published after the NSE close (15:30 IST == 10:00 GMT) are not
# knowable to a decision made at that day's close, so they are attributed to
# the following calendar day. Skipping this attribution step is a subtle and
# very common lookahead leak: an evening earnings-reaction headline would
# otherwise be fed to a model deciding at that afternoon's close.
NSE_CLOSE_GMT_HOUR = 10


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    date: str  # ISO date: the trading day this headline is *actionable* on
    title: str
    source: str
    url: str = ""
    published_utc: str = ""  # full timestamp as published, kept for audit

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def actionable_date(published: datetime) -> str:
    """Map a publication timestamp to the first trading day it can inform.

    ``published`` is GMT, as Google News RSS reports it. Anything at or after
    10:00 GMT (15:30 IST, the NSE close) belongs to the next day.
    """
    day = published.date()
    if published.hour >= NSE_CLOSE_GMT_HOUR:
        day = day + timedelta(days=1)
    return day.isoformat()


# Headline patterns that carry no information about the future: broker-SEO
# filler, live-price pages, and "prediction for tomorrow" clickbait. Left in,
# they dominate the corpus by volume and drown the real news flow.
_NOISE = re.compile(
    r"(prediction for tomorrow|share price target|live nse|stock price & chart|"
    r"technical analysis|price live|buy or sell|multibagger|should you buy|"
    r"stocks to watch|f&o levels|share price today live)",
    re.I,
)


def is_noise(title: str) -> bool:
    return bool(_NOISE.search(title))


class NewsProvider(Protocol):
    def fetch(self, symbol: str, start: str, end: str) -> list[NewsItem]: ...


def _strip_source_suffix(title: str) -> tuple[str, str]:
    """Google News appends ' - Publisher'. Split it off; it is metadata."""
    parts = title.rsplit(" - ", 1)
    if len(parts) == 2 and len(parts[1]) < 45:
        return parts[0].strip(), parts[1].strip()
    return title.strip(), ""


def _unescape(text: str) -> str:
    for a, b in (("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"'), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(a, b)
    return text


class GoogleNewsRSS:
    """Month-windowed Google News RSS search, restricted to the India edition."""

    def __init__(self, pause: float = 1.2, retries: int = 2):
        self.pause = pause
        self.retries = retries

    def _query(self, term: str, after: str, before: str) -> str:
        q = urllib.parse.quote(f'{term} share price after:{after} before:{before}')
        return f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"

    def _get(self, url: str) -> str:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(url, headers=_UA)
                return urllib.request.urlopen(req, timeout=30).read().decode("utf8", "ignore")
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"news fetch failed: {last}")

    def fetch(self, symbol: str, start: str, end: str) -> list[NewsItem]:
        term = COMPANY_NAMES.get(symbol, symbol)
        items: list[NewsItem] = []
        seen: set[str] = set()
        windows = pd.date_range(start, end, freq="MS")
        for w_start in windows:
            w_end = w_start + pd.offsets.MonthBegin(1)
            xml = self._get(
                self._query(term, w_start.date().isoformat(), w_end.date().isoformat())
            )
            for block in re.findall(r"<item>(.*?)</item>", xml, flags=re.S):
                title_m = re.search(r"<title>(.*?)</title>", block, flags=re.S)
                date_m = re.search(r"<pubDate>(.*?)</pubDate>", block, flags=re.S)
                link_m = re.search(r"<link>(.*?)</link>", block, flags=re.S)
                if not (title_m and date_m):
                    continue
                raw_title = _unescape(title_m.group(1)).strip()
                title, source = _strip_source_suffix(raw_title)
                if not title or title.lower() in seen:
                    continue
                seen.add(title.lower())
                try:
                    stamp = datetime.strptime(
                        date_m.group(1).strip(), "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except ValueError:
                    continue
                items.append(
                    NewsItem(
                        symbol=symbol,
                        date=actionable_date(stamp),
                        title=title,
                        source=source,
                        url=link_m.group(1).strip() if link_m else "",
                        published_utc=stamp.isoformat(),
                    )
                )
            time.sleep(self.pause)
        return sorted(items, key=lambda i: i.date)


class CachedCorpus:
    """Reads the JSONL corpus on disk. The provider used by every backtest."""

    def __init__(self, path: Path | str | None = None):
        # Coerce: a str path is truthy and would otherwise flow through as a
        # str and fail on the first Path method.
        self.path = Path(path) if path else (NEWS_CACHE / "headlines.jsonl")

    def all_items(self) -> list[NewsItem]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(NewsItem(**json.loads(line)))
        return out

    def fetch(self, symbol: str, start: str, end: str) -> list[NewsItem]:
        return [
            i
            for i in self.all_items()
            if i.symbol == symbol and start <= i.date <= end
        ]


def write_corpus(items: Iterable[NewsItem], path: Path | str | None = None) -> Path:
    """Persist headlines, de-duplicated on (symbol, date, title)."""
    path = Path(path) if path else (NEWS_CACHE / "headlines.jsonl")
    existing = CachedCorpus(path).all_items()
    keyed = {(i.symbol, i.date, i.title): i for i in existing}
    for item in items:
        keyed[(item.symbol, item.date, item.title)] = item
    with path.open("w") as fh:
        for item in sorted(keyed.values(), key=lambda i: (i.symbol, i.date)):
            fh.write(item.to_json() + "\n")
    return path


def to_frame(items: Iterable[NewsItem]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(i) for i in items])
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "date", "title", "source", "url"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def align_to_trading_days(
    frame: pd.DataFrame, trading_dates: Iterable[pd.Timestamp], date_column: str = "date"
) -> pd.DataFrame:
    """Roll each headline's actionable date forward to the next trading session.

    ``actionable_date`` only moves after-close news to the next *calendar* day,
    which is not enough: Friday-evening, weekend and market-holiday news lands
    on a date the exchange never opened. Joining that against a price series
    silently discards it -- 16% of this corpus, and disproportionately the
    weekend corporate-news cycle, which is exactly the flow a Monday-open
    decision should be reading.

    Rolling forward (never backward) preserves causality: a Saturday headline
    informs Monday, never Friday.
    """
    sessions = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trading_dates).unique()))
    if frame.empty or len(sessions) == 0:
        return frame.assign(**{date_column: pd.to_datetime(frame.get(date_column, pd.Series(dtype="datetime64[ns]")))})

    out = frame.copy()
    out[date_column] = pd.to_datetime(out[date_column])
    rolled = pd.merge_asof(
        out[[date_column]].reset_index().sort_values(date_column),
        pd.DataFrame({date_column: sessions, "_session": sessions}),
        on=date_column,
        direction="forward",
    ).set_index("index")
    out[date_column] = rolled["_session"]
    # Headlines after the last available session have no future day to act on.
    return out.dropna(subset=[date_column]).reset_index(drop=True)


def coverage_report(items: Iterable[NewsItem]) -> pd.DataFrame:
    """Per-symbol headline counts and true date span.

    Reported in the results rather than described in prose, because the honest
    span of the corpus is the main threat to validity for the sentiment arm.
    """
    frame = to_frame(items)
    if frame.empty:
        return frame
    return (
        frame.groupby("symbol")
        .agg(
            headlines=("title", "count"),
            days_covered=("date", "nunique"),
            first=("date", "min"),
            last=("date", "max"),
        )
        .reset_index()
    )
