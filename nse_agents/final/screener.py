"""Annual fundamentals from screener.in, parsed with the stdlib only.

Why screener.in: Yahoo exposes four years of statements; an 11-year walk-forward
needs ~12. Screener's public company page carries 12 fiscal years of P&L and
balance sheet for every NSE name.

**What this is not:** a true point-in-time database. Screener shows the latest
*restated* figures, not the figures as first printed. Restatements in large caps
are usually small (Ind-AS transition in FY17 is the big one), but this is a real
limitation and is reported as one. What we *do* control is the disclosure lag:
see ``fundamentals.availability_date``.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from ..config import DATA_CACHE

CACHE = DATA_CACHE / "fundamentals"
CACHE.mkdir(parents=True, exist_ok=True)
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _number(text: str) -> float | None:
    text = text.replace(",", "").replace("%", "").strip()
    if text in ("", "-", "--"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_section(html: str, section_id: str) -> dict[str, dict[str, float | None]]:
    """Return ``{column_header: {row_label: value}}`` for one screener section."""
    start = html.find(f'id="{section_id}"')
    if start < 0:
        return {}
    t0 = html.find("<table", start)
    t1 = html.find("</table>", t0)
    parser = _TableParser()
    parser.feed(html[t0 : t1 + 8])
    if not parser.rows:
        return {}
    header = parser.rows[0][1:]
    out: dict[str, dict[str, float | None]] = {h: {} for h in header}
    for row in parser.rows[1:]:
        if not row:
            continue
        label = re.sub(r"\s*\+$", "", row[0]).strip()
        for h, cell in zip(header, row[1:]):
            out[h][label] = _number(cell)
    return out


def _get(url: str, retries: int) -> str:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            return urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
        except Exception as exc:  # network faults only; parsing happens elsewhere
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def _n_years(section: dict) -> int:
    return sum(1 for k in section if k != "TTM")


def fetch_company(symbol: str, refresh: bool = False, retries: int = 4, min_years: int = 8) -> dict:
    """Fetch (and cache as JSON) annual P&L + balance sheet.

    Consolidated statements are preferred (they include subsidiaries, which is
    what the share price prices). A company with few or no subsidiaries can have
    a consolidated history only a few years long (NESTLEIND: 4 years), in which
    case the standalone statements are used instead and that choice is recorded.
    """
    path = CACHE / f"{symbol.replace('&', '%26')}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if _n_years(cached.get("profit_loss", {})) >= min_years or cached.get("basis") == "standalone":
            return cached
    payload = None
    for basis, suffix in (("consolidated", "consolidated/"), ("standalone", "")):
        url = f"https://www.screener.in/company/{urllib.parse.quote(symbol)}/{suffix}"
        html = _get(url, retries)
        candidate = {
            "symbol": symbol,
            "source": url,
            "basis": basis,
            "fetched": time.strftime("%Y-%m-%d"),
            "profit_loss": parse_section(html, "profit-loss"),
            "balance_sheet": parse_section(html, "balance-sheet"),
        }
        time.sleep(1.5)  # be polite: one page per 1.5 s
        if payload is None or _n_years(candidate["profit_loss"]) > _n_years(payload["profit_loss"]):
            payload = candidate
        if _n_years(payload["profit_loss"]) >= min_years:
            break
    path.write_text(json.dumps(payload, indent=1))
    return payload
