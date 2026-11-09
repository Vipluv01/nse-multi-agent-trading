"""Settings for the final-evaluation extension: wider universe, longer period."""

from __future__ import annotations

import os

from ..config import DATA_CACHE, RESULTS, UNIVERSE

EXT_PRICE_CACHE = DATA_CACHE / "prices_ext"
FINAL = RESULTS / "final"
FINAL.mkdir(parents=True, exist_ok=True)

# The mid-semester ten plus twenty more Nifty 50 constituents, chosen to fill
# sectors the original basket under-represents (autos, pharma, metals,
# utilities, cement, consumer). NB survivorship: these are *today's* index
# members, so the extended universe carries the same survivorship bias as any
# current-constituent study -- stated in the thesis, not hidden.
EXTRA = (
    "KOTAKBANK", "HINDUNILVR", "MARUTI", "SUNPHARMA", "TATASTEEL",
    "NTPC", "POWERGRID", "ULTRACEMCO", "TITAN", "WIPRO",
    "HCLTECH", "ASIANPAINT", "BAJFINANCE", "M&M", "NESTLEIND",
    "ONGC", "COALINDIA", "DRREDDY", "JSWSTEEL", "TECHM",
)
EXT_UNIVERSE: tuple[str, ...] = tuple(UNIVERSE) + EXTRA

EXT_START = "2015-01-01"
EXT_END = "2026-11-07"   # last completed trading week before the final evaluation


def use_extended_cache() -> None:
    """Point the price loader at the extended cache. Call before importing prices."""
    os.environ["NSE_AGENTS_PRICE_CACHE"] = str(EXT_PRICE_CACHE)
