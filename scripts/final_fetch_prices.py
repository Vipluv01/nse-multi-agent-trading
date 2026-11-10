"""Fetch fresh adjusted prices for the extended universe into data_cache/prices_ext.

Never touches data_cache/prices (the mid-semester cache).
Usage: .venv/bin/python scripts/final_fetch_prices.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.final.config import EXT_END, EXT_START, EXT_UNIVERSE, use_extended_cache

use_extended_cache()

from nse_agents.data.prices import PRICE_CACHE, load_prices  # noqa: E402


def main() -> int:
    print("cache:", PRICE_CACHE)
    symbols = list(EXT_UNIVERSE) + ["^NSEI", "^INDIAVIX"]
    for sym in symbols:
        # end is exclusive at Yahoo; ask a few days past EXT_END
        frame = load_prices(sym, EXT_START, "2026-11-10")
        print(f"{sym:12s} {len(frame):5d} rows  {frame['date'].min().date()}..{frame['date'].max().date()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
