"""Fetch annual fundamentals (screener.in) for the extended universe. Cached; resumable.
Usage: .venv/bin/python scripts/final_fetch_fundamentals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.final.config import EXT_UNIVERSE
from nse_agents.final.screener import fetch_company


def main() -> int:
    failed = []
    for sym in EXT_UNIVERSE:
        try:
            data = fetch_company(sym)
            years = [k for k in data["profit_loss"] if k.startswith("Mar")]
            top = "Sales" if any("Sales" in v for v in data["profit_loss"].values()) else "Revenue"
            print(f"{sym:12s} {len(years):2d} FYs  {years[0] if years else '-'}..{years[-1] if years else '-'}  top line: {top}")
        except Exception as exc:
            failed.append(sym)
            print(f"{sym:12s} FAILED: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
