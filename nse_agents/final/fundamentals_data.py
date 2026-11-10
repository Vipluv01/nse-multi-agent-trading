"""Load cached screener payloads into a FundamentalsAgent for a given set of dates."""
from __future__ import annotations

import json

import pandas as pd

from ..agents.fundamentals import FundamentalsAgent, build_fundamentals_table, point_in_time_panel
from .screener import CACHE


def load_payloads(symbols) -> dict[str, dict]:
    out = {}
    for sym in symbols:
        path = CACHE / f"{sym.replace('&', '%26')}.json"
        if path.exists():
            out[sym] = json.loads(path.read_text())
    return out


def build_agent(symbols, dates) -> tuple[FundamentalsAgent, pd.DataFrame]:
    table = build_fundamentals_table(load_payloads(symbols))
    panel = point_in_time_panel(table, pd.DatetimeIndex(dates))
    return FundamentalsAgent(panel), table
