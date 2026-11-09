"""Fundamentals agent: disclosure lag, causality, abstention, parser."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_agents.agents.fundamentals import (
    FundamentalsAgent,
    availability_date,
    build_fundamentals_table,
    point_in_time_panel,
    statements_to_frame,
)
from nse_agents.final.screener import parse_section


def _payload(sym, profits, sales=None, equity=None):
    years = [f"Mar {2015 + i}" for i in range(len(profits))]
    sales = sales or [100.0 * (i + 1) for i in range(len(profits))]
    equity = equity or [500.0 + 50 * i for i in range(len(profits))]
    return {
        "symbol": sym,
        "profit_loss": {
            **{y: {"Sales": s, "Net Profit": p} for y, s, p in zip(years, sales, profits)},
            "TTM": {"Sales": 9e9, "Net Profit": 9e9},   # must be ignored
        },
        "balance_sheet": {y: {"Equity Capital": 10.0, "Reserves": e - 10.0} for y, e in zip(years, equity)},
    }


def test_availability_is_sebi_deadline_plus_one():
    assert availability_date("Mar 2019") == pd.Timestamp("2019-05-31")


def test_ttm_column_is_never_used():
    frame = statements_to_frame("A", _payload("A", [10, 20, 30]))
    assert "TTM" not in set(frame["fiscal"])
    assert frame["net_profit"].max() == 30


def test_growth_off_negative_base_is_nan():
    frame = statements_to_frame("A", _payload("A", [-10.0, 20.0, 30.0]))
    assert np.isnan(frame.loc[1, "profit_growth"])
    assert frame.loc[2, "profit_growth"] == pytest.approx(0.5)


def test_report_invisible_before_disclosure_date():
    table = build_fundamentals_table({"A": _payload("A", [10, 20, 40, 80]), "B": _payload("B", [10, 11, 12, 13])})
    dates = pd.DatetimeIndex(["2017-05-30", "2017-05-31"])
    panel = point_in_time_panel(table, dates).set_index(["symbol", "date"])
    # On 30 May 2017 the FY2017 report is not yet public: latest is FY2016.
    assert panel.loc[("A", pd.Timestamp("2017-05-30")), "fiscal"] == "Mar 2016"
    assert panel.loc[("A", pd.Timestamp("2017-05-31")), "fiscal"] == "Mar 2017"


def test_future_reports_do_not_change_past_panel():
    base = {"A": _payload("A", [10, 20, 40]), "B": _payload("B", [10, 11, 12])}
    longer = {"A": _payload("A", [10, 20, 40, 1.0]), "B": _payload("B", [10, 11, 12, 500.0])}
    dates = pd.date_range("2016-06-01", "2017-12-31", freq="W")
    p1 = point_in_time_panel(build_fundamentals_table(base), dates)
    p2 = point_in_time_panel(build_fundamentals_table(longer), dates)
    cols = ["symbol", "date", "fiscal", "z_profit_growth"]
    pd.testing.assert_frame_equal(p1[cols].reset_index(drop=True), p2[cols].reset_index(drop=True))


def test_agent_ranks_faster_grower_higher_and_abstains_early():
    payloads = {
        "FAST": _payload("FAST", [10, 20, 40, 80]),
        "MID": _payload("MID", [10, 13, 16, 20]),
        "SLOW": _payload("SLOW", [10, 10.5, 11, 11.5]),
    }
    dates = pd.DatetimeIndex(["2015-03-01", "2018-07-02"])
    agent = FundamentalsAgent(point_in_time_panel(build_fundamentals_table(payloads), dates))
    early = agent.opine("FAST", pd.Timestamp("2015-03-01"))
    assert early.abstained
    fast = agent.opine("FAST", pd.Timestamp("2018-07-02"))
    slow = agent.opine("SLOW", pd.Timestamp("2018-07-02"))
    assert not fast.abstained and fast.stance > slow.stance
    assert "Mar 2018" in fast.rationale
    assert 0 < fast.confidence <= 0.5


def test_parser_reads_screener_style_table():
    html = """<section id="profit-loss"><table>
      <tr><th></th><th>Mar 2024</th><th>Mar 2025</th></tr>
      <tr><td>Sales +</td><td>1,200</td><td>1,500</td></tr>
      <tr><td>Net Profit +</td><td>100</td><td>-</td></tr>
    </table></section>"""
    out = parse_section(html, "profit-loss")
    assert out["Mar 2024"]["Sales"] == 1200.0
    assert out["Mar 2025"]["Net Profit"] is None


def test_irregular_fiscal_periods_are_parsed_and_not_used_for_growth():
    payload = {
        "profit_loss": {
            "Dec 2022": {"Sales": 100.0, "Net Profit": 10.0},
            "Dec 2023": {"Sales": 110.0, "Net Profit": 11.0},
            "Mar 2025 15m": {"Sales": 150.0, "Net Profit": 15.0},
            "Mar 2026": {"Sales": 130.0, "Net Profit": 13.0},
        },
        "balance_sheet": {k: {"Equity Capital": 1.0, "Reserves": 99.0}
                          for k in ("Dec 2022", "Dec 2023", "Mar 2025 15m", "Mar 2026")},
    }
    frame = statements_to_frame("N", payload).set_index("fiscal")
    assert frame.loc["Dec 2023", "available_from"] == pd.Timestamp("2024-03-01")
    assert frame.loc["Dec 2023", "revenue_growth"] == pytest.approx(0.10)
    assert np.isnan(frame.loc["Mar 2025 15m", "revenue_growth"])   # 15m vs 12m: not growth
    assert np.isnan(frame.loc["Mar 2026", "revenue_growth"])       # 12m vs 15m: not growth
    # ROE on the 15-month period is annualised: 15 * 12/15 / 100
    assert frame.loc["Mar 2025 15m", "roe"] == pytest.approx(0.12)
