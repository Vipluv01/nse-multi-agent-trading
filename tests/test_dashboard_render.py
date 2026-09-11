"""Programmatic dashboard verification via Streamlit's AppTest framework
(``streamlit.testing.v1.AppTest``) -- runs the real ``scripts/dashboard.py``
script (not a mock of it) and asserts zero exceptions across every tab, in
both the empty-account and populated-account states. This is the same
runtime-check bar KNOWN_ISSUES.md already documents for this dashboard
("HTTP 200, zero tracebacks"), now automated rather than done by hand with
curl before every commit.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("streamlit", reason="streamlit is an optional dependency ([dashboard] extra)")
from streamlit.testing.v1 import AppTest

from nse_agents.live.broker import Fill
from nse_agents.live.state_store import PaperTradingStore

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_SCRIPT = str(ROOT / "scripts" / "dashboard.py")


def _populated_db(tmp_path) -> Path:
    path = tmp_path / "state.sqlite"
    store = PaperTradingStore(path, initial_capital=1_000_000.0)
    store.apply_fill(
        Fill(symbol="TCS", side="buy", quantity=10, price=3500.0, cost=15.0, timestamp=pd.Timestamp.now()),
        "2026-09-01",
    )
    store.apply_fill(
        Fill(symbol="RELIANCE", side="buy", quantity=5, price=2900.0, cost=12.0, timestamp=pd.Timestamp.now()),
        "2026-09-02",
    )
    store.record_equity("2026-09-02", {"TCS": 3550.0, "RELIANCE": 2920.0})
    return path


def test_dashboard_loads_with_no_paper_trading_state_and_raises_nothing(tmp_path):
    """No account created yet -- the dashboard must show its own warning, not
    crash on a missing file."""
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(tmp_path / "does_not_exist.sqlite")).run()
    assert not at.exception
    assert len(at.warning) >= 1


def test_dashboard_loads_with_a_populated_account_and_raises_nothing(tmp_path):
    db_path = _populated_db(tmp_path)
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    assert not at.exception


def test_all_four_tabs_are_present(tmp_path):
    db_path = _populated_db(tmp_path)
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    assert not at.exception
    assert len(at.tabs) == 4


def test_positions_and_trades_tab_renders_real_dataframes(tmp_path):
    db_path = _populated_db(tmp_path)
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    assert not at.exception
    positions_tab = at.tabs[1]
    assert len(positions_tab.dataframe) == 2  # positions table + trade log


def test_backtest_ablation_tab_shows_a_dataframe_when_the_cache_exists(tmp_path):
    """The 4th tab must render the main study's cached summary -- or, absent
    it, an info message -- never an exception either way."""
    db_path = _populated_db(tmp_path)
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    assert not at.exception
    ablation_tab = at.tabs[3]
    has_data = len(ablation_tab.dataframe) >= 1
    has_fallback_message = len(ablation_tab.info) >= 1
    assert has_data or has_fallback_message


def test_macro_tab_symbol_selector_can_be_changed_without_raising(tmp_path):
    """The one real widget interaction on this dashboard beyond the sidebar
    path input -- switching the watched symbol must re-run cleanly."""
    db_path = _populated_db(tmp_path)
    at = AppTest.from_file(DASHBOARD_SCRIPT, default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(db_path)).run()
    assert not at.exception
    if len(at.selectbox):
        options = at.selectbox[0].options
        other = next((o for o in options if o != at.selectbox[0].value), options[0])
        at.selectbox[0].set_value(other).run()
        assert not at.exception
