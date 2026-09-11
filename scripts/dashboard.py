"""Interactive paper-trading dashboard. Launch with:

  .venv/bin/streamlit run scripts/dashboard.py

**Demonstration only, not a trading recommendation** -- see README.md. Every number
this dashboard shows comes from the persistent paper-trading account
(``nse_agents/live/state_store.py``), which has no relationship to the walk-forward
study's own evaluated results; nothing here is a new evaluation of the trading system.

All business logic lives in ``nse_agents/live/dashboard_data.py`` (no streamlit import
there), so it is unit-testable without a running Streamlit session -- this file is
UI wiring only: read the data, hand it to a chart or a table, done.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from nse_agents.config import RESULTS, SETTINGS
from nse_agents.live.dashboard_data import (
    backtest_ablation_table,
    compute_overview_stats,
    load_account_state,
    macro_regime_indicator,
    positions_table,
    sentiment_distribution,
    trades_table,
)
from nse_agents.live.state_store import DEFAULT_DB_PATH

st.set_page_config(page_title="NSE Paper Trading Dashboard", layout="wide")


@st.cache_data(ttl=300)
def _latest_prices(symbols: tuple[str, ...]) -> dict[str, float]:
    from nse_agents.data.prices import load_prices

    prices = {}
    for symbol in symbols:
        try:
            frame = load_prices(symbol, SETTINGS.start, "2100-01-01")
            if len(frame):
                prices[symbol] = float(frame["close"].iloc[-1])
        except Exception:
            continue  # a single bad symbol must not blank the whole dashboard
    return prices


@st.cache_data(ttl=1800)
def _load_regime_table(symbols: tuple[str, ...]) -> pd.DataFrame:
    from nse_agents.agents.regime import build_regime_table

    return build_regime_table(symbols=symbols, start=SETTINGS.start, end="2100-01-01")


@st.cache_data(ttl=1800)
def _load_sentiment_daily() -> pd.DataFrame:
    path = RESULTS / "sentiment" / "daily_local.csv"
    if not path.exists():
        return pd.DataFrame(columns=["symbol", "date", "sentiment"])
    return pd.read_csv(path, parse_dates=["date"])


def main() -> None:
    st.title("NSE Paper Trading Dashboard")
    st.caption(
        "**Demonstration only.** No configuration in this study has a demonstrated "
        "market-beating edge net of real costs — see README.md. Fills are simulated "
        "through MockBroker, not a real broker."
    )

    db_path = st.sidebar.text_input("Paper-trading DB path", str(DEFAULT_DB_PATH))
    if not Path(db_path).exists():
        st.warning(
            f"No paper-trading state at `{db_path}` yet. Run "
            "`scripts/run_live_signal.py --persist` at least once, then reload."
        )
        return

    store = load_account_state(db_path)
    symbols = SETTINGS.universe
    prices = _latest_prices(symbols)

    tab_overview, tab_positions, tab_macro, tab_ablation = st.tabs(
        ["Overview", "Positions & Trades", "Sentiment & Macro", "Backtest Ablation"]
    )

    with tab_overview:
        stats = compute_overview_stats(store, prices)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total equity", f"Rs {stats.total_equity:,.0f}", f"{stats.total_return:+.2%}")
        col2.metric("Max drawdown", f"{stats.max_drawdown:.2%}")
        col3.metric(
            "Sharpe (net)",
            f"{stats.sharpe:+.3f}" if stats.sharpe is not None else "withheld",
            help=f"Withheld below 60 tracked days ({stats.days_tracked} so far) — "
                 "an annualised Sharpe from a handful of days is not a conservative "
                 "estimate, it's a meaningless one.",
        )
        col4.metric(
            "Buy&Hold Sharpe (same window)",
            f"{stats.buyhold_sharpe:+.3f}" if stats.buyhold_sharpe is not None else "withheld",
        )

        history = store.equity_history()
        if len(history) >= 2:
            chart_data = history[["date", "total_equity"]].set_index("date")
            st.line_chart(chart_data, height=320)
        else:
            st.info("Not enough tracked days yet for an equity curve.")

    with tab_positions:
        st.subheader("Open positions")
        pos_table = positions_table(store, prices)
        if len(pos_table):
            st.dataframe(pos_table, width="stretch")
        else:
            st.info("No open positions.")

        st.subheader("Trade log")
        trade_log = trades_table(store)
        if len(trade_log):
            st.dataframe(trade_log, width="stretch")
        else:
            st.info("No trades recorded.")

    with tab_macro:
        st.subheader("India VIX / Nifty regime")
        regime_table = _load_regime_table(symbols)
        watch_symbol = st.selectbox("Symbol", symbols, index=0)
        indicator = macro_regime_indicator(regime_table, watch_symbol)
        c1, c2 = st.columns(2)
        c1.metric("India VIX", f"{indicator['vix']:.2f}" if indicator["vix"] is not None else "n/a")
        c2.metric(
            "Nifty 20-day momentum",
            f"{indicator['nifty_mom_20d']:+.2%}" if indicator["nifty_mom_20d"] is not None else "n/a",
        )

        st.subheader("Sentiment score distribution")
        sentiment_daily = _load_sentiment_daily()
        if len(sentiment_daily):
            dist = sentiment_distribution(sentiment_daily)
            st.bar_chart(dist.set_index("bucket"), height=280)
        else:
            st.info(
                "No sentiment data found at results/sentiment/daily_local.csv — run "
                "scripts/score_sentiment.py and scripts/aggregate_sentiment.py first."
            )

    with tab_ablation:
        st.subheader("Main walk-forward study: strategy ablation")
        st.caption(
            "This is the study's own already-published result (README.md) — not "
            "computed from the paper-trading account above, and not a new evaluation. "
            "Shown here for one unified view."
        )
        ablation = backtest_ablation_table(RESULTS)
        if len(ablation):
            st.dataframe(ablation, width="stretch")
            forest_path = RESULTS / "figures" / "sharpe_forest.png"
            if forest_path.exists():
                st.image(str(forest_path), caption="Net Sharpe ratio, 95% bootstrap CI")
        else:
            st.info(
                "No cached ablation summary found at results/agents/summary.csv — run "
                "scripts/run_agents.py first."
            )


if __name__ == "__main__":
    main()
