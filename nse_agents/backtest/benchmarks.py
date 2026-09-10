"""Benchmark return series for risk attribution, beyond the study's primary
Nifty 50 comparison.

Every series here is built with ``forward_return`` (open-to-open, attributed to the
decision date), never a naive close-to-close ``pct_change`` -- see the warning in
``metrics.py``'s benchmark-relative section for the real bug that convention mismatch
caused when this module was built (correlation 0.002 instead of the correct 0.95
between a Buy&Hold NSE portfolio and the Nifty).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..data.prices import forward_return, load_prices

# Yahoo's actual ticker for the Nifty Next 50 index -- confirmed via the chart
# endpoint's own quote metadata (longName/shortName both read "NIFTY NEXT 50"),
# not inferred from the symbol text, which is actively misleading (it looks like
# it should be a midcap index). Documented here so a future reader isn't tempted
# to "fix" what looks like a mismatched ticker.
NIFTY_NEXT_50_TICKER = "^NSMIDCP"

# Nifty Next 50 only has Yahoo history from 2020-01-01 -- about 2 years short of
# the main study's OOS window (2018-12-07 onward). Any comparison against it is
# therefore over a shorter window than the primary Nifty 50 comparison, and must
# say so rather than silently report a number that looks directly comparable.
NIFTY_NEXT_50_START = "2020-01-01"


def _forward_return_series(symbol: str, start: str, end: str) -> pd.DataFrame:
    frame = load_prices(symbol, start, end)
    frame["fwd_ret"] = forward_return(frame, horizon=SETTINGS.horizon_days)
    return frame[["date", "fwd_ret"]].dropna()


def equal_weight_universe_returns(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
) -> pd.DataFrame:
    """Daily equal-weight forward return across the study's universe -- a
    naive, no-selection, no-timing benchmark: what an investor gets from
    holding every name in the universe at equal weight, updated every day.
    This is a different question from Buy&Hold in the main study (which is
    also equal-weight-all-names via the trader's own weighting logic) only in
    that this function is the *raw index-style series* on its own, usable as
    a benchmark input to beta/Treynor/IR/capture-ratio functions rather than
    run through the cost-aware backtest engine.
    """
    frames = [_forward_return_series(symbol, start, end) for symbol in symbols]
    pooled = pd.concat(frames, ignore_index=True)
    return pooled.groupby("date", as_index=False)["fwd_ret"].mean()


def nifty_next_50_returns(start: str = SETTINGS.start, end: str = SETTINGS.end) -> pd.DataFrame:
    effective_start = max(pd.Timestamp(start), pd.Timestamp(NIFTY_NEXT_50_START))
    return _forward_return_series(NIFTY_NEXT_50_TICKER, str(effective_start.date()), end)


def nifty_50_returns(start: str = SETTINGS.start, end: str = SETTINGS.end) -> pd.DataFrame:
    from ..config import BENCHMARK

    return _forward_return_series(BENCHMARK, start, end)


def build_extended_benchmarks(
    start: str = SETTINGS.start, end: str = SETTINGS.end
) -> dict[str, pd.DataFrame]:
    """All three benchmarks, each as a (date, fwd_ret) frame ready to align
    against a strategy's own returns. Nifty Next 50's frame starts no earlier
    than 2020-01-01 regardless of ``start`` -- callers comparing against it
    must intersect dates explicitly, not assume the same window as the other two.
    """
    return {
        "Nifty50": nifty_50_returns(start, end),
        "EqualWeightUniverse": equal_weight_universe_returns(start=start, end=end),
        "NiftyNext50": nifty_next_50_returns(start, end),
    }


def align_benchmark_to_dates(benchmark: pd.DataFrame, dates: pd.DatetimeIndex) -> np.ndarray:
    """Reindex a (date, fwd_ret) benchmark frame onto ``dates``, filling any
    gap with 0.0 -- consistent with how the backtest engine treats a symbol
    with no return data for a given day (see ``run_backtest``'s
    ``np.nan_to_num`` on the return pivot).
    """
    series = benchmark.set_index("date")["fwd_ret"]
    return series.reindex(pd.DatetimeIndex(dates)).fillna(0.0).to_numpy()
