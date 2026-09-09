"""Classical technical baselines, emitted in the model's own signal format.

Every baseline produces a ``(date, symbol, prob_up, fwd_ret)`` frame and is run
through the identical ``run_backtest`` path as the learned models. That is the
only way the comparison is honest: a baseline evaluated with a different
execution assumption or a different cost model is not a baseline, it is a
strawman.

The baselines are the ones the TradingAgents paper benchmarks against
(Buy-and-Hold, MACD, KDJ+RSI, mean reversion), plus a random-signal control
that shares the learned model's turnover -- the control that reveals how much
of a result is the signal and how much is just the trading pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import SETTINGS
from ..data.features import build_features
from ..data.prices import forward_return, load_prices


def _panel(symbols, start, end) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        frame = build_features(load_prices(symbol, start, end))
        frame["fwd_ret"] = forward_return(frame, SETTINGS.horizon_days)
        frame["symbol"] = symbol
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_baseline_signals(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
) -> dict[str, pd.DataFrame]:
    """All classical baselines, keyed by name."""
    panel = _panel(symbols, start, end)
    out: dict[str, pd.DataFrame] = {}

    def emit(name: str, signal: pd.Series) -> None:
        frame = panel[["date", "symbol", "fwd_ret"]].copy()
        frame["prob_up"] = signal.to_numpy(dtype=float)
        out[name] = frame.dropna(subset=["fwd_ret"]).reset_index(drop=True)

    # Buy-and-hold: always long every name. prob_up = 1 for all rows.
    emit("Buy&Hold", pd.Series(np.ones(len(panel)), index=panel.index))

    # MACD: long while the histogram is positive (12/26/9, already a feature).
    emit("MACD", (panel["macd_hist"] > 0).astype(float))

    # RSI(14): long below 30 (oversold), flat above 70, hold otherwise. The
    # state machine is expressed per symbol so a position persists correctly.
    rsi_signal = []
    for _, group in panel.groupby("symbol", sort=False):
        rsi = group["rsi_14"].to_numpy() * 100.0
        state, series = 0.0, []
        for value in rsi:
            if np.isnan(value):
                series.append(0.0)
                continue
            if value < 30:
                state = 1.0
            elif value > 70:
                state = 0.0
            series.append(state)
        rsi_signal.append(pd.Series(series, index=group.index))
    emit("RSI(14)", pd.concat(rsi_signal).sort_index())

    # KDJ + RSI, as benchmarked in TradingAgents: long when stochastic %K is
    # oversold and RSI confirms.
    emit(
        "KDJ+RSI",
        ((panel["stoch_k_14"] < 0.25) & (panel["rsi_14"] < 0.45)).astype(float),
    )

    # Zero mean reversion: long after a down day, expecting the bounce.
    emit("MeanReversion", (panel["ret_1d"] < 0).astype(float))

    return out


def random_control(
    reference: pd.DataFrame,
    seed: int = SETTINGS.seed,
    threshold: float = 0.5,
    score_column: str = "prob_up",
) -> pd.DataFrame:
    """Random stock picks that match the reference strategy day for day.

    Answers the question a Sharpe alone cannot: would a coin flip that traded
    this often, in this market, over these dates, have done as well? For a
    long-only strategy in a rising market, the answer is very often yes.

    The match is on the **per-day count of selected names**, not on an average
    exposure rate. Independent per-name draws at a target rate do not
    reproduce the reference's exposure: with ten names and a 20% per-name cap,
    any day selecting five or more names is fully invested, so independent
    draws sit near 100% exposure regardless of the rate asked for. Matching the
    daily count instead holds exposure *and* turnover close to the reference
    and randomises only the one thing under test -- which stocks were chosen.
    """
    rng = np.random.default_rng(seed)
    frame = reference[["date", "symbol", "fwd_ret"]].copy()
    selected = (reference[score_column] > threshold).to_numpy()

    picks = np.zeros(len(frame), dtype=float)
    for _, positions in frame.groupby("date", sort=False).indices.items():
        k = int(selected[positions].sum())
        if k <= 0:
            continue
        chosen = rng.choice(positions, size=min(k, len(positions)), replace=False)
        picks[chosen] = 1.0
    frame["prob_up"] = picks
    return frame
