"""Market-structure analyst.

**This is deliberately not a fundamental agent, and the distinction matters.**
A genuine fundamental analyst needs point-in-time fundamentals: the P/E, ROE and
margins *as they were reported and known* on the decision date. What is freely
available for NSE names is the current snapshot. Substituting today's P/E into a
2019 decision is not an approximation, it is lookahead of the worst kind -- the
ratio embeds every earnings surprise between then and now, so a "value" signal
built from it will appear to predict the future because it literally contains
it.

Rather than fake that, this agent reasons about what *is* causally available:
trend regime, relative strength against the Nifty, position within the 52-week
range, and volatility state. It is named for what it does. Adding a real
fundamental agent is scoped in the README as the main data-dependent extension.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import BENCHMARK, SETTINGS
from ..data.prices import load_prices
from .base import Opinion


def build_regime_table(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
) -> pd.DataFrame:
    """Causal regime features per (symbol, date)."""
    index = load_prices(BENCHMARK, start, end)[["date", "close"]].rename(
        columns={"close": "nifty"}
    )
    frames = []
    for symbol in symbols:
        frame = load_prices(symbol, start, end)[["date", "close", "high", "low"]].copy()
        frame = frame.merge(index, on="date", how="left")
        frame["nifty"] = frame["nifty"].ffill()

        close = frame["close"]
        frame["trend"] = np.sign(close / close.rolling(200).mean() - 1.0)
        frame["trend_strength"] = (close / close.rolling(200).mean() - 1.0).clip(-0.5, 0.5)
        # Relative strength: 60-day stock return minus 60-day index return.
        frame["rel_strength_60d"] = close.pct_change(60) - frame["nifty"].pct_change(60)
        high52 = frame["high"].rolling(252).max()
        low52 = frame["low"].rolling(252).min()
        frame["pos_52w"] = ((close - low52) / (high52 - low52)).clip(0.0, 1.0)
        realized = close.pct_change().rolling(20).std() * np.sqrt(252)
        frame["vol_20"] = realized
        frame["vol_percentile"] = realized.rolling(252).rank(pct=True)
        frame["symbol"] = symbol
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


class RegimeAgent:
    name = "regime"

    def __init__(self, table: pd.DataFrame):
        frame = table.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        self._lookup = frame.set_index(["symbol", "date"]).sort_index()

    def opine(self, symbol: str, date: pd.Timestamp) -> Opinion:
        try:
            row = self._lookup.loc[(symbol, pd.Timestamp(date))]
        except KeyError:
            return Opinion.abstain(self.name, "no regime data for this date")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        trend = row["trend"]
        rel = row["rel_strength_60d"]
        pos = row["pos_52w"]
        vol_pct = row["vol_percentile"]
        if any(pd.isna(v) for v in (trend, rel, pos)):
            return Opinion.abstain(self.name, "insufficient history for a regime read")

        # Three equally-weighted votes: long-term trend, relative strength,
        # and range position. Deliberately simple; a tuned weighting here would
        # be one more parameter fitted to the same test period.
        votes = [
            float(trend),
            float(np.clip(rel * 5.0, -1.0, 1.0)),
            float(np.clip((pos - 0.5) * 2.0, -1.0, 1.0)),
        ]
        stance = float(np.clip(np.mean(votes), -1.0, 1.0))

        # High-volatility regimes are where these medium-term signals are least
        # reliable, so confidence is cut rather than the stance being flipped.
        vol_penalty = 1.0 if pd.isna(vol_pct) else float(1.0 - 0.5 * vol_pct)
        agreement = 1.0 - float(np.std(votes))
        confidence = float(np.clip(max(agreement, 0.0) * vol_penalty, 0.0, 1.0))

        return Opinion(
            agent=self.name,
            stance=stance,
            confidence=confidence,
            rationale=(
                f"{symbol} is {'above' if trend > 0 else 'below'} its 200-day average, "
                f"{rel:+.1%} vs the Nifty over 60 sessions, and sits at the "
                f"{pos:.0%} mark of its 52-week range"
                + ("" if pd.isna(vol_pct) else f" with volatility in the {vol_pct:.0%} percentile")
                + "."
            ),
            evidence={
                "trend": float(trend),
                "rel_strength_60d": float(rel),
                "pos_52w": float(pos),
                "vol_percentile": None if pd.isna(vol_pct) else float(vol_pct),
            },
        )
