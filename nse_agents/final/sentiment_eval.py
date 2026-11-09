"""Shared sentiment evaluation for the final study: earnings split + model comparison.

Everything is computed on the same aligned, trading-day sentiment views the
mid-semester study used, so a number here is directly comparable to README's.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..config import SETTINGS
from ..data.news import align_to_trading_days
from ..data.prices import forward_return, load_prices

# Results/earnings headlines. Precision was checked by reading a random sample
# of matches (see thesis appendix); ~10% of the corpus matches.
EARNINGS_PATTERN = re.compile(
    r"\bQ[1-4]\b|\bQ[1-4]FY|\bquarter(?:ly)?\b|\b(?:results?|earnings)\b|\bnet profit\b|\bPAT\b"
    r"|\bprofit (?:rises|falls|jumps|drops|surges|slumps|declines|grows|up|down)\b",
    re.IGNORECASE,
)


def is_earnings(title: str) -> bool:
    return bool(EARNINGS_PATTERN.search(str(title)))


def returns_panel(symbols=SETTINGS.universe, start=SETTINGS.start, end=SETTINGS.end,
                  horizons=(1, 5)) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        prices = load_prices(symbol, start, end)
        out = prices[["date"]].copy()
        for h in horizons:
            out[f"fwd_{h}d"] = forward_return(prices, h)
        # same-day open-to-open magnitude, as a volatility (attention) proxy
        out["abs_fwd_1d"] = out["fwd_1d"].abs()
        out["symbol"] = symbol
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def daily_views(scored: pd.DataFrame, sessions: pd.Series) -> pd.DataFrame:
    """Headline scores -> one row per (symbol, session), with an earnings flag."""
    frame = scored.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["earnings"] = frame["title"].map(is_earnings)
    aligned = align_to_trading_days(frame, sessions)
    daily = (
        aligned.groupby(["symbol", "date"])
        .agg(sentiment=("score", "mean"), n_headlines=("score", "size"),
             earnings=("earnings", "max"), p_unknown=("p_unknown", "mean"))
        .reset_index()
    )
    # Causal debias: subtract the symbol's own trailing mean tone (shifted).
    daily = daily.sort_values(["symbol", "date"])
    base = daily.groupby("symbol")["sentiment"].transform(
        lambda s: s.rolling(60, min_periods=10).mean().shift(1))
    daily["sentiment_debiased"] = daily["sentiment"] - base.fillna(0.0)
    return daily.reset_index(drop=True)


def predictability(frame: pd.DataFrame, score_col: str = "sentiment", ret_col: str = "fwd_1d") -> dict:
    """Return-predictability statistics for one subset of sentiment views."""
    frame = frame.dropna(subset=[score_col, ret_col])
    n = len(frame)
    if n < 30:
        return {"n": n}
    ic, ic_p = sps.spearmanr(frame[score_col], frame[ret_col])
    slope, _, _, p, se = sps.linregress(frame[score_col], frame[ret_col])
    ranks = frame[score_col].rank(method="first")
    bucket = pd.qcut(ranks, 3, labels=["neg", "neu", "pos"])
    pos = frame.loc[bucket == "pos", ret_col]
    neg = frame.loc[bucket == "neg", ret_col]
    t, tp = sps.ttest_ind(pos, neg, equal_var=False)
    signal = np.sign(frame[score_col])
    hit = float((signal == np.sign(frame[ret_col]))[signal != 0].mean())
    return {
        "n": n,
        "ic_spearman": float(ic),
        "ic_p": float(ic_p),
        "slope_bps": float(slope * 1e4),
        "slope_p": float(p),
        "spread_pos_minus_neg_bps": float((pos.mean() - neg.mean()) * 1e4),
        "spread_t": float(t),
        "spread_p": float(tp),
        "hit_rate": hit,
    }
