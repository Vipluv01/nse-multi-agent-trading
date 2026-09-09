"""Build today's decision inputs -- the seam between the walk-forward study and a
live run.

Nothing here is evaluated. A live snapshot has no known outcome yet, so there is no
accuracy or Sharpe to report for it; every claim about how well this pipeline predicts
comes from the walk-forward study elsewhere in this repository, not from this module.
This module's only job is to build, for one decision date (normally today), the exact
same shape of inputs each agent already consumes when replaying history -- so the same,
already-tested agent code runs unmodified in both modes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

from ..agents.regime import build_regime_table
from ..config import SETTINGS
from ..data.dataset import build_panel_dataset  # noqa: F401  (re-exported for callers)
from ..data.features import FEATURE_COLUMNS, build_features
from ..data.news import COMPANY_NAMES, GoogleNewsRSS, align_to_trading_days, is_noise
from ..data.prices import load_prices
from ..models import checkpoint as ckpt
from ..agents.sentiment import aggregate_daily, score_headlines


@dataclass
class LiveSnapshot:
    as_of: pd.Timestamp
    technical_oos: pd.DataFrame     # (date, symbol, prob_up, attn_recent5) -- today only
    regime_table: pd.DataFrame      # full causal regime history through as_of
    sentiment_daily: pd.DataFrame   # (symbol, date, sentiment, n_headlines, ...) -- recent days
    checkpoint_meta: dict


def _latest_common_date(symbols: tuple[str, ...]) -> pd.Timestamp:
    dates = [load_prices(s, SETTINGS.start, "2100-01-01", refresh=True)["date"].max() for s in symbols]
    return min(dates)


def technical_predictions_today(
    model_path,
    symbols: tuple[str, ...] = SETTINGS.universe,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.Timestamp, dict]:
    """Run the production checkpoint on the most recent available feature window."""
    model, meta = ckpt.load(model_path, device=device)
    as_of = _latest_common_date(symbols)

    rows = []
    for symbol in symbols:
        frame = build_features(load_prices(symbol, SETTINGS.start, "2100-01-01"))
        frame = frame.loc[frame["date"] <= as_of]
        if len(frame) < meta.lookback:
            continue
        window = frame[list(meta.feature_columns)].to_numpy(dtype="float32")[-meta.lookback:]
        import numpy as np

        if np.isnan(window).any():
            continue
        x = (window - meta.scaler_mean) / meta.scaler_std
        x = np.clip(x, -8.0, 8.0).astype("float32")
        with torch.no_grad():
            logit = model(torch.from_numpy(x).unsqueeze(0).to(device))
            prob = float(torch.sigmoid(logit)[0])
        attention = (
            float(model.last_attention[0, -5:].sum()) if model.last_attention is not None else None
        )
        rows.append({"date": as_of, "symbol": symbol, "prob_up": prob, "attn_recent5": attention})

    predictions = pd.DataFrame(rows)
    return predictions, as_of, {
        "trained_through": meta.trained_through,
        "trained_at": meta.trained_at,
        "architecture": meta.config.label(),
    }


def sentiment_today(
    symbols: tuple[str, ...],
    as_of: pd.Timestamp,
    backend,
    lookback_days: int = 5,
    batch_size: int = 16,
) -> pd.DataFrame:
    """Fetch and score the last few days of headlines, for live use only.

    A short, recent window -- unlike the study's full 2016-2026 corpus -- because a
    live run only needs today's actionable news, not a historical backtest corpus.
    """
    start = (as_of - pd.Timedelta(days=lookback_days + 4)).date().isoformat()
    end = (as_of + pd.Timedelta(days=1)).date().isoformat()
    provider = GoogleNewsRSS(pause=0.6)

    items = []
    for symbol in symbols:
        items.extend(provider.fetch(symbol, start, end))
    items = [i for i in items if not is_noise(i.title)]
    if not items:
        return pd.DataFrame(columns=["symbol", "date", "sentiment", "n_headlines", "dispersion", "unknown_share"])

    scored = score_headlines(backend, items, COMPANY_NAMES, batch_size=batch_size, progress=False)
    sessions = load_prices(symbols[0], SETTINGS.start, "2100-01-01")["date"]
    aligned = align_to_trading_days(scored, sessions)
    return aggregate_daily(aligned)


def build_live_snapshot(
    model_path,
    symbols: tuple[str, ...] = SETTINGS.universe,
    backend=None,
    device: str = "cpu",
) -> LiveSnapshot:
    predictions, as_of, meta = technical_predictions_today(model_path, symbols, device)
    regime_table = build_regime_table(symbols=symbols, start=SETTINGS.start, end="2100-01-01")
    sentiment = (
        sentiment_today(symbols, as_of, backend) if backend is not None
        else pd.DataFrame(columns=["symbol", "date", "sentiment", "n_headlines", "dispersion", "unknown_share"])
    )
    return LiveSnapshot(
        as_of=as_of, technical_oos=predictions, regime_table=regime_table,
        sentiment_daily=sentiment, checkpoint_meta=meta,
    )
