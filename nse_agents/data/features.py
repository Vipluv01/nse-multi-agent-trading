"""Causal technical features and sequence windowing.

Every indicator here is computed from data at or before the bar it is indexed
on. There is no centred rolling window, no ``bfill``, and no full-sample
scaling: normalisation statistics are fitted per training fold in
``models.train`` rather than here, because a scaler fitted on the whole series
leaks the test period's mean and variance into the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rsi_14",
    "macd_hist",
    "bb_pos_20",
    "atr_14_norm",
    "vol_z_20",
    "realized_vol_20",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "stoch_k_14",
    "range_pct",
)


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the feature columns to one symbol's OHLCV frame."""
    out = frame.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_10d"] = close.pct_change(10)
    out["ret_20d"] = close.pct_change(20)

    out["rsi_14"] = _rsi(close, 14) / 100.0

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = (macd - signal) / close

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bb_pos_20"] = ((close - sma20) / (2.0 * std20)).replace([np.inf, -np.inf], np.nan)

    out["atr_14_norm"] = _atr(out, 14) / close

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["vol_z_20"] = ((volume - vol_mean) / vol_std).replace([np.inf, -np.inf], np.nan)

    out["realized_vol_20"] = out["ret_1d"].rolling(20).std() * np.sqrt(252)

    for w in (20, 50, 200):
        out[f"dist_sma_{w}"] = close / close.rolling(w).mean() - 1.0

    lo14 = low.rolling(14).min()
    hi14 = high.rolling(14).max()
    out["stoch_k_14"] = ((close - lo14) / (hi14 - lo14)).replace([np.inf, -np.inf], np.nan)

    out["range_pct"] = (high - low) / close

    return out


def make_sequences(
    frame: pd.DataFrame,
    label: pd.Series,
    lookback: int = 30,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Window the features into ``(N, lookback, F)`` sequences.

    Returns ``(X, y, raw_forward_return, decision_dates)``. The sequence ending
    at row ``t`` is paired with the label for a position opened at ``t+1``'s
    open, so ``decision_dates[i]`` is the close on which the call was made.
    """
    feats = frame[list(feature_columns)].to_numpy(dtype=np.float32)
    lab = label.to_numpy(dtype=np.float32)
    dates = pd.DatetimeIndex(frame["date"])

    valid = ~np.isnan(feats).any(axis=1) & ~np.isnan(lab)
    xs, ys, rets, ds = [], [], [], []
    for end in range(lookback - 1, len(frame)):
        window = slice(end - lookback + 1, end + 1)
        if not valid[window].all():
            continue
        xs.append(feats[window])
        rets.append(lab[end])
        ys.append(1.0 if lab[end] > 0 else 0.0)
        ds.append(dates[end])

    if not xs:
        empty = np.empty((0, lookback, len(feature_columns)), dtype=np.float32)
        return empty, np.empty(0, np.float32), np.empty(0, np.float32), pd.DatetimeIndex([])
    return (
        np.stack(xs),
        np.asarray(ys, dtype=np.float32),
        np.asarray(rets, dtype=np.float32),
        pd.DatetimeIndex(ds),
    )
