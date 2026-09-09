"""Assemble the pooled, cross-sectional sequence dataset.

Rows from all symbols are interleaved and sorted by decision date, so that any
chronological slice of the array is also a chronological slice of the market.
The trainer's validation tail and the walk-forward splitter both depend on
this ordering being real rather than incidental.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import SETTINGS
from .features import FEATURE_COLUMNS, build_features, make_sequences
from .prices import forward_return, load_prices


@dataclass
class PanelDataset:
    x: np.ndarray
    y: np.ndarray
    fwd_returns: np.ndarray
    dates: pd.DatetimeIndex
    symbols: np.ndarray
    feature_columns: tuple[str, ...]
    lookback: int

    def __len__(self) -> int:
        return len(self.x)

    def summary(self) -> str:
        return (
            f"{len(self.x):,} sequences  {self.x.shape[1]}x{self.x.shape[2]}  "
            f"{len(np.unique(self.symbols))} symbols  "
            f"{self.dates.min().date()}..{self.dates.max().date()}  "
            f"up-rate {self.y.mean():.4f}"
        )


def build_panel_dataset(
    symbols: tuple[str, ...] = SETTINGS.universe,
    start: str = SETTINGS.start,
    end: str = SETTINGS.end,
    lookback: int = 30,
    horizon: int = SETTINGS.horizon_days,
) -> PanelDataset:
    xs, ys, rs, ds, ss = [], [], [], [], []
    for symbol in symbols:
        frame = build_features(load_prices(symbol, start, end))
        frame["fwd"] = forward_return(frame, horizon)
        x, y, r, dates = make_sequences(frame, frame["fwd"], lookback, FEATURE_COLUMNS)
        if len(x) == 0:
            continue
        xs.append(x)
        ys.append(y)
        rs.append(r)
        ds.append(dates)
        ss.append(np.full(len(x), symbol, dtype=object))

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    r = np.concatenate(rs)
    dates = pd.DatetimeIndex(np.concatenate([d.to_numpy() for d in ds]))
    syms = np.concatenate(ss)

    # Stable sort by date keeps symbols in a deterministic order within a day.
    order = np.argsort(dates.to_numpy(), kind="stable")
    return PanelDataset(
        x=x[order],
        y=y[order],
        fwd_returns=r[order],
        dates=dates[order],
        symbols=syms[order],
        feature_columns=FEATURE_COLUMNS,
        lookback=lookback,
    )
