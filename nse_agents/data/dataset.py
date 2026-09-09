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
    label: str = "absolute",
) -> PanelDataset:
    """Build the pooled sequence dataset.

    ``label`` selects what the model is asked to predict:

    * ``absolute`` -- did this stock rise? This is dominated by the market
      factor: on a day the Nifty gains 1%, almost every name is up, and no
      per-name technical feature can predict that common component.
    * ``cross_sectional`` -- did this stock beat the median stock *that day*?
      The common factor cancels by construction, so the label isolates the only
      thing a per-name feature could plausibly know. Roughly 50% positive by
      definition, which also removes the mild class imbalance.
    """
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

    if label == "cross_sectional":
        # Demean the forward return across the names trading that day, then
        # re-derive the binary label from the residual. Uses only same-day
        # outcomes, so it changes what is predicted, not when it is known.
        frame = pd.DataFrame({"date": dates, "r": r})
        median = frame.groupby("date")["r"].transform("median")
        y = (r > median.to_numpy()).astype(np.float32)
    elif label != "absolute":
        raise ValueError(f"unknown label {label!r}; expected absolute|cross_sectional")

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
