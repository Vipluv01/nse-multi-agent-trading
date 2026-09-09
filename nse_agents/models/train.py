"""Purged walk-forward training for the technical agent.

The design decisions that matter here are all about *not* leaking:

* **Rolling, not expanding, windows.** A fixed 3-year lookback assumes the
  market's dynamics decay; an expanding window would let a 2016 regime vote on
  a 2026 test fold with equal weight.
* **Purge and embargo.** The label at decision date ``t`` resolves at ``t+1+h``,
  so training rows within ``embargo_days`` of the test fold are dropped
  outright rather than merely ordered before it.
* **Per-fold scaling.** ``StandardScaler`` statistics come from that fold's
  training rows only. Fitting on the full panel is the single most common
  silent leak in published trading models.
* **Chronological validation split.** Early stopping uses the last 15% of the
  training window, not a random 15%. A random split lets the model early-stop
  on days interleaved with its own training days.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import SETTINGS, WalkForward
from .attn_lstm import PLSTMTAL


@dataclass
class Fold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_idx: np.ndarray = field(repr=False)
    test_idx: np.ndarray = field(repr=False)


def walk_forward_folds(
    dates: pd.DatetimeIndex, config: WalkForward = SETTINGS.walk_forward
) -> list[Fold]:
    """Rolling purged folds over a pooled, multi-symbol date index."""
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    if len(unique) == 0:
        return []
    train_span = pd.DateOffset(days=int(round(config.train_years * 365.25)))
    test_span = pd.DateOffset(months=config.test_months)
    embargo = pd.Timedelta(days=config.embargo_days)

    folds: list[Fold] = []
    cursor = unique[0] + train_span
    while True:
        train_start = cursor - train_span
        train_end = cursor - embargo  # purge: drop the embargo band entirely
        test_start = cursor
        test_end = cursor + test_span
        if test_start >= unique[-1]:
            break
        train_idx = np.flatnonzero((dates >= train_start) & (dates < train_end))
        test_idx = np.flatnonzero((dates >= test_start) & (dates < test_end))
        if len(train_idx) > 200 and len(test_idx) > 20:
            folds.append(
                Fold(
                    index=len(folds),
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=min(test_end, unique[-1]),
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )
        cursor = cursor + test_span
    return folds


def _standardise(
    train: np.ndarray, *others: np.ndarray
) -> tuple[np.ndarray, ...]:
    """Z-score using training-fold statistics only."""
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0, keepdims=True)
    std = flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    def apply(arr: np.ndarray) -> np.ndarray:
        if len(arr) == 0:
            return arr
        return np.clip((arr - mean) / std, -8.0, 8.0).astype(np.float32)

    return (apply(train), *(apply(o) for o in others))


@dataclass
class TrainConfig:
    hidden_size: int = 64
    attn_size: int = 32
    dropout: float = 0.2
    peephole: bool = True
    attention: bool = True
    epochs: int = 25
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    val_fraction: float = 0.15
    device: str = "cpu"

    def label(self) -> str:
        core = "PLSTM" if self.peephole else "LSTM"
        return f"{core}{'-TAL' if self.attention else ''}"


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    config: TrainConfig,
    seed: int = SETTINGS.seed,
) -> tuple[np.ndarray, dict]:
    """Fit on one fold, return test-set probabilities and a training summary."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Chronological validation tail -- the rows are already in date order.
    n_val = max(1, int(len(x_train) * config.val_fraction))
    x_fit, y_fit = x_train[:-n_val], y_train[:-n_val]
    x_val, y_val = x_train[-n_val:], y_train[-n_val:]

    x_fit, x_val, x_test_s = _standardise(x_fit, x_val, x_test)

    device = torch.device(config.device)
    model = PLSTMTAL(
        input_size=x_fit.shape[-1],
        hidden_size=config.hidden_size,
        attn_size=config.attn_size,
        dropout=config.dropout,
        peephole=config.peephole,
        attention=config.attention,
    ).to(device)

    # Positive-class weighting: the up/down split is near 50/50 but not exactly,
    # and an unweighted model drifts toward the majority side.
    pos = float(y_fit.sum())
    neg = float(len(y_fit) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    xt = torch.from_numpy(x_fit).to(device)
    yt = torch.from_numpy(y_fit).to(device)
    xv = torch.from_numpy(x_val).to(device)
    yv = torch.from_numpy(y_val).to(device)

    best_loss, best_state, bad_epochs, best_epoch = float("inf"), None, 0, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator).to(device)
        for start in range(0, len(xt), config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad()
            loss = criterion(model(xt[batch]), yt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv))
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        xs = torch.from_numpy(x_test_s).to(device)
        probs = torch.sigmoid(model(xs)).cpu().numpy()
        attention = (
            model.last_attention.cpu().numpy() if model.last_attention is not None else None
        )

    summary = {
        "val_loss": best_loss,
        "best_epoch": best_epoch,
        "n_train": len(x_fit),
        "n_val": len(x_val),
        "n_test": len(x_test),
        "architecture": model.describe(),
    }
    return probs, {**summary, "attention": attention}


def run_walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: np.ndarray,
    fwd_returns: np.ndarray,
    config: TrainConfig,
    wf: WalkForward = SETTINGS.walk_forward,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Train one model per fold; return the concatenated out-of-sample frame."""
    folds = walk_forward_folds(dates, wf)
    rows, summaries = [], []
    for fold in folds:
        probs, summary = train_one_fold(
            x[fold.train_idx], y[fold.train_idx], x[fold.test_idx], config
        )
        attention = summary.pop("attention")
        rows.append(
            pd.DataFrame(
                {
                    "date": dates[fold.test_idx],
                    "symbol": symbols[fold.test_idx],
                    "prob_up": probs,
                    "y_true": y[fold.test_idx],
                    "fwd_ret": fwd_returns[fold.test_idx],
                    "fold": fold.index,
                    "attn_recent5": (
                        attention[:, -5:].sum(axis=1) if attention is not None else np.nan
                    ),
                }
            )
        )
        summary.update(
            {
                "fold": fold.index,
                "train_start": fold.train_start.date(),
                "train_end": fold.train_end.date(),
                "test_start": fold.test_start.date(),
                "test_end": fold.test_end.date(),
                "test_accuracy": float(((probs > 0.5) == (y[fold.test_idx] > 0.5)).mean()),
            }
        )
        summaries.append(summary)
        if verbose:
            print(
                f"  fold {fold.index:2d} {fold.test_start.date()}..{fold.test_end.date()}  "
                f"n_train={summary['n_train']:5d}  acc={summary['test_accuracy']:.4f}  "
                f"val_loss={summary['val_loss']:.4f}  ep={summary['best_epoch']}",
                flush=True,
            )
    if not rows:
        raise RuntimeError("no walk-forward folds produced -- check the date range")
    return pd.concat(rows, ignore_index=True), summaries
