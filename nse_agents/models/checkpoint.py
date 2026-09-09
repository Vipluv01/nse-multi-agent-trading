"""Persist and reload a trained technical model for live use.

Nothing in the walk-forward study needs this: every OOS prediction reported anywhere
in this repository comes from a model trained and discarded per fold, exactly as
required for an honest evaluation. This module exists for a different, later purpose --
running the trained pipeline live, once, today -- and it is built so the two can never
be confused: a checkpoint here carries no accuracy claim, because there is no held-out
data left to have evaluated it against (today's outcome is not yet known).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.features import FEATURE_COLUMNS
from .attn_lstm import PLSTMTAL
from .train import TrainConfig


@dataclass
class Checkpoint:
    config: TrainConfig
    feature_columns: tuple[str, ...]
    lookback: int
    scaler_mean: np.ndarray
    scaler_std: np.ndarray
    trained_through: str          # ISO date of the last training row used
    trained_at: str               # ISO timestamp this checkpoint was produced


def save(
    path: Path,
    model: PLSTMTAL,
    config: TrainConfig,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    trained_through: str,
    lookback: int = 30,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
) -> None:
    """``feature_columns`` must match what the model was actually built with --
    it is recorded as given, not assumed from the module-level default. A
    caller that trains on a custom feature subset and lets this default silently
    would produce a checkpoint that claims the wrong input shape; a test
    training a small toy model caught exactly that class of bug here.
    """
    if len(feature_columns) != scaler_mean.shape[-1]:
        raise ValueError(
            f"feature_columns has {len(feature_columns)} entries but scaler_mean "
            f"has {scaler_mean.shape[-1]} -- they must describe the same features"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "feature_columns": list(feature_columns),
            "lookback": lookback,
            "scaler_mean": scaler_mean.tolist(),
            "scaler_std": scaler_std.tolist(),
            "trained_through": trained_through,
            "trained_at": pd_now_iso(),
        },
        path,
    )


def pd_now_iso() -> str:
    import datetime

    return datetime.datetime.now().isoformat(timespec="seconds")


def load(path: Path, device: str = "cpu") -> tuple[PLSTMTAL, Checkpoint]:
    raw = torch.load(path, map_location=device, weights_only=False)
    config = TrainConfig(**raw["config"])
    model = PLSTMTAL(
        input_size=len(raw["feature_columns"]),
        hidden_size=config.hidden_size,
        attn_size=config.attn_size,
        dropout=config.dropout,
        peephole=config.peephole,
        attention=config.attention,
    ).to(device)
    model.load_state_dict(raw["model_state"])
    model.eval()
    checkpoint = Checkpoint(
        config=config,
        feature_columns=tuple(raw["feature_columns"]),
        lookback=raw["lookback"],
        scaler_mean=np.asarray(raw["scaler_mean"], dtype=np.float32),
        scaler_std=np.asarray(raw["scaler_std"], dtype=np.float32),
        trained_through=raw["trained_through"],
        trained_at=raw["trained_at"],
    )
    return model, checkpoint
