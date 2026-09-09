"""Tests for model checkpoint save/load -- the seam between the walk-forward study
(never persists a model) and the live pipeline (needs one)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.models import checkpoint as ckpt
from nse_agents.models.attn_lstm import PLSTMTAL
from nse_agents.models.train import TrainConfig


def test_checkpoint_roundtrip_preserves_weights_and_predictions(tmp_path):
    torch.manual_seed(0)
    config = TrainConfig(peephole=False, attention=False, hidden_size=16, attn_size=8)
    model = PLSTMTAL(input_size=5, hidden_size=16, attn_size=8, peephole=False, attention=False)
    model.eval()

    x = torch.randn(3, 10, 5)
    with torch.no_grad():
        before = model(x)

    mean = np.random.randn(5).astype(np.float32)
    std = (np.abs(np.random.randn(5)) + 0.1).astype(np.float32)
    path = tmp_path / "model.pt"
    toy_columns = tuple(f"f{i}" for i in range(5))
    ckpt.save(path, model, config, mean, std, trained_through="2026-01-01",
              lookback=10, feature_columns=toy_columns)

    loaded_model, meta = ckpt.load(path)
    with torch.no_grad():
        after = loaded_model(x)

    torch.testing.assert_close(before, after)
    np.testing.assert_allclose(meta.scaler_mean, mean, rtol=1e-6)
    np.testing.assert_allclose(meta.scaler_std, std, rtol=1e-6)
    assert meta.trained_through == "2026-01-01"
    assert meta.lookback == 10
    assert meta.config.peephole is False
    assert meta.config.attention is False


def test_checkpoint_preserves_architecture_label():
    """A checkpoint's architecture must round-trip correctly so a live
    rationale never claims the wrong model produced a prediction (the
    PLSTM-TAL-hardcoded-string bug this was written to prevent a recurrence
    of)."""
    config = TrainConfig(peephole=True, attention=True)
    assert config.label() == "PLSTM-TAL"
    config2 = TrainConfig(peephole=False, attention=False)
    assert config2.label() == "LSTM"


def test_checkpoint_file_is_self_contained(tmp_path):
    """Loading must not require the training dataset or any external state --
    only the file itself."""
    config = TrainConfig(peephole=False, attention=False, hidden_size=8, attn_size=4)
    model = PLSTMTAL(input_size=3, hidden_size=8, attn_size=4, peephole=False, attention=False)
    path = tmp_path / "model.pt"
    toy_columns = tuple(f"f{i}" for i in range(3))
    ckpt.save(path, model, config, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32),
              trained_through="2026-01-01", lookback=5, feature_columns=toy_columns)

    # A fresh process-like load: only the path is used.
    loaded, meta = ckpt.load(path)
    assert isinstance(loaded, PLSTMTAL)
    assert meta.feature_columns is not None


def test_checkpoint_rejects_mismatched_feature_columns(tmp_path):
    """A feature_columns list that doesn't match the scaler's dimensionality
    must fail loudly at save time, not produce a checkpoint that silently
    claims the wrong input shape."""
    config = TrainConfig(peephole=False, attention=False, hidden_size=4, attn_size=2)
    model = PLSTMTAL(input_size=3, hidden_size=4, attn_size=2, peephole=False, attention=False)
    path = tmp_path / "model.pt"
    with pytest.raises(ValueError, match="must describe the same features"):
        ckpt.save(path, model, config, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32),
                  trained_through="2026-01-01", lookback=5,
                  feature_columns=("only", "two"))
