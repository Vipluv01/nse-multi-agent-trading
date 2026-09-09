"""Train ONE final model on all available history, for live use only.

This is not part of the evaluation. Every accuracy/Sharpe number reported anywhere in
this project comes from a walk-forward model that never saw its own test period; this
script trains on everything through the most recent available trading day specifically
because there is no future left to hold out -- today's outcome doesn't exist yet.

The checkpoint this produces carries no accuracy claim of its own. Its out-of-sample
performance is, by construction, whatever the walk-forward study already measured for
this architecture (LSTM: 50.45% accuracy, p=0.21 vs chance -- see README). Training on
more data than any single fold saw does not and cannot improve on that; it only lets
the model make a prediction for a date the study itself could never evaluate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.dataset import build_panel_dataset
from nse_agents.models import checkpoint as ckpt
from nse_agents.models.attn_lstm import PLSTMTAL
from nse_agents.models.train import TrainConfig, _standardise

OUT = RESULTS / "production"


def main() -> int:
    torch.manual_seed(SETTINGS.seed)
    np.random.seed(SETTINGS.seed)

    dataset = build_panel_dataset()  # LSTM's own architecture, default horizon=1
    print(f"training on {len(dataset):,} sequences through "
          f"{dataset.dates.max().date()}", flush=True)

    # Best empirical architecture from the walk-forward study (README: plain LSTM,
    # 50.45% OOS accuracy, highest of the four tested) -- not re-selected here, just
    # reused, so this script cannot be accused of picking a flattering architecture
    # after the fact.
    config = TrainConfig(peephole=False, attention=False, epochs=25)

    n_val = max(1, int(len(dataset) * 0.10))
    x_fit, y_fit = dataset.x[:-n_val], dataset.y[:-n_val]
    x_val, y_val = dataset.x[-n_val:], dataset.y[-n_val:]
    x_fit_s, x_val_s = _standardise(x_fit, x_val)
    # Recover the mean/std _standardise used, for the checkpoint (needed to
    # normalise live feature windows identically at inference time).
    flat = x_fit.reshape(-1, x_fit.shape[-1])
    mean, std = flat.mean(axis=0), flat.std(axis=0)
    std[std < 1e-8] = 1.0

    model = PLSTMTAL(
        input_size=x_fit.shape[-1], hidden_size=config.hidden_size,
        attn_size=config.attn_size, dropout=config.dropout,
        peephole=config.peephole, attention=config.attention,
    )
    pos = float(y_fit.sum())
    pos_weight = torch.tensor([(len(y_fit) - pos) / max(pos, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    xt, yt = torch.from_numpy(x_fit_s), torch.from_numpy(y_fit)
    xv, yv = torch.from_numpy(x_val_s), torch.from_numpy(y_val)
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    generator = torch.Generator().manual_seed(SETTINGS.seed)

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), config.batch_size):
            batch = order[start:start + config.batch_size]
            optimiser.zero_grad()
            loss = criterion(model(xt[batch]), yt[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(xv), yv))
        if val_loss < best_loss - 1e-5:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
        print(f"  epoch {epoch:2d}  val_loss={val_loss:.4f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    path = OUT / "technical_model.pt"
    ckpt.save(
        path, model, config, mean, std,
        trained_through=str(dataset.dates.max().date()),
    )
    print(f"\nsaved checkpoint to {path}", flush=True)
    print("NOTE: this checkpoint has no held-out accuracy of its own -- see the module\n"
          "docstring. Its expected performance is whatever the walk-forward study\n"
          "already measured for this architecture, not a new or better number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
