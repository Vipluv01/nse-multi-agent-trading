"""Walk-forward training of the technical agent, across the architecture ablation.

Runs the 2x2 grid (peephole on/off) x (temporal attention on/off) through an
identical code path, so the comparison isolates the architecture rather than
two different implementations. Writes out-of-sample predictions per variant.

Usage:  .venv/bin/python scripts/train_technical.py [--epochs 25] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import RESULTS, SETTINGS
from nse_agents.data.dataset import build_panel_dataset
from nse_agents.models.train import TrainConfig, run_walk_forward

VARIANTS = [
    dict(peephole=True, attention=True),    # PLSTM-TAL, the paper's model
    dict(peephole=True, attention=False),   # isolates attention's contribution
    dict(peephole=False, attention=True),   # isolates the peephole connections
    dict(peephole=False, attention=False),  # plain LSTM floor
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(RESULTS / "technical"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_panel_dataset()
    print("dataset:", dataset.summary(), flush=True)

    manifest = []
    for variant in VARIANTS:
        for seed_offset in range(args.seeds):
            config = TrainConfig(epochs=args.epochs, **variant)
            name = f"{config.label()}_seed{seed_offset}"
            print(f"\n=== {name} ===", flush=True)
            started = time.time()
            # Seed enters through the trainer, so every variant sees the same
            # three initialisations -- paired across variants, not independent.
            import nse_agents.models.train as train_mod

            original = train_mod.train_one_fold

            def seeded(*a, **kw):
                kw["seed"] = SETTINGS.seed + seed_offset
                return original(*a, **kw)

            train_mod.train_one_fold = seeded
            try:
                oos, summaries = run_walk_forward(
                    dataset.x,
                    dataset.y,
                    dataset.dates,
                    dataset.symbols,
                    dataset.fwd_returns,
                    config,
                )
            finally:
                train_mod.train_one_fold = original

            oos.to_csv(out_dir / f"oos_{name}.csv", index=False)
            elapsed = time.time() - started
            accuracy = float(((oos["prob_up"] > 0.5) == (oos["y_true"] > 0.5)).mean())
            manifest.append(
                {
                    "name": name,
                    "architecture": config.label(),
                    "seed": SETTINGS.seed + seed_offset,
                    "oos_accuracy": accuracy,
                    "n_oos": len(oos),
                    "elapsed_s": round(elapsed, 1),
                    "folds": summaries,
                }
            )
            print(f"{name}: OOS accuracy {accuracy:.4f} over {len(oos):,} rows "
                  f"({elapsed:.0f}s)", flush=True)
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\n=== summary ===", flush=True)
    frame = pd.DataFrame(
        [{k: m[k] for k in ("name", "architecture", "oos_accuracy", "n_oos")} for m in manifest]
    )
    print(frame.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
