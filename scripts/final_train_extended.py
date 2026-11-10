"""Walk-forward technical training on the extended universe/period (final evaluation).

Same model code, same purged/embargoed walk-forward, same seeds as the
mid-semester run -- only the universe (10 -> 30 names) and the end date
(Sep -> Nov 2026) change. Writes to results/final/extended/technical.

Usage: .venv/bin/python scripts/final_train_extended.py [--epochs 25] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.final.config import EXT_END, EXT_START, EXT_UNIVERSE, FINAL, use_extended_cache

use_extended_cache()

import torch  # noqa: E402

import nse_agents.models.train as train_mod  # noqa: E402
from nse_agents.config import SETTINGS  # noqa: E402
from nse_agents.data.dataset import build_panel_dataset  # noqa: E402
from nse_agents.models.train import TrainConfig, run_walk_forward  # noqa: E402

VARIANTS = [dict(peephole=True, attention=True), dict(peephole=False, attention=True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = FINAL / "extended" / "technical"
    out.mkdir(parents=True, exist_ok=True)

    dataset = build_panel_dataset(EXT_UNIVERSE, EXT_START, EXT_END)
    print("dataset:", dataset.summary(), flush=True)
    manifest = []
    for variant in VARIANTS:
        for seed_offset in range(args.seeds):
            config = TrainConfig(epochs=args.epochs, **variant)
            name = f"{config.label()}_seed{seed_offset}"
            if (out / f"oos_{name}.csv").exists():
                print("skip (exists):", name, flush=True)
                continue
            original = train_mod.train_one_fold

            def seeded(*a, _orig=original, _off=seed_offset, **kw):
                kw["seed"] = SETTINGS.seed + _off
                return _orig(*a, **kw)

            train_mod.train_one_fold = seeded
            t0 = time.time()
            try:
                oos, summaries = run_walk_forward(dataset.x, dataset.y, dataset.dates, dataset.symbols,
                                                  dataset.fwd_returns, config)
            finally:
                train_mod.train_one_fold = original
            oos.to_csv(out / f"oos_{name}.csv", index=False)
            acc = float(((oos["prob_up"] > 0.5) == (oos["y_true"] > 0.5)).mean())
            manifest.append({"name": name, "architecture": config.label(), "oos_accuracy": acc,
                             "n_oos": len(oos), "elapsed_s": round(time.time() - t0, 1)})
            print(f"{name}: OOS accuracy {acc:.4f} over {len(oos):,} rows ({time.time() - t0:.0f}s)", flush=True)
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
