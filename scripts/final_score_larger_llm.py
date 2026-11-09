"""Score the headline corpus with a larger local LLM (Qwen2.5-3B) for the model comparison.

Re-uses the *identical* prompt, labels and constrained-logit scoring as the
1.5B run, on the identical (already noise-filtered) headlines in
results/sentiment/scored_local.csv -- so the only thing that differs is the model.
Order: every earnings headline first, then the rest shuffled, so a partial run is
still a valid (stratified) comparison. Resumable via the SQLite LLM cache.

Usage: .venv/bin/python scripts/final_score_larger_llm.py [--limit N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.sentiment import score_headlines
from nse_agents.config import RESULTS, ROOT
from nse_agents.data.news import COMPANY_NAMES, NewsItem
from nse_agents.final.config import FINAL
from nse_agents.final.sentiment_eval import is_earnings
from nse_agents.llm.cache import CachedBackend
from nse_agents.llm.local_qwen import LocalQwenBackend

SNAP = ROOT / "hf_cache" / "models--Qwen--Qwen2.5-3B-Instruct" / "snapshots"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    base = pd.read_csv(RESULTS / "sentiment" / "scored_local.csv")
    base["earnings"] = base["title"].map(is_earnings)
    ordered = pd.concat([
        base[base["earnings"]],
        base[~base["earnings"]].sample(frac=1.0, random_state=20260909),
    ])
    if args.limit:
        ordered = ordered.head(args.limit)
    items = [NewsItem(r.symbol, str(r.date), r.title, "") for r in ordered.itertuples()]

    snapshot = next(SNAP.iterdir())
    inner = LocalQwenBackend(model_id=str(snapshot))
    inner.name = "Qwen2.5-3B-Instruct"     # cache key: never collides with the 1.5B rows
    backend = CachedBackend(inner)
    print(f"scoring {len(items):,} headlines with {inner.name} on {inner.device}", flush=True)
    t0 = time.time()
    # drop_noise=False: the input is already the 1.5B run's noise-filtered set.
    scored = score_headlines(backend, items, COMPANY_NAMES, batch_size=args.batch_size, drop_noise=False)
    scored.to_csv(FINAL / "scored_qwen3b.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s -> {FINAL / 'scored_qwen3b.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
