"""Build the headline corpus. Resumable: re-running only fetches missing months.

Usage:  .venv/bin/python scripts/fetch_news.py [--start 2016-01-01] [--symbols RELIANCE,TCS]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.config import SETTINGS
from nse_agents.data.news import (
    NEWS_CACHE,
    CachedCorpus,
    GoogleNewsRSS,
    coverage_report,
    write_corpus,
)

DONE = NEWS_CACHE / "fetched_months.json"  # overridden per shard in main()


def load_done() -> set[str]:
    return set(json.loads(DONE.read_text())) if DONE.exists() else set()


def save_done(done: set[str]) -> None:
    DONE.write_text(json.dumps(sorted(done)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default=SETTINGS.end)
    ap.add_argument("--symbols", default=",".join(SETTINGS.universe))
    ap.add_argument("--pause", type=float, default=0.8)
    ap.add_argument("--tag", default="", help="shard id; lets workers run in parallel "
                                              "without racing on the same state files")
    args = ap.parse_args()

    # Each worker owns its own done-file and corpus shard. Merging afterwards is
    # cheap; coordinating concurrent writes to one file is not.
    global DONE
    suffix = f"_{args.tag}" if args.tag else ""
    DONE = NEWS_CACHE / f"fetched_months{suffix}.json"
    corpus_path = NEWS_CACHE / f"headlines{suffix}.jsonl"

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    months = pd.date_range(args.start, args.end, freq="MS")
    provider = GoogleNewsRSS(pause=args.pause)
    done = load_done()

    total_new = 0
    for sym in symbols:
        for month in months:
            key = f"{sym}:{month.date().isoformat()}"
            if key in done:
                continue
            nxt = month + pd.offsets.MonthBegin(1)
            for attempt in range(4):
                try:
                    items = provider.fetch(
                        sym, month.date().isoformat(), nxt.date().isoformat()
                    )
                    break
                except Exception as exc:  # rate limit or transient network
                    wait = 15 * (attempt + 1)
                    print(f"  retry {sym} {month.date()} in {wait}s ({exc})", flush=True)
                    time.sleep(wait)
            else:
                print(f"  SKIP {sym} {month.date()} after 4 attempts", flush=True)
                continue

            if items:
                write_corpus(items, corpus_path)
                total_new += len(items)
            done.add(key)
            save_done(done)
            print(f"{sym} {month.date()}: +{len(items)} (total {total_new})", flush=True)

    corpus = CachedCorpus(corpus_path).all_items()
    print("\n=== corpus coverage ===", flush=True)
    print(coverage_report(corpus).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
