"""Merge the per-worker headline shards into the single corpus the agents read."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nse_agents.data.news import NEWS_CACHE, CachedCorpus, coverage_report, write_corpus

shards = sorted(NEWS_CACHE.glob("headlines_*.jsonl"))
items = []
for shard in shards:
    got = CachedCorpus(shard).all_items()
    items.extend(got)
    print(f"{shard.name}: {len(got):,}")
write_corpus(items, NEWS_CACHE / "headlines.jsonl")
corpus = CachedCorpus().all_items()
print(f"\nmerged corpus: {len(corpus):,} unique headlines")
print(coverage_report(corpus).to_string(index=False))
