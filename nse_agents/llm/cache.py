"""Content-addressed disk cache for LLM calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from ..config import LLM_CACHE
from .base import LabelScores, LLMBackend, LLMResponse


def _key(backend: str, system: str, user: str, max_tokens: int) -> str:
    blob = json.dumps([backend, system, user, max_tokens], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def _label_key(backend: str, system: str, user: str, labels: list[str], prefix: str) -> str:
    blob = json.dumps([backend, system, user, list(labels), prefix], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedBackend:
    """Wraps any backend with a SQLite-backed memo.

    SQLite rather than one file per call: a full run produces tens of thousands
    of entries, and a directory with 25,000 small files is slow to enumerate
    and unpleasant to move between machines.
    """

    def __init__(self, inner: LLMBackend, path: Path | None = None):
        self.inner = inner
        self.name = f"{inner.name}+cache"
        self.path = path or (LLM_CACHE / "responses.sqlite")
        self.hits = 0
        self.misses = 0
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            "  key TEXT PRIMARY KEY, backend TEXT, response TEXT,"
            "  prompt_tokens INTEGER, completion_tokens INTEGER)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS label_scores ("
            "  key TEXT PRIMARY KEY, backend TEXT, labels TEXT, probabilities TEXT)"
        )
        self._conn.commit()

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        key = _key(self.inner.name, system, user, max_tokens)
        row = self._conn.execute(
            "SELECT response, prompt_tokens, completion_tokens FROM responses WHERE key = ?",
            (key,),
        ).fetchone()
        if row is not None:
            self.hits += 1
            return LLMResponse(row[0], self.inner.name, cached=True,
                               prompt_tokens=row[1], completion_tokens=row[2])
        self.misses += 1
        response = self.inner.complete(system, user, max_tokens)
        self._conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?)",
            (key, self.inner.name, response.text,
             response.prompt_tokens, response.completion_tokens),
        )
        self._conn.commit()
        return response

    def classify_batch(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[LabelScores]:
        """Cached constrained scoring.

        Cached per *item*, not per batch, so a re-run with a different batch
        size or a partially-overlapping corpus still hits. Only the misses are
        forwarded to the wrapped backend, and the caller's order is restored --
        a debate pass over ~19,000 briefs is paid for once and replays free,
        which is what makes the LLM-in-the-loop backtest reproducible.
        """
        inner = self.inner
        if not hasattr(inner, "classify_batch"):
            raise AttributeError(
                f"backend {inner.name!r} does not support classify_batch"
            )

        keys = [_label_key(inner.name, system, u, labels, prefix) for u in users]
        results: list[LabelScores | None] = [None] * len(users)
        missing_positions: list[int] = []

        for position, key in enumerate(keys):
            row = self._conn.execute(
                "SELECT labels, probabilities FROM label_scores WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                missing_positions.append(position)
            else:
                self.hits += 1
                results[position] = LabelScores(json.loads(row[0]), json.loads(row[1]))

        if missing_positions:
            self.misses += len(missing_positions)
            fresh = inner.classify_batch(
                system, [users[i] for i in missing_positions], labels,
                prefix=prefix, batch_size=batch_size,
            )
            payload = []
            for position, scores in zip(missing_positions, fresh):
                results[position] = scores
                payload.append(
                    (keys[position], inner.name,
                     json.dumps(scores.labels), json.dumps(scores.probabilities))
                )
            self._conn.executemany(
                "INSERT OR REPLACE INTO label_scores VALUES (?,?,?,?)", payload
            )
            self._conn.commit()

        assert all(item is not None for item in results)
        return [item for item in results if item is not None]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "rows": self._conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0],
            "label_rows": self._conn.execute("SELECT COUNT(*) FROM label_scores").fetchone()[0],
        }
