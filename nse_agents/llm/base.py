"""Backend-agnostic LLM interface.

Every agent talks to this protocol, never to a vendor SDK. Two consequences
that matter for the study rather than just for tidiness:

* The local and hosted backends are **swapped by configuration**, so the
  Lopez-Lira & Tang claim -- that headline-sentiment return predictability
  appears only in more capable models and is absent in small ones -- can be
  tested on this data by changing one flag, instead of being cited.
* Every call goes through a content-addressed disk cache, so a backtest that
  scores 25,000 headlines is paid for once and then replays deterministically.
  Without it, no LLM-in-the-loop backtest is reproducible or affordable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class LabelScores:
    """A probability distribution over a fixed set of candidate labels.

    Lives here rather than beside the local model so that the cache and the
    agents can handle scores without importing torch.
    """

    labels: list[str]
    probabilities: list[float]

    def top(self) -> str:
        best = max(range(len(self.probabilities)), key=lambda i: self.probabilities[i])
        return self.labels[best]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.labels, self.probabilities))


@dataclass(frozen=True)
class LLMResponse:
    text: str
    backend: str
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse: ...


class EchoBackend:
    """Deterministic stand-in used by the test suite.

    Returns fixed, parseable payloads so that agent orchestration, caching and
    the backtest plumbing can be tested without a model or a network.
    """

    name = "echo"

    def __init__(self, reply: str = '{"label": "neutral", "score": 0.0, "reason": "stub"}'):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []
        self.classify_calls: list[str] = []

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text=self.reply, backend=self.name)

    def classify_batch(
        self, system: str, users: list[str], labels: list[str],
        prefix: str = "", batch_size: int = 16,
    ) -> list[LabelScores]:
        """Deterministic pseudo-scores derived from the prompt text.

        Varying with the prompt (rather than returning a constant) is what lets
        a cache test tell a hit from a miss.
        """
        out = []
        for user in users:
            self.classify_calls.append(user)
            weights = [((hash((user, label)) % 1000) + 1) / 1000.0 for label in labels]
            total = sum(weights)
            out.append(LabelScores(list(labels), [w / total for w in weights]))
        return out
