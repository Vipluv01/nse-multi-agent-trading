"""Multi-provider ensemble scoring: combine several LLM backends' constrained-label
scores into one, with an inter-provider agreement metric and graceful fallback.

Implements the same ``classify_batch`` contract every other backend does
(``system, users, labels, prefix, batch_size -> list[LabelScores]``), so it is a
drop-in replacement anywhere a backend is used today -- ``SentimentAgent``'s
``score_headlines`` or the debate layer's ``run_debates`` need no changes to consume it.

**Not run against real multi-provider traffic.** Both providers this project has hosted
backends for (Anthropic, OpenAI) need API keys this environment doesn't have -- every
test here exercises the combination/fallback/agreement logic against stub backends,
the same boundary ``tests/test_hosted_backends.py`` already tests each backend at
individually.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from ..llm.base import LabelScores, LLMBackend


@dataclass(frozen=True)
class EnsembleResult:
    """One item's combined score, plus what produced it -- the audit trail a
    plain ``list[LabelScores]`` throws away."""

    combined: LabelScores
    agreement: float                    # 0 (total disagreement) .. 1 (identical)
    per_provider: dict[str, LabelScores]  # only the providers that actually answered
    failed_providers: list[str]


def _total_variation_agreement(a: LabelScores, b: LabelScores) -> float:
    """1 - total variation distance between two label distributions over the
    same label set. Bounded exactly [0, 1] (identical distributions -> 1,
    disjoint support -> 0) with no log/entropy edge cases to guard -- unlike
    e.g. Jensen-Shannon divergence, which needs care around zero
    probabilities. Chosen for that simplicity, not because it is the only
    reasonable choice.
    """
    pa, pb = a.as_dict(), b.as_dict()
    tv = 0.5 * sum(abs(pa.get(label, 0.0) - pb.get(label, 0.0)) for label in a.labels)
    return float(max(0.0, min(1.0, 1.0 - tv)))


def _mean_pairwise_agreement(scores: list[LabelScores]) -> float:
    """Mean agreement over all provider pairs -- the average overall
    consensus strength. (The more conservative alternative, ``min`` over all
    pairs, answers a different question -- "does any pair strongly disagree"
    -- and would be the right choice if a single outlier provider should
    dominate the down-weighting; ``mean`` was chosen so one provider
    disagreeing among three does not by itself collapse confidence to the
    same degree two-out-of-two disagreeing would.)
    """
    if len(scores) < 2:
        return 1.0  # a single surviving provider cannot disagree with itself
    pairs = [
        _total_variation_agreement(scores[i], scores[j])
        for i in range(len(scores))
        for j in range(i + 1, len(scores))
    ]
    return float(sum(pairs) / len(pairs))


def _average_distribution(scores: list[LabelScores]) -> LabelScores:
    labels = scores[0].labels
    combined = [
        sum(s.as_dict().get(label, 0.0) for s in scores) / len(scores) for label in labels
    ]
    return LabelScores(labels, combined)


class EnsembleBackend:
    """Wraps two or more ``classify_batch``-capable backends into one.

    Fallback is per-item, not per-batch: if provider A raises on the whole
    batch call (a rate limit, a transient network error), the remaining
    provider(s) are used for every item, not just the ones A happened to fail
    on -- most real backend failures (auth, rate limit, network) affect the
    entire call, not individual items within it, so per-batch fallback is both
    the simpler and the more realistic failure model here. If every provider
    fails, the error is not swallowed: silently returning a neutral guess
    would let a total outage masquerade as "the model is just uncertain
    today," which downstream confidence scoring cannot tell apart from a real
    low-conviction read.
    """

    def __init__(self, backends: dict[str, LLMBackend]):
        if len(backends) < 1:
            raise ValueError("EnsembleBackend needs at least one backend")
        self.backends = backends
        self.name = "ensemble(" + "+".join(backends) + ")"

    def classify_batch_with_agreement(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[EnsembleResult]:
        per_provider: dict[str, list[LabelScores]] = {}
        failed: list[str] = []

        for name, backend in self.backends.items():
            try:
                per_provider[name] = backend.classify_batch(
                    system, users, labels, prefix=prefix, batch_size=batch_size
                )
            except Exception as exc:  # noqa: BLE001 -- any provider failure must not crash the loop
                failed.append(name)
                print(f"EnsembleBackend: provider {name!r} failed ({type(exc).__name__}: {exc}); "
                      f"falling back to the remaining provider(s)", file=sys.stderr)

        survivors = list(per_provider)
        if not survivors:
            raise RuntimeError(
                f"EnsembleBackend: every provider failed ({failed}); no fallback left. "
                f"This is a real outage, not a low-confidence read -- it must not be "
                f"silently treated as one."
            )

        results = []
        for i in range(len(users)):
            item_scores = [per_provider[name][i] for name in survivors]
            results.append(
                EnsembleResult(
                    combined=_average_distribution(item_scores),
                    agreement=_mean_pairwise_agreement(item_scores),
                    per_provider={name: per_provider[name][i] for name in survivors},
                    failed_providers=list(failed),
                )
            )
        return results

    def classify_batch(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[LabelScores]:
        """The standard backend contract -- combined scores only. Use
        ``classify_batch_with_agreement`` for the agreement metric and the
        per-provider audit trail.
        """
        return [
            r.combined
            for r in self.classify_batch_with_agreement(system, users, labels, prefix, batch_size)
        ]

    def complete(self, system: str, user: str, max_tokens: int = 256):
        """Free-form generation: no combining to do, so this just tries
        providers in order and returns the first success -- there is no
        principled way to "average" two providers' prose the way there is
        for a label distribution.
        """
        last_exc: Exception | None = None
        for name, backend in self.backends.items():
            try:
                return backend.complete(system, user, max_tokens)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                print(f"EnsembleBackend: provider {name!r} failed on complete() "
                      f"({type(exc).__name__}: {exc}); trying next provider", file=sys.stderr)
        raise RuntimeError(f"EnsembleBackend: every provider failed on complete(): {last_exc}")


def confidence_penalty_from_agreement(agreement: float, floor: float = 0.3) -> float:
    """Map an agreement score to a confidence multiplier in [floor, 1.0].

    A separate, explicit function rather than folding this into
    ``SentimentAgent`` directly: the main study's confidence formula
    (dispersion x unknown-share x headline-count) is exactly what every
    already-published number in this project was computed with, and this
    project's discipline throughout has been that a new signal is opt-in, not
    a silent change to that formula. A caller building an ensemble-backed
    sentiment pipeline multiplies this into its own confidence; the existing
    single-provider pipeline is untouched.
    """
    return float(floor + (1.0 - floor) * max(0.0, min(1.0, agreement)))
