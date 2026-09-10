"""Tests for EnsembleBackend: combination, agreement, and fallback logic.

No live API traffic -- exercises the same combination/fallback boundary
tests/test_hosted_backends.py mocks each individual backend at.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nse_agents.agents.ensemble import (
    EnsembleBackend,
    confidence_penalty_from_agreement,
)
from nse_agents.llm.base import LabelScores

LABELS = ["Good", "Bad", "Unknown"]


class _Stub:
    def __init__(self, name, scores=None, raises=None):
        self.name = name
        self._scores = scores
        self._raises = raises
        self.calls = 0

    def classify_batch(self, system, users, labels, prefix="", batch_size=16):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        return self._scores

    def complete(self, system, user, max_tokens=256):
        self.calls += 1
        if self._raises is not None:
            raise self._raises
        from nse_agents.llm.base import LLMResponse

        return LLMResponse(text=f"reply from {self.name}", backend=self.name)


def test_identical_distributions_give_perfect_agreement():
    scores = [LabelScores(LABELS, [0.7, 0.2, 0.1])]
    ens = EnsembleBackend({"a": _Stub("a", scores), "b": _Stub("b", scores)})
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    assert result.agreement == pytest.approx(1.0)


def test_disjoint_distributions_give_zero_agreement():
    a_scores = [LabelScores(LABELS, [1.0, 0.0, 0.0])]
    b_scores = [LabelScores(LABELS, [0.0, 1.0, 0.0])]
    ens = EnsembleBackend({"a": _Stub("a", a_scores), "b": _Stub("b", b_scores)})
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    assert result.agreement == pytest.approx(0.0)


def test_combined_score_is_the_average_distribution():
    a_scores = [LabelScores(LABELS, [0.8, 0.1, 0.1])]
    b_scores = [LabelScores(LABELS, [0.4, 0.5, 0.1])]
    ens = EnsembleBackend({"a": _Stub("a", a_scores), "b": _Stub("b", b_scores)})
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    d = result.combined.as_dict()
    assert d["Good"] == pytest.approx(0.6)
    assert d["Bad"] == pytest.approx(0.3)
    assert sum(d.values()) == pytest.approx(1.0)


def test_one_provider_failing_falls_back_without_crashing():
    good_scores = [LabelScores(LABELS, [0.9, 0.05, 0.05])]
    ens = EnsembleBackend({
        "broken": _Stub("broken", raises=RuntimeError("rate limited")),
        "ok": _Stub("ok", good_scores),
    })
    results = ens.classify_batch("sys", ["h"], LABELS)  # must not raise
    assert results[0].as_dict()["Good"] == pytest.approx(0.9)


def test_fallback_result_records_which_provider_failed():
    ens = EnsembleBackend({
        "broken": _Stub("broken", raises=RuntimeError("boom")),
        "ok": _Stub("ok", [LabelScores(LABELS, [0.5, 0.3, 0.2])]),
    })
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    assert result.failed_providers == ["broken"]
    assert set(result.per_provider) == {"ok"}


def test_single_surviving_provider_reports_full_agreement():
    """Agreement with yourself is undefined, not zero -- a lone surviving
    provider must not be penalised as if it disagreed with a ghost."""
    ens = EnsembleBackend({
        "broken": _Stub("broken", raises=RuntimeError("boom")),
        "ok": _Stub("ok", [LabelScores(LABELS, [0.5, 0.3, 0.2])]),
    })
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    assert result.agreement == pytest.approx(1.0)


def test_every_provider_failing_raises_not_silently_guesses():
    ens = EnsembleBackend({
        "a": _Stub("a", raises=RuntimeError("down")),
        "b": _Stub("b", raises=RuntimeError("also down")),
    })
    with pytest.raises(RuntimeError, match="every provider failed"):
        ens.classify_batch("sys", ["h"], LABELS)


def test_complete_falls_back_to_the_next_provider_in_order():
    ens = EnsembleBackend({
        "broken": _Stub("broken", raises=RuntimeError("down")),
        "ok": _Stub("ok"),
    })
    response = ens.complete("sys", "user prompt")
    assert response.backend == "ok"


def test_complete_raises_when_every_provider_fails():
    ens = EnsembleBackend({"a": _Stub("a", raises=RuntimeError("x"))})
    with pytest.raises(RuntimeError, match="every provider failed"):
        ens.complete("sys", "user prompt")


def test_ensemble_requires_at_least_one_backend():
    with pytest.raises(ValueError):
        EnsembleBackend({})


def test_confidence_penalty_is_monotonic_in_agreement():
    low = confidence_penalty_from_agreement(0.0)
    high = confidence_penalty_from_agreement(1.0)
    assert low < high
    assert high == pytest.approx(1.0)


def test_confidence_penalty_never_goes_below_the_floor():
    penalty = confidence_penalty_from_agreement(0.0, floor=0.3)
    assert penalty == pytest.approx(0.3)


def test_three_providers_uses_mean_pairwise_not_worst_pair():
    """Two agree, one is an outlier -- mean pairwise agreement should be
    higher than the single most-disagreeing pair, distinguishing it from a
    min-based metric."""
    agree_1 = [LabelScores(LABELS, [0.9, 0.05, 0.05])]
    agree_2 = [LabelScores(LABELS, [0.85, 0.1, 0.05])]
    outlier = [LabelScores(LABELS, [0.05, 0.9, 0.05])]
    ens = EnsembleBackend({
        "a": _Stub("a", agree_1), "b": _Stub("b", agree_2), "c": _Stub("c", outlier),
    })
    result = ens.classify_batch_with_agreement("sys", ["h"], LABELS)[0]
    worst_pair_agreement = 1.0 - 0.5 * sum(
        abs(d1 - d2) for d1, d2 in zip(agree_1[0].probabilities, outlier[0].probabilities)
    )
    assert result.agreement > worst_pair_agreement
