"""Tests for the Anthropic/OpenAI structured-output classify_batch implementations.

Neither backend can be exercised against a live API in this environment (no
ANTHROPIC_API_KEY or OPENAI_API_KEY), so these tests mock the client at the
documented response-shape boundary and verify the parsing and confidence-to-
probability conversion logic -- the part that is actually this project's code,
as opposed to the vendor SDK's own tested behaviour.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_anthropic_backend_refuses_without_a_key(monkeypatch):
    from nse_agents.llm.anthropic_backend import AnthropicBackend

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicBackend()


def test_openai_backend_refuses_without_a_key(monkeypatch):
    from nse_agents.llm.openai_backend import OpenAIBackend

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIBackend()


def test_anthropic_classify_batch_parses_a_forced_tool_call():
    """The tool-call's label/confidence must convert to a probability
    distribution that sums to 1 and puts the stated confidence on the chosen
    label -- the exact mechanism 'stance-text alignment' depends on."""
    from nse_agents.llm.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend.__new__(AnthropicBackend)  # bypass __init__ (no key needed)
    backend.model = "claude-sonnet-5"
    backend.name = "claude-sonnet-5"

    tool_call = SimpleNamespace(
        type="tool_use",
        input={"reasoning": "Clearly positive for the stock.", "label": "Good", "confidence": 0.82},
    )
    response = SimpleNamespace(content=[tool_call])

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs["tool_choice"] == {"type": "tool", "name": "classify"}
            return response

    backend._client = SimpleNamespace(messages=FakeMessages())

    scores = backend.classify_batch("sys", ["Reliance profit jumps 20%"], ["Good", "Bad", "Unknown"])
    assert len(scores) == 1
    d = scores[0].as_dict()
    assert d["Good"] == pytest.approx(0.82)
    assert sum(d.values()) == pytest.approx(1.0)
    assert scores[0].top() == "Good"


def test_anthropic_classify_batch_falls_back_safely_with_no_tool_call():
    """If the model somehow returns text instead of the forced tool call,
    this must degrade to a neutral distribution, not crash the pipeline."""
    from nse_agents.llm.anthropic_backend import AnthropicBackend

    backend = AnthropicBackend.__new__(AnthropicBackend)
    backend.model = "claude-sonnet-5"
    backend.name = "claude-sonnet-5"
    text_block = SimpleNamespace(type="text", text="oops")
    response = SimpleNamespace(content=[text_block])
    backend._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response))

    scores = backend.classify_batch("sys", ["headline"], ["Good", "Bad", "Unknown"])
    assert len(scores) == 1
    assert sum(scores[0].probabilities) == pytest.approx(1.0)


def test_openai_classify_batch_parses_structured_json():
    from nse_agents.llm.openai_backend import OpenAIBackend
    import json

    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.model = "gpt-4o"
    backend.name = "gpt-4o"

    payload = json.dumps({"reasoning": "Negative outlook.", "label": "Bad", "confidence": 0.7})
    message = SimpleNamespace(content=payload)
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )

    scores = backend.classify_batch("sys", ["headline"], ["Good", "Bad", "Unknown"])
    d = scores[0].as_dict()
    assert d["Bad"] == pytest.approx(0.7)
    assert sum(d.values()) == pytest.approx(1.0)


def test_openai_classify_batch_falls_back_on_malformed_json():
    from nse_agents.llm.openai_backend import OpenAIBackend

    backend = OpenAIBackend.__new__(OpenAIBackend)
    backend.model = "gpt-4o"
    backend.name = "gpt-4o"
    message = SimpleNamespace(content="not json")
    choice = SimpleNamespace(message=message)
    response = SimpleNamespace(choices=[choice])
    backend._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))
    )

    scores = backend.classify_batch("sys", ["headline"], ["Good", "Bad", "Unknown"])
    assert sum(scores[0].probabilities) == pytest.approx(1.0)
