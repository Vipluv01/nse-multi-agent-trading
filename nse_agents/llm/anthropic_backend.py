"""Optional hosted-Claude backend.

Activated only when ``ANTHROPIC_API_KEY`` is set; the study runs end to end
without it. Its purpose is the capability comparison: Lopez-Lira & Tang report
that headline-sentiment return predictability is absent in small models and
emerges only in larger ones, and the only way to test that here is to run the
identical prompts through a materially more capable model.
"""

from __future__ import annotations

import os

from .base import LLMResponse

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicBackend:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Run with --backend local to use "
                "the offline Qwen backend instead."
            )
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the anthropic package is not installed: uv pip install -e '.[anthropic]'"
            ) from exc
        self._client = anthropic.Anthropic(api_key=key)
        self.model = model
        self.name = model

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            temperature=0.0,  # reproducibility over variety
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return LLMResponse(
            text=text.strip(),
            backend=self.name,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )
