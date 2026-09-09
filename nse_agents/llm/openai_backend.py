"""Optional hosted GPT-4o backend.

Follows the same contract as ``AnthropicBackend`` -- activates only when
``OPENAI_API_KEY`` is set, so it is a second, independent frontier-capability
option alongside Claude, useful because the sentiment-emergence question this
project is asking ("does capability, not just scale, determine whether headline
sentiment predicts returns") is better answered by more than one capable model,
not tied to a single vendor.

Unverified against a live API: this environment has no OPENAI_API_KEY, so this
module is implemented against the documented Structured Outputs contract but
has not been run end-to-end. Same caveat as AnthropicBackend.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .base import LabelScores, LLMResponse

DEFAULT_MODEL = "gpt-4o"


class OpenAIBackend:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Run with --backend local to use "
                "the offline Qwen backend instead."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "the openai package is not installed: uv pip install -e '.[openai]'"
            ) from exc
        self._client = openai.OpenAI(api_key=key)
        self.model = model
        self.name = model

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        choice = response.choices[0].message
        return LLMResponse(
            text=(choice.content or "").strip(),
            backend=self.name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
        )

    def classify_batch(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[LabelScores]:
        """Structured classification via OpenAI's Structured Outputs (JSON schema).

        Same design as AnthropicBackend.classify_batch: the schema requires
        ``reasoning`` before ``label``/``confidence`` in one call, so the score
        cannot be generated independently of the stated reasoning. See that
        method's docstring for why this is the right shape for "stance-text
        alignment" with an API model that exposes no raw logits.
        """
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification",
                "schema": {
                    "type": "object",
                    "properties": {
                        "reasoning": {"type": "string"},
                        "label": {"type": "string", "enum": labels},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["reasoning", "label", "confidence"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
        results: list[LabelScores] = []
        for user in users:
            response = self._client.chat.completions.create(
                model=self.model, max_tokens=300, temperature=0.0,
                response_format=schema,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prefix + user}],
            )
            try:
                parsed = json.loads(response.choices[0].message.content)
                chosen = parsed.get("label", labels[0])
                confidence = float(np.clip(parsed.get("confidence", 1.0 / len(labels)), 0.0, 1.0))
            except (json.JSONDecodeError, AttributeError):
                chosen, confidence = labels[0], 1.0 / len(labels)
            remainder = (1.0 - confidence) / max(len(labels) - 1, 1)
            probs = [confidence if lbl == chosen else remainder for lbl in labels]
            results.append(LabelScores(labels, probs))
        return results
