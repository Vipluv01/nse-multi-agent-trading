"""Optional hosted-Claude backend.

Activated only when ``ANTHROPIC_API_KEY`` is set; the study runs end to end
without it. Its purpose is the capability comparison: Lopez-Lira & Tang report
that headline-sentiment return predictability is absent in small models and
emerges only in larger ones, and the only way to test that here is to run the
identical prompts through a materially more capable model.
"""

from __future__ import annotations

import os

import numpy as np

from .base import LabelScores, LLMResponse

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

    def classify_batch(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[LabelScores]:
        """Structured, schema-enforced classification via native tool-calling.

        Not raw-logit scoring (the API exposes no logits) -- instead, a forced
        tool call whose schema requires ``reasoning`` before ``label`` and
        ``confidence``, so the confidence cannot be filled in independently of
        the stated reasoning: both come from the same generated turn, same
        temperature-0 call, same tool-call object. This is what "stance-text
        alignment" means in practice for an API model with no logit access --
        a two-call design (generate reasoning, then separately ask for a
        score) would let the two drift; this cannot.

        Sequential per-item calls, appropriate for the live pipeline's small
        batches (tens of items). Anthropic's async Batches API would be the
        right tool for rescoring a large historical corpus, not this call
        path -- swap it in there if that need arises.

        Unverified against a live API: this environment has no
        ANTHROPIC_API_KEY, so this method is implemented against the
        documented tool-use contract but has not been run end-to-end.
        """
        tool = {
            "name": "classify",
            "description": "Classify the input against the given labels.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": "Brief reasoning, one or two sentences, written before choosing a label."},
                    "label": {"type": "string", "enum": labels},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                                   "description": "Confidence in `label`, consistent with `reasoning`."},
                },
                "required": ["reasoning", "label", "confidence"],
            },
        }
        results: list[LabelScores] = []
        for user in users:
            message = self._client.messages.create(
                model=self.model, max_tokens=300, system=system, temperature=0.0,
                tools=[tool], tool_choice={"type": "tool", "name": "classify"},
                messages=[{"role": "user", "content": prefix + user}],
            )
            call = next((b for b in message.content if b.type == "tool_use"), None)
            if call is None:
                results.append(LabelScores(labels, [1.0 / len(labels)] * len(labels)))
                continue
            chosen = call.input.get("label", labels[0])
            confidence = float(np.clip(call.input.get("confidence", 1.0 / len(labels)), 0.0, 1.0))
            remainder = (1.0 - confidence) / max(len(labels) - 1, 1)
            probs = [confidence if lbl == chosen else remainder for lbl in labels]
            results.append(LabelScores(labels, probs))
        return results
