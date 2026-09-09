"""Local Qwen2.5-Instruct backend.

Two modes, for two different jobs:

* ``classify_batch`` -- the workhorse. Instead of generating text and hoping to
  parse it, it runs one forward pass and reads the logits of the candidate
  label tokens at the first generated position, renormalising over just those
  labels. For a 1.5B model this is the difference between a usable experiment
  and one where a third of the outputs are unparseable prose. It is also
  ~50x faster (one forward pass, batched) and exactly deterministic, which
  matters when the same corpus is scored repeatedly across ablations.
* ``complete`` -- free generation, used only by the debate agents, where the
  natural-language rationale *is* the deliverable.

Runs on MPS when available. Note this is inference only: LoRA-style training on
this machine's MPS backend is a known-bad path and nothing here needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import LabelScores, LLMResponse

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class LocalQwenBackend:
    """transformers-backed local model. Loads lazily so importing is cheap."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_id = model_id
        self.name = model_id.split("/")[-1]
        self.device = pick_device(device)
        self.dtype = dtype or (torch.float32 if self.device == "cpu" else torch.float16)
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, padding_side="left")
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=self.dtype
            ).to(self.device)
            self._model.eval()
        return self._model, self._tokenizer

    def _chat_prompt(self, system: str, user: str, prefix: str = "") -> str:
        _, tokenizer = self._load()
        text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return text + prefix

    def _label_token_ids(self, labels: list[str]) -> list[int]:
        """First token id of each label, which is what the logits are read at."""
        _, tokenizer = self._load()
        ids = []
        for label in labels:
            encoded = tokenizer.encode(label, add_special_tokens=False)
            if not encoded:
                raise ValueError(f"label {label!r} encodes to nothing")
            ids.append(encoded[0])
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"labels {labels} share a first token; they cannot be distinguished "
                "by first-token logits -- choose labels with distinct openings"
            )
        return ids

    @torch.no_grad()
    def classify_batch(
        self,
        system: str,
        users: list[str],
        labels: list[str],
        prefix: str = "",
        batch_size: int = 16,
    ) -> list[LabelScores]:
        model, tokenizer = self._load()
        label_ids = self._label_token_ids(labels)
        all_prompts = [self._chat_prompt(system, u, prefix) for u in users]

        # Batch similar lengths together so padding is not carried through the
        # forward pass, then restore the caller's order. Results are unaffected:
        # each row is independent, and left padding keeps the final position
        # aligned to each sequence's last real token.
        order = sorted(range(len(all_prompts)), key=lambda i: len(all_prompts[i]))
        scored: list[LabelScores | None] = [None] * len(all_prompts)

        for start in range(0, len(order), batch_size):
            positions = order[start : start + batch_size]
            prompts = [all_prompts[i] for i in positions]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                max_length=1024).to(self.device)
            # ``logits_to_keep=1`` runs the LM head on the final position only.
            # Without it the head projects every position to the full 152k
            # vocabulary -- for a batch of 16 x ~150 tokens that is a 364M-element
            # tensor computed and discarded, and it dominates the runtime.
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
            selected = logits[:, label_ids]
            probs = torch.softmax(selected, dim=-1).cpu().tolist()
            for position, prob in zip(positions, probs):
                scored[position] = LabelScores(labels, prob)

        assert all(item is not None for item in scored)
        return [item for item in scored if item is not None]

    @torch.no_grad()
    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        model, tokenizer = self._load()
        prompt = self._chat_prompt(system, user)
        encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_tokens,
            do_sample=False,          # greedy: the study needs reproducibility
            pad_token_id=tokenizer.pad_token_id,
        )
        text = tokenizer.decode(generated[0][encoded["input_ids"].shape[1]:], skip_special_tokens=True)
        return LLMResponse(
            text=text.strip(),
            backend=self.name,
            prompt_tokens=int(encoded["input_ids"].shape[1]),
            completion_tokens=int(generated.shape[1] - encoded["input_ids"].shape[1]),
        )
