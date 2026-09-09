"""Backend selection in one place."""

from __future__ import annotations

import os

from .base import EchoBackend, LLMBackend
from .cache import CachedBackend


def build_backend(kind: str = "local", cache: bool = True, **kwargs) -> LLMBackend:
    if kind == "local":
        from .local_qwen import LocalQwenBackend

        backend: LLMBackend = LocalQwenBackend(**kwargs)
    elif kind == "anthropic":
        from .anthropic_backend import AnthropicBackend

        backend = AnthropicBackend(**kwargs)
    elif kind == "echo":
        backend = EchoBackend(**kwargs)
    else:
        raise ValueError(f"unknown backend {kind!r}; expected local|anthropic|echo")
    return CachedBackend(backend) if cache else backend


def available_backends() -> dict[str, bool]:
    return {
        "local": True,
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "echo": True,
    }
